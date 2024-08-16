import argparse

import torch
from accelerate.hooks import remove_hook_from_submodules

from finetune import print_memory_stats
from main import perplexity_eval
from src.datautils import get_loaders
from src.modelutils import get_model
from src.aq import QuantizedWeight
from noise import NoisyHadamarLinear
from lm_eval import evaluator
import lm_eval.models.huggingface
import lm_eval.tasks
import requests


def add_noisy_layers(model, relative_mse):
    for child_name, child in model.named_children():
        if not isinstance(child, torch.nn.Linear):
            add_noisy_layers(child, relative_mse)
            continue

        new_linear = NoisyHadamarLinear(child.weight, child.bias, had_block_size=64, relative_mse=relative_mse)
        setattr(model, child_name, new_linear)

    return model


def main():
    parser = argparse.ArgumentParser(add_help=True)
    # Model params
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["arc_easy",],
        help="Tasks to run evaluation on",
    )
    parser.add_argument(
        "--effective_wbits",
        type=float,
        default=1.0,
    )
    # Misc params
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--num_fewshots",
        type=int,
        default=1,
        help="number of fewshots for tasks",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=[None, "cpu", "auto"],
        help="accelerate device map",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    parser.add_argument(
        "--eval_base",
        action="store_true",
        help="Whether to eval base model.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to log to wandb.",
    )
    args = parser.parse_args()

    if args.wandb:
        import wandb

        wandb.init(config=vars(args))

    # get device
    assert torch.cuda.is_available()
    device = "cuda"
    args.devices = [device]  # needed for perplexity eval

    model = get_model(args.base_model, None, args.dtype, args.device_map,
                               trust_remote_code=args.trust_remote_code)
    if not args.device_map:
        model = model.to(device)

    if args.effective_wbits == -1.0:
        relative_mse = 0.0
    else:
        relative_mse = 4 ** (-args.effective_wbits)

    print(f'{relative_mse=}')
    if args.wandb:
        wandb.log({"relative_mse": relative_mse})

    add_noisy_layers(model.model.layers, relative_mse=relative_mse)

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
    )

    tasks = lm_eval.tasks.get_task_dict(args.tasks)
    if args.num_fewshots != 1:
        for task_name in tasks:
            task = tasks[task_name]
            if isinstance(task, tuple):
                task = task[1]
            if task is None:
                continue
            task.config.num_fewshot = args.num_fewshots

    results = evaluator.evaluate(
        lm=lm_eval_model,
        task_dict=lm_eval.tasks.get_task_dict(args.tasks),
    )

    result_dict = {task_name: task_result['acc,none'] for task_name, task_result in results['results'].items()}
    result_err_dict = {f'{task_name}_err': task_result['acc_stderr,none'] for task_name, task_result in results['results'].items()}
    result_dict = dict(list(result_dict.items()) + list(result_err_dict.items()))

    if args.num_fewshots != 1:
        result_dict = {f'{task_name}@{args.num_fewshots}': acc for task_name, acc in result_dict.items()}

    for task_name, acc in result_dict.items():
        print(f'{task_name}: {acc}')

    if args.wandb:
        wandb.log(result_dict)


if __name__ == '__main__':
    while True:
        try:
            main()
            break
        except requests.exceptions.SSLError:
            pass
