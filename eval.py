import argparse

import torch
from accelerate.hooks import remove_hook_from_submodules

from finetune import print_memory_stats
from main import perplexity_eval
from src.datautils import get_loaders
from src.modelutils import get_model
from src.aq import QuantizedWeight
from noise import NoisyHadamarLinear


def add_noisy_layers(model, relative_mse):
    for child_name, child in model.named_children():
        if not isinstance(child, torch.nn.Linear):
            add_noisy_layers(child, relative_mse)
            continue

        new_linear = NoisyHadamarLinear(child.weight, child.bias, had_block_size=64, relative_mse=relative_mse)
        setattr(model, child_name, new_linear)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    # Model params
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4"],
        help="Datasets to run evaluation on",
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

    orig_model = get_model(args.base_model, None, args.dtype, args.device_map,
                               trust_remote_code=args.trust_remote_code)
    if not args.device_map:
        orig_model = orig_model.to(device)

    relative_mse = 4 ** (-args.effective_wbits)

    add_noisy_layers(orig_model.model.layers, relative_mse=relative_mse)

    orig_model.save('model.py')

