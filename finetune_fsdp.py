"""Early prototype of FSDP fine-tuning; TODO clean-up imports"""
import argparse
import torch.distributed
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel

from src.aq import QuantizedWeight
from src.aq_ops import IntCodes
from src.modelutils import get_model

try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False



def add_finetuning_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="path to quantized model",
    )
    # Data params
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="number of samples",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--eval_model_seqlen",
        type=int,
        default=None,
        help="Model seqlen on validation. By default is equal to model_seqlen.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="size of validation split",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4"],
        help="Datasets to run evaluation on",
    )
    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="finetuning learning rate",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.90,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="Adam beta2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to apply gradient checkpointing",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Whether to use amp",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized statistics.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    add_finetuning_args(parser)
    args = parser.parse_args()

    args.microbatch_size = args.microbatch_size or args.batch_size
    if args.amp:
        assert args.finetune_dtype == torch.float32, "AMP works only with original model in fp32."

    assert torch.cuda.is_available() and torch.distributed.is_available()
    torch.distributed.init_process_group()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    base_model = get_model(
        args.base_model, load_quantized=None, dtype=args.dtype, trust_remote_code=args.trust_remote_code
    )
    base_model = FullyShardedDataParallel(base_model, device_id=device)

    quantized_model = get_model(
        args.base_model, args.quantized_model, dtype=args.dtype, trust_remote_code=args.trust_remote_code
    )
    if args.dtype != 'auto':
        quantized_model = quantized_model.to(getattr(torch, args.dtype))
    for name, module in quantized_model.named_modules():
        if isinstance(module, QuantizedWeight):
            print(f"Converting {name} for FSDP")
            assert module.codes is not None
            module.codes = module.codes.to(torch.int32)
            module.wrap_codes_for_fsdp_()
            assert module.codes is None and isinstance(module.codes_storage, IntCodes)
    quantized_model = FullyShardedDataParallel(
        quantized_model, auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, IntCodes)
    )
    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)

    print(base_model)
    print("QQQQQ" * 50)
    print(quantized_model)
    raise NotImplementedError()

