"""Fine-tune an LLM that was previously quantized with AQLM"""
import argparse
from functools import partial
from typing import Tuple

from tqdm.auto import tqdm
import transformers
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel

from src.aq import QuantizedWeight
from src.aq_ops import IntCodes
from src.datautils import group_texts
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
        required=True,
        help="Training dataset name (from HF datasets) or path to data where to extract calibration data from",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Training dataset split name, e.g. 'train'",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir for huggingface datasets",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="If set, re-run data preprocessing even if it is cached",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of CPU workers for preprocessing and data loading",
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
        default=0.98,
        help="Adam beta2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size - how many samples are processed per optimizer step, between all GPUs in total",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size - how many samples are processed per GPU per forward pass",
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
    parser.add_argument("--save", type=str, default=None, help="Path to save training snapshot.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
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
    parser.add_argument(
        "--block_type", type=str, required=True,
        help="string name of a transformer layer to wrap, e.g. LlamaDecoderLayer"
    )
    parser.add_argument(
        "--attn_implementation", type=str, default=None,
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2"
    )


def infer_block_classes(model: nn.Module, block_type: str) -> Tuple[type, ...]:
    """find transformer block classes that should be wrapped with inner FullyShardedDataParallel (auto_wrap_policy)"""
    transformer_block_types = []
    for module in model.modules():
        if module.__class__.__name__ == block_type:
            transformer_block_types.append(type(module))
    if not transformer_block_types:
        raise ValueError(f"Could not find {block_type} among model layers")
    transformer_block_types = tuple(transformer_block_types)
    assert any(isinstance(module, transformer_block_types) for module in model.modules())
    return transformer_block_types


def load_base_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    base_model = get_model(
        args.base_model, load_quantized=None, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(dtype=args.dtype if args.dtype != 'auto' else None)
    base_model.train(False)
    for param in base_model.parameters():
        param.requires_grad = False

    transformer_block_types = infer_block_classes(base_model, args.block_type)
    return FullyShardedDataParallel(
        base_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, transformer_block_types),
        device_id=device
    )


def load_quantized_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    quantized_model = get_model(
        args.base_model, args.quantized_model, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation
    ).to(torch.float32)  # master parameters

    quantized_model.train(True)  # note: HF gradient checkpoints do not work for some models without train(True); see
    # https://github.com/huggingface/transformers/blob/2d92db8/src/transformers/models/llama/modeling_llama.py#L1006
    if args.gradient_checkpointing:
        quantized_model.gradient_checkpointing_enable()
        quantized_model.enable_input_require_grads()

    transformer_block_types = infer_block_classes(quantized_model, args.block_type)

    # convert QuantizedModel state dict to make it compatible with FSDP
    for name, module in quantized_model.named_modules():
        if isinstance(module, QuantizedWeight):
            assert module.codes is not None
            if not hasattr(module, 'codes_storage'):
                module.codes_storage = None  # backwards compatibility with older snapshots
            module.codes = nn.Parameter(module.codes.to(torch.int32), requires_grad=module.codes.requires_grad)
            module.wrap_codes_for_fsdp_()
            assert module.codes is None and isinstance(module.codes_storage, IntCodes)
    assert any(isinstance(module, IntCodes) for module in quantized_model.modules())

    blocks_to_wrap = (IntCodes,) + transformer_block_types
    return FullyShardedDataParallel(
        quantized_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, blocks_to_wrap),
        use_orig_params=True,
        device_id=device,
    )


def prepare_training_dataset(args: argparse.Namespace, tokenizer: transformers.PreTrainedTokenizer) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        streaming=False
    )
    text_column_name = 'text' if 'text' in dataset.column_names else dataset.column_names[0]
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples[text_column_name]),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=dataset.column_names,
        load_from_cache=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    lm_dataset = tokenized_dataset.map(
        partial(group_texts, block_size=args.model_seqlen),
        batched=True,
        num_proc=args.num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {args.model_seqlen}",
    )
    return lm_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    add_finetuning_args(parser)
    args = parser.parse_args()

    args.microbatch_size = args.microbatch_size or args.batch_size
    if args.dtype != 'auto':
        args.dtype = getattr(torch, args.dtype)

    assert torch.cuda.is_available() and torch.distributed.is_available()
    torch.distributed.init_process_group()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    base_model = load_base_model(args, device)
    quantized_model = load_quantized_model(args, device)

    if rank == 0:
        print(quantized_model)
        for name, param in quantized_model.named_parameters():
            print(name, param.shape, param.dtype)

    #TODO this is DEBUG AREA
    base_model.train(True)
    for param in base_model.parameters():
        if param.dtype == torch.bfloat16:
            param.requires_grad = True
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()


    input_ids = torch.arange(8 * 2048).reshape(-1, 2048).to(device) % 16_000
    for i in tqdm(range(100)):
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16):
            y = base_model(input_ids)
        y.logits.norm().backward()

