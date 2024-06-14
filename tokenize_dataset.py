"""
Fine-tune an LLM that was previously quantized with AQLM;
based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import argparse
import os
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Tuple, Optional

import transformers
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType, FullStateDictConfig, MixedPrecision
from tqdm.auto import tqdm

from src.aq import QuantizedWeight, QuantizedLinear
from src.aq_ops import IntCodes, master_rank_first, one_rank_at_a_time, is_signed
from src.datautils import group_texts, split_long_texts, get_loaders, evaluate_perplexity
from src.modelutils import get_model


def add_model_args(parser: argparse.ArgumentParser):
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


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_name",
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
        "--dataset_inner_name",
        type=str,
        default=None,
        required=True,
        help="dataset name",
    )
    parser.add_argument(
        "--dataset_n_chars",
        type=int,
        default=None,
        required=True,
        help="n chars to keep in dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        required=True,
        help="Cache dir for huggingface datasets",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="If set, re-run data preprocessing even if it is cached",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        required=True,
        help="Number of CPU workers for preprocessing; overrides num_workers",
    )
    parser.add_argument(
        "--preprocessing_chunk_length",
        type=int,
        default=None,
        required=True,
        help="Texts exceeding this length will be split approximately in the middle",
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
        "--save_dataset_and_exit",
        type=str,
        default=None,
        required=True,
        help="If not None, save tokenized dataset to this path and exit training immediately",
    )


def prepare_training_dataset(args: argparse.Namespace, tokenizer: transformers.PreTrainedTokenizer) -> datasets.Dataset:
    if os.path.exists(args.dataset_name):
        dataset = datasets.load_from_disk(args.dataset_name)
    elif args.dataset_n_chars is not None:
        dataset = datasets.load_dataset(
            args.dataset_name,
            split=args.split,
            name=args.dataset_inner_name,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            streaming=True,
        )
        def i_dataset_head(dataset, n_chars, pbar=None, text_field_name = 'text'):
            n_outputed_chars = 0
            for sample in dataset:
                assert text_field_name in sample, f'{text_field_name} not in {list(sample.keys())}'
                n_new_chars = len(sample[text_field_name])
                n_outputed_chars += n_new_chars
                if pbar is not None:
                    pbar.update(n_new_chars)
                
                yield sample
                if n_outputed_chars >= n_chars:
                    return

        with tqdm(total=args.dataset_n_chars) as pbar:
            dataset = datasets.Dataset.from_list(list(i_dataset_head(dataset, args.dataset_n_chars, pbar=pbar)))
    else:
        dataset = datasets.load_dataset(
            args.dataset_name,
            split=args.split,
            name=args.dataset_inner_name,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            streaming=False,
        )


    def is_tokenized(dataset):
        return 'input_ids' in dataset.column_names
    if is_tokenized(dataset):
        if torch.distributed.get_rank() == 0:
            print("Dataset already tokenized")
        return dataset

    assert 'text' in dataset.column_names
    text_column_name = 'text'

    if args.preprocessing_chunk_length is not None:
        dataset = dataset.map(
            lambda examples: {text_column_name: split_long_texts(
                examples[text_column_name], args.preprocessing_chunk_length)},
            batched=True,
            num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
            remove_columns=list(dataset.column_names),
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Splitting dataset over newline into chunks of ~{args.preprocessing_chunk_length} characters",
        )

    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example[text_column_name]),
        num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
        remove_columns=list(dataset.column_names),
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    lm_dataset = tokenized_dataset.map(
        partial(group_texts, block_size=args.model_seqlen, add_labels=False),
        batched=True,
        num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {args.model_seqlen}",
    )
    assert is_tokenized(lm_dataset)
    return lm_dataset


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

    base_model.config.use_cache = False
    transformer_block_types = infer_block_classes(base_model, args.block_type)
    return FullyShardedDataParallel(
        base_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, transformer_block_types),
        device_id=device
    )


def load_quantized_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    if not args.monkeypatch_old_pickle:
        quantized_model = get_model(
            args.base_model, args.quantized_model, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation
        ).to(args.master_dtype)  # master parameters
    else:
        quantized_model = _scary_load_quantized_model(args).to(args.master_dtype)

    quantized_model.config.use_cache = False
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
            if args.code_dtype is not None:
                assert module.nbits_per_codebook <= torch.iinfo(args.code_dtype).bits - is_signed(args.code_dtype)
                module.codes = nn.Parameter(module.codes.to(args.code_dtype), requires_grad=module.codes.requires_grad)
            module.wrap_codes_for_fsdp_()
            assert module.codes is None and isinstance(module.codes_storage, IntCodes)
    assert any(isinstance(module, IntCodes) for module in quantized_model.modules())

    blocks_to_wrap = (IntCodes,) + transformer_block_types
    mixed_precision = None
    if args.amp_dtype is not None:
        mixed_precision = MixedPrecision(
            param_dtype=args.amp_dtype,
            reduce_dtype=args.amp_dtype,
            _module_classes_to_ignore=(IntCodes,) + tuple(transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
        )
    return FullyShardedDataParallel(
        quantized_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, blocks_to_wrap),
        mixed_precision=mixed_precision,
        use_orig_params=True,
        device_id=device,
    )


def _scary_load_quantized_model(args: argparse.Namespace):
    """Hacky way to allow compatibility between old *pickled* layers and new transformers"""
    # because patching it for the fourth time is better than writing a proper saver once >.<
    import transformers.activations
    if not hasattr(transformers.activations, 'SiLUActivation'):
        transformers.activations.SiLUActivation = deepcopy(torch.nn.SiLU)
        transformers.activations.SiLUActivation.inplace = False
        # https://github.com/huggingface/transformers/issues/28496
    if not hasattr(transformers.models.llama.modeling_llama.LlamaAttention, 'attention_dropout'):
        transformers.models.llama.modeling_llama.LlamaAttention.attention_dropout = 0
    quantized_model = get_model(
        args.base_model, None, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation).to(args.master_dtype)
    quantized_model_src = get_model(
        args.base_model, args.quantized_model, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation
    )
    for module in quantized_model_src.modules():
        if isinstance(module, QuantizedWeight) and not hasattr(module, 'codes_storage'):
            module.codes_storage = None  # backwards compatibility with older pickled snapshots

    lut = {}
    for name, module in quantized_model_src.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, QuantizedWeight):
                lut[name + '.' + child_name] = child_module
    print(f"found {len(lut)} quantized weight matrices")
    for name, module in quantized_model.named_modules():
        for child_name, child_module in module.named_children():
            if name + '.' + child_name + '.quantized_weight' in lut:
                quantized_weight = lut.pop(name + '.' + child_name + '.quantized_weight')
                assert isinstance(child_module, nn.Linear)
                setattr(module, child_name, QuantizedLinear(quantized_weight, bias=child_module.bias))
    assert not lut, list(lut.keys())
    quantized_model.to(args.master_dtype)
    quantized_model.load_state_dict(quantized_model_src.state_dict())
    import warnings
    warnings.warn("You should be ashamed of yourself.")
    return quantized_model


def compute_loss_on_batch(
        batch: dict, base_model: nn.Module, quantized_model: nn.Module, amp_dtype: Optional[torch.dtype]
) -> torch.Tensor:
    with torch.no_grad():
        teacher_logprobs = F.log_softmax(base_model(**batch).logits, dim=-1)
    with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
        student_logprobs = F.log_softmax(quantized_model(**batch).logits, dim=-1)
        loss = F.kl_div(
            input=student_logprobs.flatten(0, -2),
            target=teacher_logprobs.flatten(0, -2),
            log_target=True,
            reduction="batchmean",
        ).mean()
    return loss


def compute_validation_perplexities(args: argparse.Namespace, model: nn.Module, eval_datasets: dict):
    rank = torch.distributed.get_rank()
    perplexities = {}
    for dataset_name, eval_dataset in eval_datasets.items():
        if rank == 0:
            print(f"Evaluating perplexity on {dataset_name} ...")
        device = next(model.parameters()).device
        amp_dtype = args.amp_dtype if args.amp_dtype is not None else (args.dtype if args.dtype != 'auto' else None)
        ppl = evaluate_perplexity(model, eval_dataset, args.model_seqlen, device=device, amp_dtype=amp_dtype)
        if rank == 0:
            print(f"{dataset_name} perplexity: {ppl:.9f}")
        perplexities[dataset_name] = ppl
    return perplexities


def _load_state(args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer):
    rank = torch.distributed.get_rank()
    if args.save is None or not os.path.exists(args.save):
        if args.save is not None and rank == 0:
            print(f"No checkpoint found at {args.save}")
    else:
        with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
            state_dict_ptr = quantized_model.state_dict()
            loaded_state_dict = torch.load(os.path.join(args.save, f'quantized_model_state_dict_rank{rank}.pt'))
            with torch.no_grad():
                for key in state_dict_ptr:
                    state_dict_ptr[key].copy_(loaded_state_dict.pop(key))
                assert len(loaded_state_dict) == 0, f"Unused keys:, {tuple(loaded_state_dict.keys())}"
            del state_dict_ptr, loaded_state_dict

        optimizer.load_state_dict(torch.load(
            os.path.join(args.save, f'optimizer_state_dict_rank{rank}.pt'),
            map_location='cpu'))
        metadata.update(torch.load(os.path.join(args.save, 'metadata.pt')))
        if args.eval_datasets is not None and metadata['early_stop_on'] not in args.eval_datasets:
            if rank == 0:
                print(f"Stopping criterion {metadata['early_stop_on']} is not in eval_datasets; resetting best loss.")
            metadata['early_stop_on'] = next(iter(args.eval_datasets))
            metadata['best_eval_perplexity'] = float('inf')
            metadata['best_step'] = 0
        if rank == 0:
            print(f"Loaded training state from {args.save}: {metadata}")


def _save_state(args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer):
    if args.save is None:
        return
    rank = torch.distributed.get_rank()
    os.makedirs(args.save, exist_ok=True)
    if rank == 0:
        print(f"Saving snapshot to {args.save}")
        torch.save(metadata, os.path.join(args.save, 'metadata.pt'))
    with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
        torch.save(quantized_model.state_dict(), os.path.join(args.save, f'quantized_model_state_dict_rank{rank}.pt'))
    torch.save(optimizer.state_dict(), os.path.join(args.save, f'optimizer_state_dict_rank{rank}.pt'))
    if args.on_save:
        exec(args.on_save)



def _save_model(args: argparse.Namespace, quantized_model: nn.Module):
    """Save consolidated model state dict"""
    os.makedirs(args.save, exist_ok=True)
    rank = torch.distributed.get_rank()
    with FullyShardedDataParallel.state_dict_type(
            quantized_model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        model_state_dict = quantized_model.state_dict()
        if rank == 0:
            torch.save(model_state_dict, os.path.join(args.save, f'best_model_state_dict.pt'))
            print(f"Saved {os.path.join(args.save, f'best_model_state_dict.pt')}")


def main():
    parser = argparse.ArgumentParser(add_help=True)
    add_model_args(parser)
    add_data_args(parser)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert tokenizer.eos_token_id is not None
    tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_training_dataset(args, tokenizer)
    dataset.save_to_disk(args.save_dataset_and_exit)


if __name__ == "__main__":
    main()
