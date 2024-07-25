import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from tqdm import tqdm
import os
import json
import shutil
from huggingface_hub import snapshot_download
import argparse


SCALES_INPUT_SUFFIX = '.quantized_weight.scales'
SCALES_OUTPUT_SUFFIX = '.scales'

CODEBOOKS_INPUT_SUFFIX = '.quantized_weight.codebooks'
CODEBOOKS_OUTPUT_SUFFIX = '.codebooks'

CODES_INPUT_SUFFIX = '.quantized_weight.codes_storage.data'
CODES_OUTPUT_SUFFIX = '.codes'


class IntCodes(nn.Module):
    """
    A storage for integer codes that makes them compatible with FullyShardedDataParallel,
    see https://github.com/pytorch/pytorch/issues/123528 for details
    """
    def __init__(self, codes: torch.tensor, storage_dtype: torch.dtype = torch.float64):
        super().__init__()
        assert torch.finfo(storage_dtype).bits % torch.iinfo(codes.dtype).bits == 0
        self.dtype, self.shape, self.numel = codes.dtype, codes.shape, codes.numel()
        size_ratio = torch.finfo(storage_dtype).bits // torch.iinfo(codes.dtype).bits
        codes = F.pad(codes.flatten().clone(), pad=[0, -codes.numel() % size_ratio])
        assert len(codes.untyped_storage()) == codes.nbytes  # no offset / stride / tail
        self.storage_dtype = storage_dtype
        self.data = nn.Parameter(
            torch.as_tensor(codes.untyped_storage(), device=codes.device, dtype=storage_dtype),
            requires_grad=False)

    def forward(self):
        assert self.data.is_contiguous() and self.data.dtype == self.storage_dtype
        byte_offset = self.data.storage_offset() * self.data.nbytes // self.data.numel()
        return torch.as_tensor(
            self.data.untyped_storage()[byte_offset: byte_offset + self.data.nbytes],
            device=self.data.device, dtype=self.dtype
        )[:self.numel].view(*self.shape)


def get_hf_aqlm_state_dict_by_best_model_state_dict(best_model_state_dict, base_model_state_dict, args):
    output = {}

    for param_name, param_value in tqdm(best_model_state_dict.items()):
        if param_name.endswith(SCALES_INPUT_SUFFIX):
            prefix = param_name[:-len(SCALES_INPUT_SUFFIX)]

            output[f'{prefix}{SCALES_OUTPUT_SUFFIX}'] = param_value.to(torch.float16)
            continue

        if param_name.endswith(CODEBOOKS_INPUT_SUFFIX):
            prefix = param_name[:-len(CODEBOOKS_INPUT_SUFFIX)]

            output[f'{prefix}{CODEBOOKS_OUTPUT_SUFFIX}'] = param_value.to(torch.float16)
            continue

        if param_name.endswith(CODES_INPUT_SUFFIX):
            prefix = param_name[:-len(CODES_INPUT_SUFFIX)]

            output_size, input_size = base_model_state_dict[f'{prefix}.weight'].shape

            codes = IntCodes(torch.empty(size=(output_size // args.out_group_size, input_size // args.in_group_size, 1),
                                         dtype=torch.int16), storage_dtype=torch.float64)
            assert codes.data.data.dtype == param_value.dtype
            assert codes.data.data.shape == param_value.shape
            codes.data[...] = param_value.contiguous()

            output[f'{prefix}{CODES_OUTPUT_SUFFIX}'] = codes()
            continue

        output[param_name] = param_value

    return output


def get_linear_weights_not_to_quantize(base_model, best_model_state_dict):
    output = []

    for module_name, module in base_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if f'{module_name}{SCALES_INPUT_SUFFIX}' in best_model_state_dict:
            continue
        for param_name, param in module.named_parameters():
            output.append(f'{module_name}.{param_name}')

    return output


def save_pt_model(args):
    with transformers.modeling_utils.no_init_weights():
        base_model_config = AutoConfig.from_pretrained(args.base_model)
        base_model = AutoModelForCausalLM.from_config(
            base_model_config,
            trust_remote_code=True,
        )
        base_model_state_dict = base_model.state_dict()

    best_model_state_dict = torch.load(args.best_model_state_dict_path)
    hf_aqlm_model_state_dict = get_hf_aqlm_state_dict_by_best_model_state_dict(
        best_model_state_dict,
        base_model_state_dict,
        args,
    )

    linear_weights_not_to_quantize = get_linear_weights_not_to_quantize(
        base_model,
        best_model_state_dict,
    )

    config = base_model_config.to_diff_dict()
    config['quantization_config'] = {
        'quant_method': 'aqlm',
        'nbits_per_codebook': args.nbits_per_codebook,
        'num_codebooks': args.num_codebooks,
        'out_group_size': args.out_group_size,
        'in_group_size': args.in_group_size,
        'linear_weights_not_to_quantize': linear_weights_not_to_quantize,
    }
    config['torch_dtype'] = 'float16'

    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)
    torch.save(hf_aqlm_model_state_dict, os.path.join(args.out_path, "pytorch_model.bin"))


def make_safetensors(args):
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.out_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    hf_model.save_pretrained(args.out_path)
    os.remove(os.path.join(args.out_path, "pytorch_model.bin"))


def populate_hf_files(args):
    base_model_path = snapshot_download(args.base_model, ignore_patterns=['*README*', '*safetensors*'])

    for filename in os.listdir(base_model_path):
        if 'README' in filename:
            continue
        if 'safetensors' in filename:
            continue
        if filename == 'config.json':
            continue

        filepath = os.path.join(base_model_path, filename)
        assert os.path.isfile(filepath)

        shutil.copyfile(filepath, os.path.join(args.out_path, filename))


def main():
    parser = argparse.ArgumentParser(description="Model configuration")

    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2-72B-Instruct',
                        help='Path to the model to base config on, as in AutoConfig.from_pretrained()')
    parser.add_argument('--best_model_state_dict_path', type=str, default='./best_model_state_dict.pt',
                        help='Path to the best model state dictionary')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Output path')
    parser.add_argument('--in_group_size', type=int, default=16,
                        help='Input group size')
    parser.add_argument('--out_group_size', type=int, default=1,
                        help='Output group size')
    parser.add_argument('--nbits_per_codebook', type=int, default=16,
                        help='Number of bits per codebook')
    parser.add_argument('--num_codebooks', type=int, default=1,
                        help='Number of codebooks')
    args = parser.parse_args()

    save_pt_model(args)
    make_safetensors(args)
    populate_hf_files(args)


if __name__ == '__main__':
    main()
