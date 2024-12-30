from edenn import higgs_quantize_dequantize, pad_to_block, HadLinear
import torch
import math
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform


@torch.no_grad()
def quantize_linear_layer(layer: nn.Linear, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
    weight = layer.weight.float()
    # Pad to Hadamard transform size
    weight = pad_to_block(weight, [1], hadamard_groupsize)

    # Scale and Hadamard transform
    mult = weight.shape[1] // hadamard_groupsize
    weight = weight.reshape(-1, mult, hadamard_groupsize)
    scales = torch.linalg.norm(weight, axis=-1)
    weight = hadamard_transform(weight) / scales[:, :, None]

    # Pad to edenn_d and project
    weight = pad_to_block(weight, [2], edenn_d).reshape(weight.shape[0], mult, -1, edenn_d)

    for i in range(0, weight.shape[0], 64):
        weight[i:i + 64], entorpy = higgs_quantize_dequantize(weight[i:i + 64], edenn_d, edenn_n)
    weight = weight.reshape(weight.shape[0], mult, -1)

    # Cut the padded values
    weight = weight[..., :hadamard_groupsize]

    # Unscale
    weight = (weight * scales[:, :, None]).reshape(weight.shape[0], -1)

    return HadLinear(weight.half(), hadamard_groupsize), entorpy


def quantize_dequantize(weight, hadamard_groupsize: int, edenn_d: int, edenn_n: int):
    layer = nn.Linear(weight.shape[1], weight.shape[0])
    layer.weight.data = weight
    output_layer, _ = quantize_linear_layer(layer.cuda(), hadamard_groupsize, edenn_d, edenn_n)
    return output_layer(torch.eye(weight.shape[1], device='cuda').half()).T.detach().contiguous().clone().to(
        weight.device)


import sys
import time
from typing import Optional, Dict, Any

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import sklearn.linear_model

import transformers
from tqdm.auto import tqdm, trange
import main as calibration_utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class args:  # wannabe argparse namespace
    model_name = "unsloth/Llama-3.2-3B"
    torch_dtype = "auto"
    compute_dtype = None  # used for computing activations
    dataset = "pajama"
    total_nsamples = 16 # todo: use larger datasets for actual training
    model_seqlen = 8192
    seed = 42
    offload_activations = True  # if True, store hidden states in RAM
    devices = [device]
    wandb = False  # <-- to be implemented


tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model_name, torch_dtype=args.torch_dtype, low_cpu_mem_usage=True, use_cache=False
)
layers = calibration_utils.get_layers(model)

train_data = calibration_utils.get_loaders(
    args.dataset,
    nsamples=args.total_nsamples,
    seed=args.seed,
    model_path=args.model_name,
    seqlen=args.model_seqlen,
)


def update_outs_inplace_(args, layer, inps, outs, **kwargs):
    """
    Forward a single layer on a collection of :inps: to update :outs: in-place
    see docs at https://github.com/Vahe1994/AQLM/blob/main/main.py#L530
    """
    assert len(inps) == len(outs) == len(args.devices)
    if len(args.devices) == 1:
        return calibration_utils.update_outs(layer, inps[0], outs[0], **kwargs)
    else:
        return calibration_utils.update_outs_parallel(args.devices, layer, inps, outs, **kwargs)

class OutputCatcher(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.outputs = []
    def forward(self, inp, **kwargs):
        output = self.inner(inp)
        self.outputs.append(copy.deepcopy(output))
        return output


def get_predictor(old_values, new_values):
    X = torch.concatenate(old_values, dim=0).detach().clone().cpu().float()
    y = torch.concatenate(new_values, dim=0).detach().clone().cpu().float()

    N, dim = X.shape

    assert X.shape == y.shape == (N, dim)
    X_train = X[N // 2:]
    y_train = y[N // 2:]
    X_val = X[:N // 2]
    y_val = y[:N // 2]

    model = sklearn.linear_model.LinearRegression().fit(X_train.numpy(), y_train.numpy())
    predictor = nn.Linear(dim, dim)
    predictor.weight.data = torch.tensor(model.coef_)
    predictor.bias.data = torch.tensor(model.intercept_)

    mse_train = ((predictor(X_train) - y_train).norm() / y_train.norm()).item() ** 2
    mse_val = ((predictor(X_val) - y_val).norm() / y_val.norm()).item() ** 2

    return predictor, mse_train, mse_val


def get_dequant_values(old_values, values, pred):
    old_values_tensor = torch.concatenate([elem[None, :, :] for elem in old_values]).float().cpu()
    values_tensor = torch.concatenate([elem[None, :, :] for elem in values]).float().cpu()
    pred = pred.float().cpu()
    values_pred_tensor = pred(old_values_tensor)
    values_delta_tensor = values_tensor - values_pred_tensor

    values_delta_dequant_tensor = quantize_dequantize(
        values_delta_tensor.reshape(-1, values_delta_tensor.shape[-1]),
        1024,
        6,
        4096,
    ).reshape(values_delta_tensor.shape)

    values_dequantized_tensor = values_delta_dequant_tensor + values_pred_tensor

    return list(values_dequantized_tensor)
