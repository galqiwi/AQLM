import numpy as np
import torch
import functools

from pathlib import Path

import os

import matplotlib.pyplot as plt


from typing import List

from admm_impl import (
    admm_reference,
)


@functools.cache
def get_tensor(path: str) -> torch.Tensor:
    return torch.load(path, map_location='cuda').detach()


def get_XTX() -> torch.Tensor:
    return get_tensor('LLAMA2_XTX.pt')


def get_loss(delta_weight: torch.Tensor, XTX: torch.Tensor) -> torch.Tensor:
    delta_weight = delta_weight.double()
    return (delta_weight @ XTX.double()).flatten() @ delta_weight.flatten() / len(delta_weight)


def wanda(target: torch.Tensor, XTX: torch.Tensor, sparsity: float) -> torch.Tensor:
    out_size, in_size = target.shape
    assert XTX.shape == (in_size, in_size)

    norm = torch.diag(XTX.detach()).sqrt() + 1e-8
    assert norm.shape == (in_size,)

    W = (target.float().detach() * norm).T

    topk = torch.topk(W.abs().flatten(), k=int(W.numel() * sparsity), largest=False)
    mask = torch.ones(W.numel(), dtype=torch.bool, device=W.device)
    mask[topk.indices] = 0
    mask = mask.reshape(W.shape)

    return mask.T


def custom_admm(
    target: torch.Tensor,
    XTX: torch.Tensor,
    sparsity: float,
    percdamp: float = 1e-3,
    n_iters: int = 20,
    rho: float = 1,
    mask_rule: any = None,
) -> torch.Tensor:
    if mask_rule is None:
        mask_rule = wanda

    out_size, in_size = target.shape
    assert XTX.shape == (in_size, in_size)

    XTX_orig = XTX.clone().detach()

    XTX = XTX.clone().detach()
    target = target.clone().detach()

    norm = torch.diag(XTX).sqrt() + 1e-8
    XTX = XTX / norm
    XTX = (XTX.T / norm).T
    W = (target.float().detach() * norm).T

    rho0 = percdamp * torch.diag(XTX).mean()
    diag = torch.arange(XTX.shape[0], device=XTX.device)
    XTX[diag, diag] += rho0

    XY = XTX.matmul(W)
    XTX[diag, diag] += rho
    torch.cuda.empty_cache()

    XTXinv = torch.inverse(XTX)

    U = torch.zeros_like(W)

    assert n_iters > 0
    for itt in range(n_iters):
        mask = mask_rule(target=(W + U).T / norm, XTX=XTX_orig, sparsity=sparsity).T
        Z = (W + U) * mask
        U = U + (W - Z)
        W = XTXinv.matmul(XY + rho * (Z - U))

    Z = (W + U) * mask
    out = (Z.T / norm)
    out = out.reshape(target.shape).to(target.data.dtype)

    return out


def get_log_sensitivity(delta_weight: torch.Tensor, XTX: torch.Tensor):
    delta_weight = delta_weight.clone().detach()
    delta_weight.requires_grad = True

    loss = get_loss(delta_weight, XTX)
    loss.backward()
    return delta_weight.grad.abs().log10()


def gradient_mask(target: torch.Tensor, XTX: torch.Tensor, sparsity: float) -> torch.Tensor:
    out_size, in_size = target.shape
    assert XTX.shape == (in_size, in_size)

    log_sensitivity = get_log_sensitivity(target, XTX)

    top_elems = torch.topk(
        log_sensitivity.flatten(),
        k=int(log_sensitivity.numel() * sparsity),
        largest=False,
    ).indices
    mask = torch.ones(target.numel(), dtype=torch.bool, device=target.device)
    mask[top_elems] = 0
    mask = mask.reshape(target.shape)

    return mask


def test(target_name: str = 'LLAMA2_TARGET_BASE.pt') -> List[any]:
    results = []
    target_base = get_tensor(target_name)

    exp_name = f'{"admm_reference":<17} | {target_name:<25} |'

    outliers = admm_reference(target=target_base, XTX=get_XTX(), sparsity=0.99)
    loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
    results.append({
        'name': f'{exp_name} iterative_prune',
        'loss': loss,
    })

    exp_name = f'{"admm_custom":<17} | {target_name:<25} |'

    outliers = custom_admm(target=target_base, XTX=get_XTX(), sparsity=0.99)
    loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
    results.append({
        'name': f'{exp_name}',
        'loss': loss,
    })

    exp_name = f'{"admm_custom_grad":<17} | {target_name:<25} |'

    outliers = custom_admm(target=target_base, XTX=get_XTX(), sparsity=0.99, mask_rule=gradient_mask)
    loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
    results.append({
        'name': f'{exp_name}',
        'loss': loss,
    })

    return results


def plot_rho(target_name: str):
    target_base = get_tensor(target_name)

    percdamps = np.logspace(-2, 0, 21)
    losses = []
    for percdamp in percdamps:
        print(percdamp)
        outliers = custom_admm(target=target_base, XTX=get_XTX(), sparsity=0.99, percdamp=percdamp)
        loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
        losses.append(loss)

    plt.plot(percdamps, losses)
    Path("./output").mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join('./output', f'{target_name}.png'))
    plt.clf()


def main():
    targets = [
        'LLAMA2_TARGET_BASE.pt',
        'LLAMA2_DIFF_BASE.pt',
        'LLAMA2_DIFF_TUNED.pt',
    ]

    # for target in targets:
    #     print(target)
    #     plot_rho(target)

    results = []
    for target in targets:
        results.extend(test(target))

    for result in results:
        print(f'{result["name"]:<65} | {result["loss"]}')


if __name__ == '__main__':
    main()
