import numpy as np
import torch
import functools
import tqdm

import itertools

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
    is_iterative: bool = False,
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
    mask = mask_rule(target=(W + U).T / norm, XTX=XTX_orig, sparsity=sparsity).T
    for itt in range(n_iters):
        if is_iterative:
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


def sparsegpt_mask(target: torch.Tensor, XTX: torch.Tensor, sparsity: float) -> torch.Tensor:
    out_size, in_size = target.shape
    assert XTX.shape == (in_size, in_size)

    XTXInvDiag = torch.diag(torch.inverse(XTX))
    assert XTXInvDiag.shape == (in_size,)

    XTXDiag = torch.diag(XTX)
    assert XTXDiag.shape == (in_size,)

    errors = (target ** 2) / XTXInvDiag

    top_elems = torch.topk(
        errors.flatten(),
        k=int(errors.numel() * sparsity),
        largest=False,
    ).indices
    mask = torch.ones(target.numel(), dtype=torch.bool, device=target.device)
    mask[top_elems] = 0
    mask = mask.reshape(target.shape)

    return mask


def custom_galqiwi_mask(target: torch.Tensor, XTX: torch.Tensor, sparsity: float) -> torch.Tensor:
    out_size, in_size = target.shape
    assert XTX.shape == (in_size, in_size)

    B = target @ XTX
    assert B.shape == (out_size, in_size)

    A = torch.diag(XTX)
    assert A.shape == (in_size,)

    errors = (B ** 2) / A

    top_elems = torch.topk(
        errors.flatten(),
        k=int(errors.numel() * sparsity),
        largest=False,
    ).indices
    mask = torch.ones(target.numel(), dtype=torch.bool, device=target.device)
    mask[top_elems] = 0
    mask = mask.reshape(target.shape)

    return mask


def custom_galqiwi_mask_2(target: torch.Tensor, XTX: torch.Tensor, sparsity: float, alpha: float = 0) -> torch.Tensor:
    out_size, in_size = target.shape
    assert XTX.shape == (in_size, in_size)

    XTXInvDiag = torch.diag(torch.inverse(XTX))
    assert XTXInvDiag.shape == (in_size,)

    XTXDiag = torch.diag(XTX)
    assert XTXDiag.shape == (in_size,)

    errors = (target ** 2) * torch.exp(torch.log(XTXDiag) * (1 - alpha) - torch.log(XTXInvDiag) * alpha)

    top_elems = torch.topk(
        errors.flatten(),
        k=int(errors.numel() * sparsity),
        largest=False,
    ).indices
    mask = torch.ones(target.numel(), dtype=torch.bool, device=target.device)
    mask[top_elems] = 0
    mask = mask.reshape(target.shape)

    return mask


def test(target_name: str = 'LLAMA2_TARGET_BASE.pt', is_iterative: bool = False) -> List[any]:
    results = []
    target_base = get_tensor(target_name)

    # target_base = target_base[:1024]

    exp_name = f'{"admm_reference":<17} | {target_name:<25} | ' + ('iter' if is_iterative else '')

    outliers = admm_reference(target=target_base, XTX=get_XTX(), sparsity=0.99, iterative_prune=15 if is_iterative else 0)
    loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
    results.append({
        'name': f'{exp_name}',
        'loss': loss,
    })

    exp_name = f'{"admm_custom":<17} | {target_name:<25} | ' + ('iter' if is_iterative else '')

    outliers = custom_admm(target=target_base, XTX=get_XTX(), sparsity=0.99, is_iterative=is_iterative)
    loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
    results.append({
        'name': f'{exp_name}',
        'loss': loss,
    })

    # exp_name = f'{"admm_custom_grad":<17} | {target_name:<25} | ' + ('iter' if is_iterative else '')
    #
    # outliers = custom_admm(target=target_base, XTX=get_XTX(), sparsity=0.99, mask_rule=gradient_mask, is_iterative=is_iterative)
    # loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
    # results.append({
    #     'name': f'{exp_name}',
    #     'loss': loss,
    # })

    exp_name = f'{"admm_custom_sgpt":<17} | {target_name:<25} | ' + ('iter' if is_iterative else '')

    outliers = custom_admm(target=target_base, XTX=get_XTX(), sparsity=0.99, mask_rule=sparsegpt_mask, is_iterative=is_iterative)
    loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
    results.append({
        'name': f'{exp_name}',
        'loss': loss,
    })

    # exp_name = f'{"admm_custom_galq":<17} | {target_name:<25} | ' + ('iter' if is_iterative else '')
    #
    # outliers = custom_admm(target=target_base, XTX=get_XTX(), sparsity=0.99, mask_rule=custom_galqiwi_mask_2, is_iterative=is_iterative)
    # loss = get_loss(delta_weight=target_base - outliers, XTX=get_XTX()).item()
    # results.append({
    #     'name': f'{exp_name}',
    #     'loss': loss,
    # })

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
    for is_iterative, target in tqdm.tqdm(list(itertools.product((False, True), targets))):
        results.extend(test(target_name=target, is_iterative=is_iterative))

    for result in results:
        print(f'{result["name"]:<65} | {result["loss"]}')


"""
admm_reference    | LLAMA2_TARGET_BASE.pt     |                   | 0.04655745510796658
admm_custom       | LLAMA2_TARGET_BASE.pt     |                   | 0.04610034123178573
admm_custom_sgpt  | LLAMA2_TARGET_BASE.pt     |                   | 0.049215821234794094
admm_reference    | LLAMA2_DIFF_BASE.pt       |                   | 0.0007004170255205616
admm_custom       | LLAMA2_DIFF_BASE.pt       |                   | 0.0006986690087244959
admm_custom_sgpt  | LLAMA2_DIFF_BASE.pt       |                   | 0.0007511789466704419
admm_reference    | LLAMA2_DIFF_TUNED.pt      |                   | 0.0009394660339419621
admm_custom       | LLAMA2_DIFF_TUNED.pt      |                   | 0.0009252707204601611
admm_custom_sgpt  | LLAMA2_DIFF_TUNED.pt      |                   | 0.0009146751103682076
admm_reference    | LLAMA2_TARGET_BASE.pt     | iter              | 0.04027909150722395
admm_custom       | LLAMA2_TARGET_BASE.pt     | iter              | 0.03978602941067679
admm_custom_sgpt  | LLAMA2_TARGET_BASE.pt     | iter              | 0.04508117530962251
admm_reference    | LLAMA2_DIFF_BASE.pt       | iter              | 0.0006523642361295563
admm_custom       | LLAMA2_DIFF_BASE.pt       | iter              | 0.0006505168824564085
admm_custom_sgpt  | LLAMA2_DIFF_BASE.pt       | iter              | 0.0007074907882695715
admm_reference    | LLAMA2_DIFF_TUNED.pt      | iter              | 0.0009036392631954713
admm_custom       | LLAMA2_DIFF_TUNED.pt      | iter              | 0.0008862151520391128
admm_custom_sgpt  | LLAMA2_DIFF_TUNED.pt      | iter              | 0.0008948043425832325
"""


if __name__ == '__main__':
    main()
