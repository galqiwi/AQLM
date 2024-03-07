import torch


def admm_reference(target, XTX, sparsity, percdamp=.1, iterative_prune=15, iters=20, per_out=False):
    # TODO(galqiwi): refactor me and put somewhere
    XTX = XTX.clone().detach()
    target = target.clone().detach()

    assert 0. <= sparsity <= 1.
    assert iterative_prune < iters, 'galqiwi didn\'t test admm_prune when iterative_prune >= iters'

    norm = torch.diag(XTX).sqrt() + 1e-8
    XTX = XTX / norm
    XTX = (XTX.T / norm).T
    W = (target.float().detach() * norm).T

    rho0 = percdamp * torch.diag(XTX).mean()
    diag = torch.arange(XTX.shape[0], device=XTX.device)
    XTX[diag, diag] += rho0

    mask = None

    if iterative_prune == 0:
        if per_out:
            thres = (W).abs().sort(dim=0)[0][int(W.shape[0] * sparsity)]
            mask = ((W).abs() >= thres.unsqueeze(0))
            del thres
        else:
            topk = torch.topk(W.abs().flatten(), k=int(W.numel() * sparsity), largest=False)
            # topk will have .indices and .values
            mask = torch.ones(W.numel(), dtype=torch.bool, device=W.device)
            mask[topk.indices] = 0
            mask = mask.reshape(W.shape)
            del topk

    assert iters > 0

    rho = 1

    XY = XTX.matmul(W)
    XTX[diag, diag] += rho
    torch.cuda.empty_cache()

    XTXinv = torch.inverse(XTX)

    U = torch.zeros_like(W)
    old_mask = None

    for itt in range(iters):
        if iterative_prune > 0 and itt < iterative_prune:
            cur_sparsity = sparsity - sparsity * (1 - (itt + 1) / iterative_prune) ** 3
            if per_out:
                thres = (W + U).abs().sort(dim=0)[0][int(W.shape[0] * cur_sparsity)]
                mask = ((W + U).abs() >= thres.unsqueeze(0))
                del thres
            else:
                topk = torch.topk((W + U).abs().flatten(), k=int(W.numel() * sparsity), largest=False)
                # topk will have .indices and .values
                if mask is not None:
                    old_mask = mask.clone().detach()
                mask = torch.ones(W.numel(), dtype=torch.bool, device=W.device)
                mask[topk.indices] = 0
                mask = mask.reshape(W.shape)
                # if old_mask is not None:
                #     print(((old_mask == mask) & old_mask).sum() / mask.sum())
                del topk

        Z = (W + U) * mask

        U = U + (W - Z)

        W = XTXinv.matmul(XY + rho * (Z - U))

    Z = (W + U) * mask
    out = (Z.T / norm)

    target.data = out.reshape(target.shape).to(target.data.dtype)

    assert abs((target == 0).float().mean() - sparsity) / (1 - sparsity) < 0.1

    return target