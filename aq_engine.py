from __future__ import annotations

import math
import random
import time
from argparse import Namespace
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import Gather

from src.aq import QuantizedWeight
from src.utils import ellipsis


class OutlierOptimizer:
    def __init__(
        self,
        XTX: torch.Tensor,
        sparsity: float,
        n_iters: int = 20,
        rho: float = 1.,
        is_iterative: bool = True,
    ):
        self.XTX = XTX.detach()
        self.device = self.XTX.device
        self.sparsity = sparsity
        self.n_iters = n_iters
        self.rho = rho
        self.is_iterative = is_iterative

        self.in_size, _ = self.XTX.shape
        assert self.XTX.shape == (self.in_size, self.in_size)

        self.norm = torch.diag(XTX).sqrt().clone().detach() + 1e-8
        self.XTX_norm = (
            (XTX.clone().detach() / self.norm).T / self.norm
        ).T

        self.XTX_admm = (
            self.XTX_norm +
            torch.eye(self.in_size, device=self.device) * rho
        )

        self.XTX_admm_inv = torch.inverse(self.XTX_admm)

    def get_loss(self, target_diff: torch.Tensor) -> torch.Tensor:
        return (target_diff.double() @ self.XTX.double()).flatten() @ target_diff.double().flatten() / len(target_diff)

    def wanda(self, target: torch.Tensor) -> torch.Tensor:
        target = target.detach()

        out_size, in_size = target.shape
        assert in_size == self.in_size

        XTXDiag = torch.diag(self.XTX)
        assert XTXDiag.shape == (in_size,)

        top_idxs = torch.topk(
            ((target ** 2) * XTXDiag).flatten(),
            k=int(target.numel() * self.sparsity),
            largest=False,
        ).indices

        mask = torch.ones(target.numel(), dtype=torch.bool, device=target.device)
        mask[top_idxs] = 0
        mask = mask.reshape(target.shape)

        return mask

    def _get_outliers(self, target: torch.Tensor):
        W = (target.float().detach() * self.norm).T
        XY = self.XTX_norm.matmul(W)

        U = torch.zeros_like(W)

        mask = self.wanda(target=(W + U).T / self.norm).T

        for iter_idx in range(self.n_iters):
            if self.is_iterative and iter_idx > 0:
                mask = self.wanda(target=(W + U).T / self.norm).T

            Z = (W + U) * mask
            U = U + (W - Z)
            W = self.XTX_admm_inv.matmul(XY + self.rho * (Z - U))

        Z = (W + U) * mask
        out = (Z.T / self.norm)
        out = out.reshape(target.shape).to(target.data.dtype)

        return out

    def get_outliers(self, target: torch.Tensor, old_outliers: Optional[torch.Tensor] = None) -> torch.Tensor:
        outliers = self._get_outliers(target)

        if old_outliers is None:
            return outliers

        old_loss = self.get_loss(target - old_outliers.detach())
        new_loss = self.get_loss(target - outliers.detach())

        if new_loss < old_loss:
            return outliers
        else:
            return old_outliers


class AQEngine(nn.Module):
    """A wrapper class that runs AQ training for a single linear layer. All the important math is in aq.py"""

    def __init__(self, layer: nn.Linear, accumultor_dtype: torch.dtype = torch.float64):
        super().__init__()
        self.layer = layer
        self.device = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.register_buffer(
            "XTX", torch.zeros((self.columns, self.columns), dtype=accumultor_dtype, device=self.device)
        )
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.nsamples = 0
        self.outliers_optimizer: Optional[torch.Tensor] = None

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """Accumulate a minibatch of layer inputs and update the X.T @ X (aka half hessian)"""
        assert self.XTX is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        tmp = inp.shape[0]
        inp = inp.t()

        self.XTX *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(1 / self.nsamples) * inp.to(self.XTX.dtype)
        self.XTX += inp.matmul(inp.t())

    @torch.enable_grad()
    def quantize(self, *, args: Namespace, verbose: bool = True) -> QuantizedWeight:
        """create a QuantizedLinear with specified args based on the collected hessian (XTX) data"""
        assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
        assert args.devices[0] == self.device, (args.devices[0], self.XTX.device)
        # https://github.com/pytorch/pytorch/issues/90613
        # TODO(galqiwi): fix me
        for device in args.devices:
            torch.inverse(torch.ones((1, 1), device=device))
        self.quantized_weight = QuantizedWeight(
            XTX=self.XTX.to(device=self.device, dtype=torch.float32),
            reference_weight=self.layer.weight.detach().to(device=self.device, dtype=torch.float32),
            out_group_size=args.out_group_size,
            in_group_size=args.in_group_size,
            num_codebooks=args.num_codebooks,
            nbits_per_codebook=args.nbits_per_codebook,
            codebook_value_nbits=args.codebook_value_nbits,
            codebook_value_num_groups=args.codebook_value_num_groups,
            scale_nbits=args.scale_nbits,
            max_iter=args.init_max_iter,
            max_points_per_centroid=args.init_max_points_per_centroid,
            devices=args.devices,
            verbose=True,
        )
        self.quantized_weight.outliers.requires_grad = True

        differentiable_parameters = nn.ParameterDict(
            {name: param for name, param in self.quantized_weight.named_parameters() if param.requires_grad}
        )
        opt = torch.optim.Adam(differentiable_parameters.values(), lr=args.lr, betas=(0.0, 0.95), amsgrad=True)

        replicas = None
        if len(args.devices) > 1:
            replicas = torch.nn.parallel.replicate(self, args.devices)
            replicas[0] = self

        previous_best_loss = float("inf")  # for early stopping
        for epoch in range(args.max_epochs):
            # train codebooks and scales
            for step in range(args.steps_per_epoch):
                if len(args.devices) == 1:
                    loss = self._compute_mse()
                else:
                    loss = self._compute_mse_parallel(args.devices, replicas, differentiable_parameters)

                if not torch.isfinite(loss).item():
                    raise ValueError(f"Quantization loss is {loss}")
                if step == 0 and args.relative_mse_tolerance is not None:
                    if loss.item() / previous_best_loss > (1.0 - args.relative_mse_tolerance):
                        print(f'outliers part: {(self.quantized_weight.outliers != 0).detach().float().mean().cpu().numpy()}')
                        self.quantized_weight.outliers.requires_grad = True
                        return self.quantized_weight  # early stopping; no updates after last epoch's beam search
                    previous_best_loss = min(previous_best_loss, loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and (epoch * args.steps_per_epoch + step) % args.print_frequency == 0:
                    print(f"epoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")

                assert (
                    args.steps_per_epoch % args.outliers_update_period == 0,
                    'steps_per_epoch should be divisible by outliers_update_period',
                )
                if (step + 1) % args.outliers_update_period == 0:
                    self.update_outliers(
                        args.devices,
                        replicas,
                        differentiable_parameters,
                        outliers_percentile=args.outliers_percentile,
                    )

            # search for better codes (cluster indices)
            seed = random.getrandbits(256)
            self.beam_search_update_codes_(
                args.devices,
                replicas,
                differentiable_parameters,
                seed=seed,
                beam_size=args.beam_size,
                verbose=True,
            )

        self.quantized_weight.outliers.requires_grad = True
        print(f'outliers part: {(self.quantized_weight.outliers != 0).detach().float().mean().cpu().numpy()}')
        return self.quantized_weight

    def _compute_mse(self, selection: Union[slice, ellipsis] = ...) -> torch.Tensor:
        """
        Compute the activation MSE error = ||X @ quantized_weight - X @ reference_weight||^2
        Use the square-of-difference formula to avoid materializing per-batch predictions
        :param selection:  By default, compute MSE normally. If selection is specified, this method will instead
            compute MSE over a portion of output channels that align with the selected out_groups (for parallelism)
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        """
        assert self.quantized_weight is not None, "must be called inside / after AQUtil.quantize"
        quantized_weight = self.quantized_weight(selection)

        if isinstance(selection, ellipsis):
            reference_weight = self.layer.weight.detach().to(quantized_weight.dtype)
        else:
            assert isinstance(selection, slice)
            out_channel_selection = slice(
                selection.start * self.quantized_weight.out_group_size,
                selection.stop * self.quantized_weight.out_group_size,
            )

            reference_weight = self.layer.weight.detach()[out_channel_selection].to(quantized_weight.dtype)
        delta_weight = (quantized_weight - reference_weight).to(self.XTX.dtype)
        return (delta_weight @ self.XTX).flatten() @ delta_weight.flatten() / self.quantized_weight.out_features

    def _replace_and_compute_mse(self, params_to_replace: nn.ParameterDict, selection: slice) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then compute MSE"""
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        return self._compute_mse(selection)

    def _compute_mse_parallel(
        self, devices: Sequence[torch.device], replicas: Sequence[AQEngine], parameters_to_replicate: nn.ParameterDict
    ) -> torch.Tensor:
        """Compute MSE in parallel over output channels"""
        replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
        num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
        shard_size = (num_output_groups - 1) // len(devices) + 1
        active_slices_by_replica = [
            slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
        ]
        funcs_by_replica = [replica._replace_and_compute_mse for replica in replicas]
        inputs_by_replica = [(dict(), active_slices_by_replica[0])]  # no replacements needed for 0-th replica (master)
        for i in range(1, len(devices)):
            inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
        mse_components = torch.nn.parallel.parallel_apply(funcs_by_replica, inputs_by_replica, devices=devices)
        return Gather.apply(devices[0], 0, *(mse.view(1) for mse in mse_components)).sum()

    def _replace_and_beam_search(self, params_to_replace: nn.ParameterDict, selection: slice, **kwargs) -> torch.Tensor:
        """Utility for parallelism: replace the specified parameters of self.quantized_weight, then run beam search"""
        dtype = self.quantized_weight.codebooks.dtype
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        out_channel_selection = slice(
            selection.start * self.quantized_weight.out_group_size,
            selection.stop * self.quantized_weight.out_group_size,
        )
        reference_weight = self.layer.weight.detach()[out_channel_selection].to(dtype)
        return self.quantized_weight.beam_search_update_codes_(
            self.XTX.to(dtype), reference_weight, selection=selection, **kwargs
        ).clone()

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        devices: Sequence[torch.device],
        replicas: Sequence[AQEngine],
        parameters_to_replicate: nn.ParameterDict,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Update self.quantized_weight.codes in-place via beam search"""
        if len(devices) == 1:  # single device
            assert replicas is None
            dtype = self.quantized_weight.codebooks.dtype
            self.quantized_weight.beam_search_update_codes_(
                self.XTX.to(dtype), self.layer.weight.detach().to(dtype), dim_rng=random.Random(seed), **kwargs
            )
        else:
            assert replicas[0] is self
            replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices)
            num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
            shard_size = (num_output_groups - 1) // len(devices) + 1
            active_slices_by_replica = [
                slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
            ]

            funcs_by_replica = [replica._replace_and_beam_search for replica in replicas]
            inputs_by_replica = [(dict(), active_slices_by_replica[0])]
            for i in range(1, len(devices)):
                inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
            kwargs_by_replica = [dict(kwargs, dim_rng=random.Random(seed)) for _ in range(len(devices))]
            new_code_parts_by_replica = torch.nn.parallel.parallel_apply(
                funcs_by_replica, inputs_by_replica, kwargs_by_replica, devices=devices
            )
            # gather all code parts and assign them to each replica
            for device, replica in zip(devices, replicas):
                replica.quantized_weight.codes[...] = Gather.apply(device, 0, *new_code_parts_by_replica)

    def setup_outliers_optimizer(self, outliers_percentile):
        if self.outliers_optimizer is not None:
            return
        self.outliers_optimizer = OutlierOptimizer(
            XTX=self.XTX,
            sparsity=(100. - outliers_percentile) / 100.,
        )

    def _replace_and_update_outliers(self, params_to_replace: nn.ParameterDict, selection: slice, outliers_percentile: float) -> torch.Tensor:
        dtype = self.quantized_weight.codebooks.dtype
        for param_name, param_value in params_to_replace.items():
            replace_parameter_(self.quantized_weight, param_name, param_value)
        out_channel_selection = slice(
            selection.start * self.quantized_weight.out_group_size,
            selection.stop * self.quantized_weight.out_group_size,
        )
        reference_weight = self.layer.weight.detach()[out_channel_selection].to(dtype)

        self.setup_outliers_optimizer(outliers_percentile)

        return self.quantized_weight.update_outliers(
            reference_weight=reference_weight,
            outliers_optimizer=self.outliers_optimizer,
            selection=selection,
        ).clone()


    @torch.no_grad()
    def update_outliers(
        self,
        devices: Sequence[torch.device],
        replicas: Sequence[AQEngine],
        parameters_to_replicate: nn.ParameterDict,
        outliers_percentile: float,
    ):
        begin = time.perf_counter()
        """Update self.quantized_weight.codes in-place via beam search"""
        if len(devices) == 1:  # single device
            dtype = self.quantized_weight.codebooks.dtype
            assert replicas is None

            self.setup_outliers_optimizer(outliers_percentile)

            self.quantized_weight.update_outliers(
                reference_weight=self.layer.weight.detach().to(dtype),
                outliers_optimizer=self.outliers_optimizer,
            )
            return

        assert replicas[0] is self
        replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices)
        num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
        shard_size = (num_output_groups - 1) // len(devices) + 1
        active_slices_by_replica = [
            slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
        ]

        funcs_by_replica = [replica._replace_and_update_outliers for replica in replicas]
        inputs_by_replica = [(dict(), active_slices_by_replica[0])]
        for i in range(1, len(devices)):
            inputs_by_replica.append((replicated_parameters[i], active_slices_by_replica[i]))
        kwargs_by_replica = [dict(outliers_percentile=outliers_percentile) for _ in range(len(devices))]
        new_outliers_by_replica = torch.nn.parallel.parallel_apply(
            funcs_by_replica, inputs_by_replica, kwargs_by_replica, devices=devices
        )
        # gather all code parts and assign them to each replica
        for device, replica in zip(devices, replicas):
            replica.quantized_weight.outliers[...] = Gather.apply(device, 0, *new_outliers_by_replica)
        print(time.perf_counter() - begin)


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)
