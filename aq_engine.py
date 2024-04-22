from __future__ import annotations

import math
import random
from argparse import Namespace
from typing import Optional, Sequence, Union

import torch.cuda.comm
import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import Gather

from src.aq import QuantizedWeight
from src.utils import ellipsis


def tensor_to_str(x: torch.Tensor):
    def get_non_finite_features(x: torch.Tensor):
        features = []
        n_nans = torch.isnan(x).sum().item()
        if n_nans > 0:
            features.append(f'n_nans={n_nans}')
        n_infs = torch.isinf(x).sum().item()
        if n_infs > 0:
            features.append(f'n_infs={n_infs}')
        return features

    def get_finite_features(x: torch.Tensor):
        features = []
        if not x.dtype == torch.bool:
            features.append(f'min={x.min().item()}')
            features.append(f'max={x.max().item()}')

        if not torch.is_floating_point(x):
            features.append(f'sum={x.sum().item()}')
            return features

        features.append(f'mean={x.mean().item()}')
        if x.numel() > 1:
            features.append(f'std={x.std().item()}')

        n_zeroes = (x == 0.0).sum().item()
        if n_zeroes > 0:
            features.append(f'n_zeroes={n_zeroes}')

        return features

    def get_features(x: torch.Tensor):
        features = []
        features.append(f'shape={tuple(x.shape)}')

        if (~torch.isfinite(x)).sum().item() > 0:
            features.extend(get_non_finite_features(x))
        else:
            features.extend(get_finite_features(x))

        features.append(f'dtype={x.dtype}')
        features.append(f'device={x.device}')
        return features

    return 'Tensor(' + ', '.join(get_features(x)) + ')'


class AQEngine(nn.Module):
    """A wrapper class that runs AQ training for a single linear layer. All the important math is in aq.py"""

    def __init__(self, layer: nn.Linear, accumulator_dtype: torch.dtype = torch.float64):
        super().__init__()
        self.layer = layer
        self.device = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.register_buffer(
            "XTX", torch.zeros((self.columns, self.columns), dtype=accumulator_dtype, device=self.device)
        )
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.nsamples = 0

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

    @staticmethod
    @torch.no_grad()
    def _optimize_scales(quantized_weight, reference_weight, XTX):
        out_size, in_size = reference_weight.shape
        assert quantized_weight.scales.shape == (out_size, 1, 1, 1)

        old_scales = quantized_weight.scales.data[:, 0, 0, 0]
        assert old_scales.shape == (out_size,)

        quantized_weight.scales.data = torch.ones_like(quantized_weight.scales.data)

        XTX = XTX.double()
        quantized_weight_value = quantized_weight().double()
        assert quantized_weight_value.shape == (out_size, in_size)

        quantized_weight_XTX = quantized_weight_value @ XTX
        assert quantized_weight_XTX.shape == (out_size, in_size)

        optimal_scales_num = (quantized_weight_XTX * reference_weight).sum(dim=1)
        optimal_scales_denum = (quantized_weight_XTX * quantized_weight_value).sum(dim=1)

        optimal_scales = optimal_scales_num / optimal_scales_denum
        assert optimal_scales.shape == (out_size,)

        # optimal_scales[0] = float('nan')
        # optimal_scales[1] = float('+inf')
        # optimal_scales[2] = float('-inf')

        print(f'optimizer optimal_scales_num={tensor_to_str(optimal_scales_num)}')
        print(f'optimizer optimal_scales_denum={tensor_to_str(optimal_scales_denum)}')
        print(f'optimizer optimal_scales={tensor_to_str(optimal_scales)}')

        nan_mask = ~torch.isfinite(optimal_scales)
        if nan_mask.sum().item() != 0:
            optimal_scales[nan_mask] = old_scales[nan_mask].to(optimal_scales.dtype)
            print(f'optimizer new_optimal_scales={tensor_to_str(optimal_scales)}')

        optimal_scales = optimal_scales.reshape(out_size, 1, 1, 1)
        optimal_scales = optimal_scales.to(quantized_weight.scales.data.dtype)

        quantized_weight.scales.data = optimal_scales.to(old_scales.dtype)

    def optimize_scales(self, devices, replicas, reference_weight):
        self._optimize_scales(self.quantized_weight, reference_weight, self.XTX)
        if len(devices) == 1:
            return
        for replica, scales in zip(
            replicas[1:],
            torch.cuda.comm.broadcast(
                self.quantized_weight.scales,
                devices=devices[1:]
            )
        ):
            replica.quantized_weight.scales.data = scales

    @torch.enable_grad()
    def quantize(self, *, args: Namespace, verbose: bool = True) -> QuantizedWeight:
        """create a QuantizedLinear with specified args based on the collected hessian (XTX) data"""
        assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
        assert args.devices[0] == self.device, (args.devices[0], self.XTX.device)
        reference_weight = self.layer.weight.detach().to(device=self.device, dtype=torch.float32)
        self.quantized_weight = QuantizedWeight(
            XTX=self.XTX.to(device=self.device, dtype=torch.float32),
            reference_weight=reference_weight,
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
        self.quantized_weight.scales.requires_grad = False
        self._optimize_scales(
            quantized_weight=self.quantized_weight,
            reference_weight=reference_weight,
            XTX=self.XTX,
        )

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
                if (step + 1) % 10 == 0:
                    self.optimize_scales(
                        devices=args.devices,
                        replicas=replicas,
                        reference_weight=reference_weight,
                    )

                if len(args.devices) == 1:
                    loss = self._compute_mse()
                else:
                    loss = self._compute_mse_parallel(args.devices, replicas, differentiable_parameters)

                if not torch.isfinite(loss).item():
                    raise ValueError(f"Quantization loss is {loss}")
                if step == 0 and args.relative_mse_tolerance is not None:
                    if loss.item() / previous_best_loss > (1.0 - args.relative_mse_tolerance):
                        self.quantized_weight.scales.requires_grad = True
                        return self.quantized_weight  # early stopping; no updates after last epoch's beam search
                    previous_best_loss = min(previous_best_loss, loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and (epoch * args.steps_per_epoch + step) % args.print_frequency == 0:
                    print(f"epoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")

            # search for better codes (cluster indices)
            seed = random.getrandbits(256)
            self.beam_search_update_codes_(
                args.devices,
                replicas,
                differentiable_parameters,
                seed=seed,
                beam_size=args.beam_size,
                verbose=False,
            )
        self.quantized_weight.scales.requires_grad = True
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


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)
