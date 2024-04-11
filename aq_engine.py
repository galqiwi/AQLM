from __future__ import annotations

import math
import random
import time
from argparse import Namespace
from typing import Optional, Sequence, Union

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import Gather

from src.aq import QuantizedWeight
from src.info import _calculate_code_entropy, _get_entropy_penalties_upper_bound
from src.utils import ellipsis


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

    def quantize(self, *, args: Namespace, verbose: bool = True) -> QuantizedWeight:
        info_regularizers = []
        entropy_values = []
        quantized_weights = []

        def get_entropy(info_regularizer):
            args.info_regularizer = info_regularizer
            quantized_weight = self._quantize(args=args, verbose=verbose)
            quantized_weights.append(quantized_weight)
            info_regularizers.append(info_regularizer)

            entropy = _calculate_code_entropy(
                quantized_weight.codes, codebook_size=2 ** args.nbits_per_codebook).mean().item()
            entropy_values.append(entropy)

            return entropy

        entropy_target = args.target_group_entropy
        entropy_target_err = 0.1
        regularizer_start = args.info_regularizer

        regularizer_l = None
        entropy_l = None
        regularizer_r = None
        entropy_r = None
        # entropy(regularizer_r) <= entropy_target
        # entropy_target < entropy(regularizer_l)

        entropy_start = get_entropy(regularizer_start)
        if entropy_start <= entropy_target:
            regularizer_r = regularizer_start
            entropy_r = entropy_start
        else:
            regularizer_l = regularizer_start
            entropy_l = entropy_start

        regularizer = regularizer_start

        while regularizer_r is None:
            regularizer *= 10
            entropy = get_entropy(regularizer)
            print('BINSEARCH', regularizer, entropy)
            if entropy <= entropy_target:
                regularizer_r = regularizer
                entropy_r = entropy
            else:
                regularizer_l = regularizer
                entropy_l = entropy

        regularizer = regularizer_start

        while regularizer_l is None:
            regularizer /= 10
            entropy = get_entropy(regularizer)
            print('BINSEARCH', regularizer, entropy)
            if entropy_target < entropy:
                regularizer_l = regularizer
                entropy_l = entropy
            else:
                regularizer_r = regularizer
                entropy_r = entropy

        print(f'BINSEARCH ')
        print(f'BINSEARCH finished calibrating')
        print(f'BINSEARCH {regularizer_l=}')
        print(f'BINSEARCH {entropy_l=}')
        print(f'BINSEARCH {regularizer_r=}')
        print(f'BINSEARCH {entropy_r=}')
        print(f'BINSEARCH ')

        for step in range(10):
            if abs(entropy_l - entropy_target) < entropy_target_err:
                break
            if abs(entropy_r - entropy_target) < entropy_target_err:
                break
            t = (entropy_target - entropy_l) / (entropy_r - entropy_l)

            t = max(t, 0.01)
            t = min(t, 0.99)

            regularizer_m = np.exp(
                np.log(regularizer_l) + (np.log(regularizer_r) - np.log(regularizer_l)) * t
            )

            print(f'BINSEARCH ')
            print(f'BINSEARCH {regularizer_l=}')
            print(f'BINSEARCH {entropy_l=}')
            print(f'BINSEARCH {regularizer_r=}')
            print(f'BINSEARCH {entropy_r=}')
            print(f'BINSEARCH {t=}')
            print(f'BINSEARCH ')
            print(f'BINSEARCH {regularizer_m=}')

            start = time.perf_counter()
            entropy_m = get_entropy(regularizer_m)
            stop = time.perf_counter()

            print(f'BINSEARCH {entropy_m=}')
            print(f'BINSEARCH took {stop - start}s')
            print(f'BINSEARCH ')

            if abs(entropy_m - entropy_target) < entropy_target_err:
                break

            if entropy_m <= entropy_target:
                regularizer_r = regularizer_m
                entropy_r = entropy_m
            else:
                regularizer_l = regularizer_m
                entropy_l = entropy_m

        final_info_regularizer = None
        final_entropy_value = None
        final_quantized_weight = None

        for (info_regularizer, entropy_value, quantized_weight) in zip(
            info_regularizers,
            entropy_values,
            quantized_weights,
        ):
            if final_entropy_value is None or (
                abs(final_entropy_value - entropy_target) > abs(entropy_value - entropy_target)
            ):
                final_info_regularizer = info_regularizer
                final_entropy_value = entropy_value
                final_quantized_weight = quantized_weight

        print(f'BINSEARCH {final_info_regularizer=}')
        print(f'BINSEARCH {final_entropy_value=}')

        return final_quantized_weight

    @torch.enable_grad()
    def _quantize(self, *, args: Namespace, verbose: bool = True) -> QuantizedWeight:
        """create a QuantizedLinear with specified args based on the collected hessian (XTX) data"""
        assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
        assert args.devices[0] == self.device, (args.devices[0], self.XTX.device)
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

        differentiable_parameters = nn.ParameterDict(
            {name: param for name, param in self.quantized_weight.named_parameters() if param.requires_grad}
        )
        opt = torch.optim.Adam(differentiable_parameters.values(), lr=args.lr, betas=(0.0, 0.95), amsgrad=True)

        replicas = None
        if len(args.devices) > 1:
            replicas = torch.nn.parallel.replicate(self, args.devices)
            replicas[0] = self

        dynamic_regularizer_coefficient = None
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
                if dynamic_regularizer_coefficient is None:
                    dynamic_regularizer_coefficient = args.info_regularizer * loss.item()  # scale with initial mse

                if step == 0 and args.relative_mse_tolerance is not None:
                    penalty = self._compute_regularizer_penalty(dynamic_regularizer_coefficient, args)
                    full_loss = loss.item() + penalty
                    print(f"Full loss: {full_loss:.5f} = {loss.item():.9f} (mse) + {penalty:.9f} (reg)")
                    if full_loss / previous_best_loss > (1.0 - args.relative_mse_tolerance):
                        return self.quantized_weight  # early stopping; no updates after last epoch's beam search
                    previous_best_loss = min(previous_best_loss, full_loss)

                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and (epoch * args.steps_per_epoch + step) % args.print_frequency == 0:
                    print(f"epoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")

            # search for better codes (cluster indices)
            seed = random.getrandbits(256)
            print("Entropy before beam search:", _calculate_code_entropy(
                self.quantized_weight.codes, codebook_size=2 ** args.nbits_per_codebook).mean().item(), flush=True)
            code_penalties = _get_entropy_penalties_upper_bound(
                self.quantized_weight.codes, codebook_size=2 ** args.nbits_per_codebook,
                regularizer=dynamic_regularizer_coefficient)

            begin = time.perf_counter()
            self.beam_search_update_codes_(
                args.devices,
                replicas,
                differentiable_parameters,
                seed=seed,
                beam_size=args.beam_size,
                code_penalties=code_penalties,
                verbose=True,
            )
            print(f'beam search took {time.perf_counter()-begin}s')
            print("Entropy after beam search:", _calculate_code_entropy(
                self.quantized_weight.codes, codebook_size=2 ** args.nbits_per_codebook).mean().item(), flush=True)

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

    def _replace_and_beam_search(
        self, params_to_replace: nn.ParameterDict, selection: slice,
        code_penalties: Optional[torch.Tensor], **kwargs,
    ) -> torch.Tensor:
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
            self.XTX.to(dtype), reference_weight, selection=selection, code_penalties=code_penalties, **kwargs
        ).clone()

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        devices: Sequence[torch.device],
        replicas: Sequence[AQEngine],
        parameters_to_replicate: nn.ParameterDict,
        seed: Optional[int] = None,
        code_penalties: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Update self.quantized_weight.codes in-place via beam search"""
        if len(devices) == 1:  # single device
            assert replicas is None
            dtype = self.quantized_weight.codebooks.dtype
            self.quantized_weight.beam_search_update_codes_(
                self.XTX.to(dtype), self.layer.weight.detach().to(dtype), dim_rng=random.Random(seed),
                code_penalties=code_penalties, **kwargs
            )
        else:
            assert replicas[0] is self
            replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices)

            replicated_code_penalties = [
                None if code_penalties is None else code_penalties.to(device)
                for device in devices
            ]

            num_output_groups = self.quantized_weight.out_features // self.quantized_weight.out_group_size
            shard_size = (num_output_groups - 1) // len(devices) + 1
            active_slices_by_replica = [
                slice(i * shard_size, min((i + 1) * shard_size, num_output_groups)) for i in range(len(devices))
            ]

            funcs_by_replica = [replica._replace_and_beam_search for replica in replicas]
            inputs_by_replica = [(dict(), active_slices_by_replica[0], replicated_code_penalties[0])]
            for i in range(1, len(devices)):
                inputs_by_replica.append((
                    replicated_parameters[i],
                    active_slices_by_replica[i],
                    replicated_code_penalties[i],
                ))
            kwargs_by_replica = [dict(kwargs, dim_rng=random.Random(seed)) for _ in range(len(devices))]
            new_code_parts_by_replica = torch.nn.parallel.parallel_apply(
                funcs_by_replica, inputs_by_replica, kwargs_by_replica, devices=devices
            )
            # gather all code parts and assign them to each replica
            for device, replica in zip(devices, replicas):
                replica.quantized_weight.codes[...] = Gather.apply(device, 0, *new_code_parts_by_replica)

    @torch.no_grad()
    def _compute_regularizer_penalty(self, regularizer_coefficient: float, args: Namespace) -> float:
        if regularizer_coefficient == 0:
            return 0.
        # Compute counts for each code in each codebook, initialize regularizer
        codebook_ids = torch.arange(args.num_codebooks, device=args.devices[0]).view(1, 1, 1, -1)
        code_penalties = _get_entropy_penalties_upper_bound(
            self.quantized_weight.codes, codebook_size=2 ** args.nbits_per_codebook,
            regularizer=regularizer_coefficient)
        per_channel_regularizers = code_penalties[codebook_ids, self.quantized_weight.codes].sum(
            dim=(-2, -1))  # [num_out_groups]
        return per_channel_regularizers.mean().item()


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)
