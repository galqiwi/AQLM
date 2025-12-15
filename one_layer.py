import os
os.environ["OMP_NUM_THREADS"] = "4"

import sys

import time
import random
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aq import QuantizedWeight

torch.set_num_threads(4)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_loading_dir = '/DIR/layer10.self_attn.q_proj.input_activation.pt'
num_codebooks = 2
nbits_per_codebook = 8
out_group_size = 1
in_group_size = 8
batch_size = 16384
beam_size = 1
beam_search_epochs = 100
sparsity_regularizer = 0
print_frequency = 10
scale_nbits = 0    # 0 means no scales, 16 means no compression;
codebook_values_nbits = 16  # less than 16 means we quantize codebooks as well
init_max_iter = 100


reference_weight = torch.randn((2048, 1024)).float()

X = torch.randn((4096, 1024))
XTX = (X.T @ X + torch.eye(1024, 1024).to(device)).float()

quantized_weight = QuantizedWeight.from_unquantized(
    reference_weight=reference_weight, num_codebooks=num_codebooks,
    nbits_per_codebook=nbits_per_codebook, scale_nbits=scale_nbits,
    out_group_size=out_group_size, in_group_size=in_group_size,
    verbose=True, max_iter=init_max_iter,   # faster init, not tested
)

opt = torch.optim.Adam(quantized_weight.parameters(), lr=1e-4, betas=(0.0, 0.95), amsgrad=True)

for epoch in range(1000):
    start = time.perf_counter()
    delta_weight = (quantized_weight() - reference_weight).double()
    loss = (delta_weight @ XTX.double()).flatten() @ delta_weight.flatten() / len(delta_weight)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % print_frequency == 0:
        print(f"loss={loss.item():.10f}\t",
              f"time_on_epoch {epoch} = {time.perf_counter() - start}")
    if (epoch + 1) % beam_search_epochs == 0:
        quantized_weight.beam_search_update_codes_(
            XTX=XTX, reference_weight=reference_weight, beam_size=beam_size, sparsity_regularizer=sparsity_regularizer,
            dim_rng=random.Random(), verbose=True)
