import torch
import torch.nn.functional as F


def get_c(codes, n_codes):
    return F.one_hot(codes, num_classes=n_codes).to(torch.float32)
