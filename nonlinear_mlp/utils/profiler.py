import torch
from torch import nn

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}

def approximate_flops_linear(in_features, out_features):
    # MACs ~ in*out; FLOPs ~ 2*MACs
    return 2 * in_features * out_features

def model_linear_flops(model):
    # naive: sum over Linear layers only
    flops = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            in_f = m.in_features
            out_f = m.out_features
            flops += approximate_flops_linear(in_f, out_f)
    return flops