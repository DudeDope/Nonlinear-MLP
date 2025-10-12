import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import math
import random

Pattern = Literal["structured", "random", "alternating"]

class MixedActivationLayer(nn.Module):
    """
    Approach 1: Fixed percentage linear vs ReLU neurons.
    Supports patterns:
      structured: first N% linear, rest ReLU
      random: randomly assigned (fixed mask)
      alternating: linear, relu, linear, relu...
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        linear_ratio: float = 0.5,
        pattern: Pattern = "structured",
        device=None,
        dtype=None,
        seed: Optional[int] = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, **factory_kwargs)
        self.out_features = out_features
        self.linear_ratio = linear_ratio
        self.pattern = pattern
        self.seed = seed
        n_linear = int(round(out_features * linear_ratio))
        if seed is not None:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        if pattern == "structured":
            mask = torch.zeros(out_features, dtype=torch.bool)
            mask[:n_linear] = True
        elif pattern == "random":
            perm = torch.randperm(out_features)
            mask = torch.zeros(out_features, dtype=torch.bool)
            mask[perm[:n_linear]] = True
        elif pattern == "alternating":
            mask = torch.zeros(out_features, dtype=torch.bool)
            mask[::2] = True  # half linear approximate
            # adjust to desired ratio:
            current = mask.sum().item()
            diff = n_linear - current
            if diff > 0:
                # turn on more
                off_idx = (~mask).nonzero(as_tuple=True)[0]
                mask[off_idx[:diff]] = True
            elif diff < 0:
                on_idx = mask.nonzero(as_tuple=True)[0]
                mask[on_idx[: -diff]] = False
        else:
            raise ValueError(f"Unknown pattern {pattern}")
        if seed is not None:
            torch.random.set_rng_state(rng_state)
        self.register_buffer("linear_mask", mask)  # True -> linear neuron

    def forward(self, x):
        out = self.linear(x)
        # Apply ReLU only where mask is False (nonlinear neurons)
        if self.linear_mask is None:
            return F.relu(out)
        # Clone for in-place safety if needed
        relu_part = F.relu(out)
        out = torch.where(self.linear_mask.unsqueeze(0), out, relu_part)
        return out