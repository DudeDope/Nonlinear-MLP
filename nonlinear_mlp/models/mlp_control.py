import math
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Union


def _compute_keep_ratios(
    num_layers: int,
    fixed_cfg: Dict[str, Union[float, List[float]]]
) -> List[float]:
    """
    Derive per-layer keep ratios from 'fixed' config:
      - If fixed.per_layer is provided: keep_i = 1 - per_layer[i]  (per_layer[i] is linear_ratio_i)
      - Else: keep_i = 1 - fixed.linear_ratio (uniform)
    """
    per_layer = fixed_cfg.get("per_layer", None)
    if isinstance(per_layer, (list, tuple)) and len(per_layer) >= num_layers:
        return [max(0.0, min(1.0, 1.0 - float(per_layer[i]))) for i in range(num_layers)]
    linear_ratio = float(fixed_cfg.get("linear_ratio", 0.0))
    keep = max(0.0, min(1.0, 1.0 - linear_ratio))
    return [keep] * num_layers


def _shrink_dims(base_hidden_dims: List[int], keep_ratios: List[float], min_width: int = 1) -> List[int]:
    """
    Reduce each layer width: out_i' = max(min_width, round(out_i * keep_i)).
    """
    assert len(base_hidden_dims) == len(keep_ratios), "hidden dims and keep ratios length mismatch"
    reduced = []
    for out_w, keep in zip(base_hidden_dims, keep_ratios):
        target = int(round(out_w * keep))
        reduced.append(max(min_width, target))
    return reduced


class MLPControl(nn.Module):
    """
    Control MLP that structurally reduces hidden widths to match the desired nonlinearity fraction.

    Example:
      - If your fixed run used linear_ratio=0.75 (=> nonlinearity=25%), this control model will
        keep 25% of neurons in each hidden layer structurally (reduce widths).

    Args:
      input_dim: int, required for standalone MLPs (e.g., MNIST)
      base_hidden_dims: list[int], the original hidden sizes (before shrinking)
      num_classes: int
      fixed_cfg: dict-like, uses 'linear_ratio' or 'per_layer' to derive keep ratios
      layerwise_cfg: ignored here (kept for signature consistency)
      min_width: int, lower bound for any hidden layer to avoid zero-width layers
    """
    def __init__(
        self,
        input_dim: int,
        base_hidden_dims: List[int],
        num_classes: int,
        fixed_cfg: Optional[Dict] = None,
        layerwise_cfg: Optional[Dict] = None,
        min_width: int = 1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.base_hidden_dims = list(base_hidden_dims)
        self.num_classes = int(num_classes)
        fixed_cfg = fixed_cfg or {}

        # Derive per-layer keep ratios and shrink widths
        keep_ratios = _compute_keep_ratios(len(self.base_hidden_dims), fixed_cfg)
        self.reduced_hidden_dims = _shrink_dims(self.base_hidden_dims, keep_ratios, min_width=min_width)

        # Build a simple ReLU MLP with reduced widths
        dims = [self.input_dim] + self.reduced_hidden_dims + [self.num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten images for MLPs (matches training/eval loops expectations)
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        return self.net(x)

    @torch.no_grad()
    def control_stats(self) -> Dict[str, Union[int, float, List[int]]]:
        base_total = sum(self.base_hidden_dims)
        reduced_total = sum(self.reduced_hidden_dims)
        return {
            "base_hidden_dims": self.base_hidden_dims,
            "reduced_hidden_dims": self.reduced_hidden_dims,
            "total_neurons_base": base_total,
            "total_neurons_reduced": reduced_total,
            "overall_keep_ratio": (reduced_total / base_total) if base_total > 0 else 0.0,
        }
