import torch
import torch.nn as nn
from typing import List, Union, Optional, Dict

from nonlinear_mlp.layers.mixed import MixedActivationLayer
from nonlinear_mlp.layers.gated import GatedActivationLayer

class MLP(nn.Module):
    """
    Flexible MLP supporting:
      - Fixed mixed activations (Approach 1)
      - Gated layers (Approach 2)
      - Layer-wise ratio schedule (Approach 5)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        approach: str = "fixed",
        fixed_cfg: Optional[Dict] = None,
        gating_cfg: Optional[Dict] = None,
        layerwise_cfg: Optional[Dict] = None
    ):
        super().__init__()
        self.approach = approach
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            in_f, out_f = dims[i], dims[i+1]
            if approach in ["fixed", "layerwise", "mixed"]:
                if layerwise_cfg and layerwise_cfg.get("enabled") and layerwise_cfg.get("schedule"):
                    ratio = layerwise_cfg["schedule"].get(i, fixed_cfg.get("linear_ratio", 0.5))
                else:
                    ratio = fixed_cfg.get("per_layer", [])[i] if fixed_cfg.get("per_layer") else fixed_cfg.get("linear_ratio", 0.5)
                layer = MixedActivationLayer(
                    in_f,
                    out_f,
                    linear_ratio=ratio,
                    pattern=fixed_cfg.get("pattern", "structured")
                )
            elif approach == "gating":
                layer = GatedActivationLayer(
                    in_f,
                    out_f,
                    init_alpha=gating_cfg.get("init_alpha", 0.75),
                    temperature=gating_cfg.get("temperature", 1.0),
                    clamp=gating_cfg.get("clamp", True)
                )
            else:
                raise ValueError(f"Unknown approach {approach}")
            layers.append(layer)
        self.feature_layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for layer in self.feature_layers:
            x = layer(x)
        return self.classifier(x)

    def gather_gating_stats(self):
        stats = []
        if self.approach == "gating":
            for i, layer in enumerate(self.feature_layers):
                stats.append({"layer": i, **layer.stats()})
        return stats

    def harden_gates(self, threshold=0.1):
        if self.approach == "gating":
            for layer in self.feature_layers:
                layer.harden(threshold)