import torch
import torch.nn as nn
from typing import List, Optional, Dict

from nonlinear_mlp.layers.mixed import MixedActivationLayer
from nonlinear_mlp.layers.gated import GatedActivationLayer
from nonlinear_mlp.layers.nl_dropout_layer import NonlinearityDropoutLayer

class MLP(nn.Module):
    """
    Flexible MLP supporting:
      - Fixed mixed activations (Approach 1)
      - Gated layers (Approach 2)
      - Layer-wise ratio schedule (Approach 5)
      - Deterministic per-neuron zeroing ('nl_dropout')
        -> exactly N% of neurons per hidden layer are set to 0; remaining use ReLU
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
        fixed_cfg = fixed_cfg or {}
        gating_cfg = gating_cfg or {}
        layerwise_cfg = layerwise_cfg or {}

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            in_f, out_f = dims[i], dims[i+1]

            if approach in ["fixed", "layerwise", "mixed"]:
                if layerwise_cfg and layerwise_cfg.get("enabled") and layerwise_cfg.get("schedule"):
                    # schedule may be dict keyed by layer index, or list
                    sched = layerwise_cfg["schedule"]
                    if isinstance(sched, dict):
                        ratio = sched.get(i, fixed_cfg.get("linear_ratio", 0.5))
                    else:
                        ratio = sched[i] if i < len(sched) else fixed_cfg.get("linear_ratio", 0.5)
                else:
                    per_layer = fixed_cfg.get("per_layer", [])
                    ratio = per_layer[i] if (isinstance(per_layer, list) and i < len(per_layer)) else fixed_cfg.get("linear_ratio", 0.5)
                layer = MixedActivationLayer(
                    in_f,
                    out_f,
                    linear_ratio=float(ratio),
                    pattern=fixed_cfg.get("pattern", "structured")
                )

            elif approach == "gating":
                layer = GatedActivationLayer(
                    in_f,
                    out_f,
                    init_alpha=float(gating_cfg.get("init_alpha", 0.75)),
                    temperature=float(gating_cfg.get("temperature", 1.0)),
                    clamp=bool(gating_cfg.get("clamp", True))
                )

            elif approach == "nl_dropout":
                # Deterministic zeroing: zero_ratio per layer; remaining neurons use ReLU
                if layerwise_cfg and layerwise_cfg.get("enabled") and layerwise_cfg.get("schedule"):
                    sched = layerwise_cfg["schedule"]
                    if isinstance(sched, dict):
                        zero_ratio = sched.get(i, fixed_cfg.get("linear_ratio", 0.5))
                    else:
                        zero_ratio = sched[i] if i < len(sched) else fixed_cfg.get("linear_ratio", 0.5)
                else:
                    per_layer = fixed_cfg.get("per_layer", [])
                    zero_ratio = per_layer[i] if (isinstance(per_layer, list) and i < len(per_layer)) else fixed_cfg.get("linear_ratio", 0.5)

                layer = NonlinearityDropoutLayer(
                    in_features=in_f,
                    out_features=out_f,
                    zero_ratio=float(zero_ratio),
                    pattern=fixed_cfg.get("pattern", "structured"),
                    layer_index=i,
                    base_seed=int(fixed_cfg.get("base_seed", 1337)),
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
