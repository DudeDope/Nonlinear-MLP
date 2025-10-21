import torch
import torch.nn as nn
import torch.nn.functional as F

class NonlinearityDropoutLayer(nn.Module):
    """
    Deterministic per-neuron zeroing for an MLP hidden block:
      y = ReLU(Wx + b) with exactly zero_ratio fraction of output neurons forced to 0.
    No randomness at train or eval. No scaling (this is not standard dropout).

    Args:
      in_features:  int
      out_features: int
      zero_ratio:   float in [0,1], fraction of output neurons to zero permanently
      pattern:      'structured' (last K indices) or 'random' (deterministic, seeded)
      layer_index:  int, for deterministic 'random' selection
      base_seed:    int, base seed for 'random' selection
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        zero_ratio: float = 0.5,
        pattern: str = "structured",
        layer_index: int = 0,
        base_seed: int = 1337
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.out_features = int(out_features)
        self.zero_ratio = float(max(0.0, min(1.0, zero_ratio)))
        self.pattern = str(pattern)
        self.layer_index = int(layer_index)
        self.base_seed = int(base_seed)

        keep = self._build_keep_mask(self.out_features, self.zero_ratio, self.pattern, self.layer_index, self.base_seed)
        # 1 = keep ReLU output, 0 = force to 0
        self.register_buffer("keep_mask", keep, persistent=False)

    def _build_keep_mask(self, D: int, zero_ratio: float, pattern: str, layer_index: int, base_seed: int) -> torch.Tensor:
        k = int(round(zero_ratio * D))  # number of neurons to zero
        keep = torch.ones(D, dtype=torch.float32)
        if k <= 0:
            return keep
        if k >= D:
            return torch.zeros(D, dtype=torch.float32)

        if pattern == "structured":
            keep[-k:] = 0.0  # drop last K channels deterministically
        elif pattern == "random":
            g = torch.Generator()
            g.manual_seed(base_seed + 1000 * layer_index + 17)
            idx = torch.randperm(D, generator=g)[:k]
            keep[idx] = 0.0
        else:
            keep[-k:] = 0.0
        return keep

    def set_zero_ratio(self, zero_ratio: float, pattern: str | None = None):
        zr = float(max(0.0, min(1.0, zero_ratio)))
        pat = self.pattern if pattern is None else pattern
        self.keep_mask = self._build_keep_mask(self.out_features, zr, pat, self.layer_index, self.base_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        a = F.relu(z, inplace=False)
        m = self.keep_mask.view(1, -1)
        return a * m

    @torch.no_grad()
    def stats(self):
        dropped = int((self.keep_mask == 0).sum().item())
        return {
            "out_features": self.out_features,
            "zero_ratio": self.zero_ratio,
            "dropped_neurons": dropped,
            "kept_neurons": self.out_features - dropped,
        }
