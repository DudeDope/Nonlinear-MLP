import torch
import torch.nn as nn
import torch.nn.functional as F

class NonlinearityDropout(nn.Module):
    """
    Deterministic per-neuron zeroing of nonlinearity for MLP hidden layers.

    Behavior (deterministic, not random):
      - Exactly zero_ratio fraction of output neurons are 'dropped' (forced to 0)
      - The remaining neurons use ReLU

    Let keep_mask be a 1D vector of shape [D] with values in {0,1}:
      y[:, j] = keep_mask[j] * ReLU(x[:, j])          # dropped neurons output 0 always

    Notes:
      - This is not standard dropout: there is no randomness per batch, and no scaling.
      - The mask is fixed at initialization time according to the chosen pattern:
          * 'structured' -> last K channels are dropped (indices D-K ... D-1)
          * 'random'     -> deterministically choose K indices via a fixed seed per layer
      - layer_index and base_seed are used only for deterministic 'random' selection.
    """
    def __init__(
        self,
        out_features: int,
        zero_ratio: float = 0.5,
        pattern: str = "structured",
        layer_index: int = 0,
        base_seed: int = 1337,
    ):
        super().__init__()
        self.out_features = int(out_features)
        self.zero_ratio = float(max(0.0, min(1.0, zero_ratio)))
        self.pattern = str(pattern)
        self.layer_index = int(layer_index)
        self.base_seed = int(base_seed)

        keep = self._build_keep_mask(self.out_features, self.zero_ratio, self.pattern, self.layer_index, self.base_seed)
        # 1 means keep (nonlinear ReLU); 0 means drop (output forced to 0)
        self.register_buffer("keep_mask", keep, persistent=False)

    def _build_keep_mask(self, D: int, zero_ratio: float, pattern: str, layer_index: int, base_seed: int) -> torch.Tensor:
        k = int(round(zero_ratio * D))  # number to drop (set to zero)
        keep = torch.ones(D, dtype=torch.float32)
        if k <= 0:
            return keep
        if k >= D:
            return torch.zeros(D, dtype=torch.float32)

        if pattern == "structured":
            # Drop last K neurons deterministically
            keep[-k:] = 0.0
        elif pattern == "random":
            # Deterministic 'random' selection using fixed seed per layer
            g = torch.Generator()
            g.manual_seed(base_seed + 1000 * layer_index + 17)
            idx = torch.randperm(D, generator=g)[:k]
            keep[idx] = 0.0
        else:
            # Fallback to structured
            keep[-k:] = 0.0
        return keep

    def set_zero_ratio(self, zero_ratio: float, pattern: str | None = None):
        """Allow changing the mask deterministically at runtime (optional)."""
        zr = float(max(0.0, min(1.0, zero_ratio)))
        pat = self.pattern if pattern is None else pattern
        self.keep_mask = self._build_keep_mask(self.out_features, zr, pat, self.layer_index, self.base_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, D]; fallback to last-dim broadcasting if needed
        y_relu = F.relu(x, inplace=False)
        # Broadcast keep mask over batch and any leading dims
        view = [1] * (y_relu.ndim - 1) + [y_relu.shape[-1]]
        m = self.keep_mask.view(*view)
        return m * y_relu