import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelMixedActivation2d(nn.Module):
    """
    Per-output-channel mixture of Identity and ReLU for 2D feature maps.

    Behavior:
      y[:, c, :, :] = ReLU(x[:, c, :, :]) if mask[c] == 1 (nonlinear)
                      x[:, c, :, :]       if mask[c] == 0 (linear)

    mask: 1D tensor of shape [C] with values {0,1}. Stored as a buffer and
          broadcast to [1, C, 1, 1] in forward.
    """
    def __init__(self, num_channels: int, nonlinear_mask: torch.Tensor | None = None):
        super().__init__()
        if nonlinear_mask is None:
            # Default: all nonlinear
            mask = torch.ones(num_channels, dtype=torch.float32)
        else:
            mask = nonlinear_mask.to(dtype=torch.float32)
            assert mask.ndim == 1 and mask.numel() == num_channels, "nonlinear_mask must be 1D [C]"
            mask = mask.clamp(0, 1)
        self.register_buffer("mask", mask, persistent=False)

    def set_mask(self, nonlinear_mask: torch.Tensor):
        assert nonlinear_mask.ndim == 1 and nonlinear_mask.numel() == self.mask.numel()
        self.mask = nonlinear_mask.to(self.mask.dtype).clamp(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        if self.mask is None:
            return F.relu(x, inplace=False)
        m = self.mask.view(1, -1, 1, 1)  # [1, C, 1, 1]
        y_relu = F.relu(x, inplace=False)
        # y = (1 - m) * x + m * relu(x)
        return (1.0 - m) * x + m * y_relu