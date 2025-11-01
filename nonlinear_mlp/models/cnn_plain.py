import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelMixedActivation2d(nn.Module):
    """
    Per-channel activation control:
      mask[c] = 1 -> ReLU on channel c
      mask[c] = 0 -> Identity on channel c
    """
    def __init__(self, num_channels: int, mask: Optional[torch.Tensor] = None):
        super().__init__()
        if mask is None:
            mask = torch.ones(num_channels, dtype=torch.float32)
        self.register_buffer("mask", mask, persistent=False)

    def set_linear_ratio(self, linear_ratio: float, pattern: str = "structured"):
        C = int(self.mask.numel())
        k = max(0, min(C, int(round(linear_ratio * C))))
        mask = torch.ones(C, dtype=torch.float32, device=self.mask.device)
        if k > 0:
            if pattern == "structured":
                mask[-k:] = 0.0  # last k channels linear (identity)
            else:
                g = torch.Generator(device=self.mask.device)
                g.manual_seed(1337)
                idx = torch.randperm(C, generator=g)[:k]
                mask[idx] = 0.0
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        relu = F.relu(x, inplace=False)
        m = self.mask.view(1, -1, 1, 1)
        return relu * m + x * (1.0 - m)


class ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class PlainCNN9(nn.Module):
    """
    9-layer plain CNN with BatchNorm and channel-wise activation control after each conv.
    - Downsample with MaxPool after conv indices 2 and 5 (after 3rd and 6th conv).
    - Global average pooling + Linear classifier.

    linear_ratio: fraction of channels that bypass ReLU (identity). The remainder use ReLU.
    """
    def __init__(self, num_classes: int = 100, linear_ratio: float = 0.0, pattern: str = "structured"):
        super().__init__()
        ch = [3, 64, 64, 64, 128, 128, 128, 256, 256, 256]  # 9 conv blocks
        self.convs = nn.ModuleList([ConvBN(ch[i], ch[i+1]) for i in range(9)])
        self.acts = nn.ModuleList([ChannelMixedActivation2d(ch[i+1]) for i in range(9)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(ch[-1], num_classes)

        self.set_linear_ratio(linear_ratio, pattern)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def set_linear_ratio(self, linear_ratio: float, pattern: str = "structured"):
        for act in self.acts:
            act.set_linear_ratio(linear_ratio, pattern)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(9):
            x = self.convs[i](x)
            x = self.acts[i](x)
            if i == 2 or i == 5:
                x = self.pool(x)
        x = x.mean(dim=[2, 3])  # global average pooling
        return self.classifier(x)
