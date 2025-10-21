import torch
import torch.nn as nn
import torch.nn.functional as F
from nonlinear_mlp.layers.mixed2d import ChannelMixedActivation2d

class PlainCNN9(nn.Module):
    """
    9-layer plain CNN for CIFAR-10 with per-channel nonlinearity control.

    Convs (9 total): [3->64, 64->64, 64->64, 64->128, 128->128, 128->128, 128->256, 256->256, 256->256]
    - After each conv, apply ChannelMixedActivation2d (per-channel Identity vs ReLU).
    - MaxPool2d(2) after conv indices 2 and 5 for downsampling.
    - Global AvgPool and a Linear head to num_classes.

    Control:
    - linear_ratio ∈ [0,1] applied per conv layer: fraction of channels set to Identity (linear).
    - pattern='structured': choose the deepest channels in each layer to be linear
      (i.e., highest channel indices). You can extend with 'random' if needed.

    Stats:
    - gather_linearization_stats() to see the effective linear channel counts.
    """
    def __init__(self, num_classes=10, linear_ratio: float = 0.0, pattern: str = "structured"):
        super().__init__()
        self.num_classes = num_classes
        # Define conv channels
        channels = [64, 64, 64, 128, 128, 128, 256, 256, 256]
        in_c = 3
        self.convs = nn.ModuleList()
        for out_c in channels:
            self.convs.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True))
            in_c = out_c

        # Channel-wise nonlinear activations per conv
        self.acts = nn.ModuleList([ChannelMixedActivation2d(c) for c in channels])

        # Classifier head
        self.fc = nn.Linear(256, num_classes)

        # Initialize per-channel masks
        self.set_linear_ratio(linear_ratio, pattern)

    def _make_channel_mask(self, C: int, linear_ratio: float, pattern: str) -> torch.Tensor:
        """
        Returns a 1D tensor of length C with values {0,1} where:
          - 1 means nonlinear (ReLU)
          - 0 means linear (Identity)
        structured: set the last K channels to 0 (linear).
        """
        r = float(max(0.0, min(1.0, linear_ratio)))
        k = int(round(r * C))
        mask = torch.ones(C, dtype=torch.float32)  # default nonlinear
        if k <= 0:
            return mask
        if pattern == "structured":
            mask[-k:] = 0.0
        else:
            # fallback structured
            mask[-k:] = 0.0
        return mask

    def set_linear_ratio(self, linear_ratio: float, pattern: str = "structured"):
        """
        Apply the same per-channel linear_ratio to every conv's activation.
        """
        for i, act in enumerate(self.acts):
            C = act.mask.numel()
            act.set_mask(self._make_channel_mask(C, linear_ratio, pattern))

    @torch.no_grad()
    def gather_linearization_stats(self):
        total_channels = 0
        linear_channels = 0
        per_layer = []
        for i, act in enumerate(self.acts):
            C = act.mask.numel()
            lin = int((act.mask == 0).sum().item())
            per_layer.append({"layer": i, "channels": C, "linear_channels": lin, "linear_ratio_layer": lin / C})
            total_channels += C
            linear_channels += lin
        return {
            "total_conv_layers": len(self.acts),
            "total_channels": total_channels,
            "total_linear_channels": linear_channels,
            "linear_ratio_effective": linear_channels / total_channels if total_channels else 0.0,
            "per_layer": per_layer,
        }

    def forward(self, x):
        # Downsample after conv idx 2 and 5
        for i, (conv, act) in enumerate(zip(self.convs, self.acts)):
            x = conv(x)
            x = act(x)
            if i in (2, 5):
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
