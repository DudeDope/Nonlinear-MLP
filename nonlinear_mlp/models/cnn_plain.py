import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainCNN9(nn.Module):
    """
    9-layer plain CNN for CIFAR-10:
      - Convs: [3->64, 64->64, 64->64, 64->128, 128->128, 128->128, 128->256, 256->256, 256->256]
      - ReLU after each conv by default
      - MaxPool2d(2) after convs 3 and 6 (i.e., indices 2 and 5)
      - Global AvgPool and FC head to num_classes
    Nonlinearity control:
      - linear_ratio in [0,1]: fraction of conv ReLUs to remove (replace with Identity)
      - pattern='structured': remove from deeper layers first (i.e., last convs)
    """
    def __init__(self, num_classes=10, linear_ratio: float = 0.0, pattern: str = "structured"):
        super().__init__()
        self.num_classes = num_classes
        # Channels per conv layer (9 convs)
        channels = [64, 64, 64, 128, 128, 128, 256, 256, 256]
        in_c = 3
        self.convs = nn.ModuleList()
        for out_c in channels:
            self.convs.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True))
            in_c = out_c

        # One activation module per conv; default ReLU
        self.activations = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(len(self.convs))])

        # FC head after GAP; last conv output channels = 256
        self.fc = nn.Linear(256, num_classes)

        # Apply nonlinearity scheme
        self.set_linear_ratio(linear_ratio, pattern)

    def _compute_relu_mask(self, linear_ratio: float, pattern: str):
        """
        Returns a boolean list of length 9; True means 'linear' (Identity), False means ReLU kept.
        structured: remove from deeper layers first (end of the stack backward).
        """
        n = len(self.convs)
        k = int(round(max(0.0, min(1.0, float(linear_ratio))) * n))
        mask = [False] * n  # default keep ReLU
        if k <= 0:
            return mask
        if pattern == "structured":
            # Deep-first removal: last k relus become Identity
            for i in range(n - k, n):
                mask[i] = True
        else:
            # Fallback to structured if unknown
            for i in range(n - k, n):
                mask[i] = True
        return mask

    def set_linear_ratio(self, linear_ratio: float, pattern: str = "structured"):
        mask = self._compute_relu_mask(linear_ratio, pattern)
        # Replace selected ReLU with Identity
        for i, linearize in enumerate(mask):
            self.activations[i] = nn.Identity() if linearize else nn.ReLU(inplace=True)

    @torch.no_grad()
    def gather_linearization_stats(self):
        # Returns how many activations are Identity
        total = len(self.activations)
        linearized = sum(1 for a in self.activations if isinstance(a, nn.Identity))
        return {
            "total_conv_layers": total,
            "linearized_relus": linearized,
            "linear_ratio_effective": linearized / total if total else 0.0
        }

    def forward(self, x):
        # Two downsampling points after conv idx 2 and 5
        for i, (conv, act) in enumerate(zip(self.convs, self.activations)):
            x = act(conv(x))
            if i in (2, 5):
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Global average pool to 1x1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x