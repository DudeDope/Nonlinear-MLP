import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedActivationLayer(nn.Module):
    """
    Approach 2: Learned soft gating
    output = alpha * ReLU(Wx + b) + (1-alpha) * (Wx + b)
    alpha is per-neuron in [0,1].
    Regularization handled externally (entropy, L1, sparsity target).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_alpha: float = 0.75,
        temperature: float = 1.0,
        clamp: bool = True,
        device=None, dtype=None
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, device=device, dtype=dtype)
        # parameterizing alpha via logits for stable optimization
        init_alpha = torch.full((out_features,), float(init_alpha))
        init_alpha = torch.clamp(init_alpha, 1e-4, 1 - 1e-4)
        init_logit = torch.log(init_alpha) - torch.log(1 - init_alpha)
        self.alpha_logit = nn.Parameter(init_logit)
        self.temperature = temperature
        self.clamp = clamp

    def alpha(self):
        a = torch.sigmoid(self.alpha_logit / self.temperature)
        if self.clamp:
            return torch.clamp(a, 0.0, 1.0)
        return a

    def forward(self, x):
        z = self.linear(x)
        a = self.alpha()
        relu_part = F.relu(z)
        return a * relu_part + (1 - a) * z

    def harden(self, threshold=0.1):
        with torch.no_grad():
            a = self.alpha()
            mask = (a >= threshold).float()
            # rewrite forward path: convert to mixture of (ReLU or identity)
            # Instead we freeze alpha_logit to extremes
            new_alpha = mask
            eps = 1e-4
            new_alpha = torch.clamp(new_alpha, eps, 1 - eps)
            new_logit = torch.log(new_alpha) - torch.log(1 - new_alpha)
            self.alpha_logit.copy_(new_logit)

    def stats(self):
        with torch.no_grad():
            a = self.alpha()
            return {
                "alpha_mean": float(a.mean()),
                "alpha_median": float(a.median()),
                "alpha_min": float(a.min()),
                "alpha_max": float(a.max()),
                "alpha_lt_0.1": float((a < 0.1).float().mean()),
                "alpha_lt_0.25": float((a < 0.25).float().mean()),
                "alpha_gt_0.9": float((a > 0.9).float().mean()),
            }