import torch
from collections import defaultdict

class ActivationTracker:
    """
    Tracks pre-activation distributions to evaluate:
      - fraction positive
      - fraction negative
      - nonlinear contribution score: E[ |min(0,z)| / (|z| + eps) ]
    """
    def __init__(self, eps=1e-6, max_batches=50, device="cpu"):
        self.eps = eps
        self.max_batches = max_batches
        self.layer_stats = defaultdict(lambda: {
            "positive": 0,
            "negative": 0,
            "total": 0,
            "nonlinear_score_sum": 0.0
        })
        self._handles = []
        self._batch_count = 0

    def _hook_factory(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                z = out.detach()
            elif isinstance(out, (tuple, list)):
                z = out[0].detach()
            else:
                return
            pos = (z > 0).sum().item()
            neg = (z < 0).sum().item()
            total = z.numel()
            nonlinear_contrib = (z.clamp(max=0).abs() / (z.abs() + self.eps)).mean().item()
            st = self.layer_stats[name]
            st["positive"] += pos
            st["negative"] += neg
            st["total"] += total
            st["nonlinear_score_sum"] += nonlinear_contrib
        return hook

    def register(self, model):
        for i, m in enumerate(model.modules()):
            if isinstance(m, torch.nn.Linear):
                self._handles.append(m.register_forward_hook(self._hook_factory(f"linear_{i}")))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def run(self, model, dataloader, device):
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                if x.ndim == 4:  # images
                    pass
                model(x)
                self._batch_count += 1
                if self._batch_count >= self.max_batches:
                    break
        out = {}
        for k, v in self.layer_stats.items():
            if v["total"] == 0:
                continue
            out[k] = {
                "positive_frac": v["positive"] / v["total"],
                "negative_frac": v["negative"] / v["total"],
                "nonlinear_score": v["nonlinear_score_sum"] / self._batch_count
            }
        return out