import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from nonlinear_mlp.utils.metrics import accuracy
import time
import os
import json
from typing import Dict, Any, Callable, Optional
from torch import amp  # modern AMP API

def compute_regularization(model, cfg, device):
    # Always return a 0-dim tensor on the correct device
    reg_loss = torch.zeros((), device=device)
    if getattr(cfg, "approach", None) == "gating":
        for layer in getattr(model, "feature_layers", []):
            if not hasattr(layer, "alpha"):
                continue
            a = layer.alpha()
            if getattr(cfg.gating, "entropy_reg", 0.0) > 0:
                entropy = - (a * torch.log(a + 1e-8) + (1 - a) * torch.log(1 - a + 1e-8))
                reg_loss = reg_loss + cfg.gating.entropy_reg * entropy.mean()
            if getattr(cfg.gating, "l1_reg", 0.0) > 0:
                reg_loss = reg_loss + cfg.gating.l1_reg * a.abs().mean()
        if getattr(cfg.gating, "sparsity_target", None) is not None:
            all_a = torch.cat([l.alpha() for l in model.feature_layers if hasattr(l, "alpha")])
            sparsity = (all_a < 0.5).float().mean()
            diff = sparsity - cfg.gating.sparsity_target
            reg_loss = reg_loss + getattr(cfg.gating, "sparsity_loss_weight", 0.0) * diff.abs()
    return reg_loss

def train_one_epoch(
    model,
    loader,
    optimizer,
    device: str,
    scaler,
    epoch: int,
    cfg,
    criterion=nn.CrossEntropyLoss(),
    log_interval: int = 100,
    on_step: Optional[Callable[[Dict[str, Any], int], None]] = None,  # NEW: per-step callback(record, global_step)
    start_step: int = 0,  # NEW: global step offset from caller
):
    """
    Trains for a single epoch.
    - If on_step is provided, it is invoked after each batch with (record, global_step).
    - Returns usual epoch stats plus 'end_step' so caller can maintain a global step counter.
    """
    model.train()
    running = {"loss": 0.0, "acc": 0.0, "n": 0}
    start = time.time()
    gstep = int(start_step)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if x.ndim == 4 and model.__class__.__name__ == "MLP":
            x = x.view(x.size(0), -1)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type="cuda", enabled=getattr(cfg.training, "amp", False) and device.startswith("cuda")):
            logits = model(x)
            cls_loss = criterion(logits, y)
            reg_loss = compute_regularization(model, cfg, device)
            total_loss = cls_loss + reg_loss

        if scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            acc1 = accuracy(logits, y, topk=(1,))[0]

        running["loss"] += cls_loss.item() * x.size(0)
        running["acc"] += acc1 * x.size(0)
        running["n"] += x.size(0)

        # Optional console log
        if (batch_idx + 1) % log_interval == 0:
            print(
                f"Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"Loss: {running['loss']/running['n']:.4f} "
                f"Acc: {running['acc']/running['n']:.2f} "
                f"Reg: {float(reg_loss.item()):.4f}"
            )

        # NEW: per-step callback to e.g., W&B
        if on_step is not None:
            try:
                on_step(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "batch_size": int(x.size(0)),
                        "loss": float(cls_loss.item()),
                        "acc": float(acc1),  # already in %
                        "reg": float(reg_loss.item()),
                        "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
                    },
                    gstep,
                )
            except Exception as _:
                # Don't break training if logging fails
                pass

        gstep += 1

    duration = time.time() - start
    return {
        "train_loss": running["loss"] / running["n"],
        "train_acc": running["acc"] / running["n"],
        "train_time_s": duration,
        "end_step": gstep,  # NEW: hand back the next global step
    }

@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if x.ndim == 4 and model.__class__.__name__ == "MLP":
            x = x.view(x.size(0), -1)
        logits = model(x)
        loss = criterion(logits, y)
        acc1 = accuracy(logits, y, topk=(1,))[0]
        total_loss += loss.item() * x.size(0)
        total_acc += acc1 * x.size(0)
        n += x.size(0)
    return {"val_loss": total_loss / n, "val_acc": total_acc / n}

def save_checkpoint(model, optimizer, cfg, epoch, record, out_dir):
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "record": record,
        "config": json.loads(cfg.to_json()) if hasattr(cfg, "to_json") else {},
    }
    torch.save(obj, os.path.join(out_dir, f"checkpoint_{epoch}.pt"))

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
