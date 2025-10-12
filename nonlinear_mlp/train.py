import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from nonlinear_mlp.utils.metrics import accuracy
import time
import os
import json
from typing import Dict, Any
from torch import amp  # modern AMP API

def compute_regularization(model, cfg, device):
    # Always return a 0-dim tensor on the correct device
    reg_loss = torch.zeros((), device=device)
    if cfg.approach == "gating":
        for layer in model.feature_layers:
            a = layer.alpha()
            if cfg.gating.entropy_reg > 0:
                entropy = - (a * torch.log(a + 1e-8) + (1 - a) * torch.log(1 - a + 1e-8))
                reg_loss = reg_loss + cfg.gating.entropy_reg * entropy.mean()
            if cfg.gating.l1_reg > 0:
                reg_loss = reg_loss + cfg.gating.l1_reg * a.abs().mean()
        if cfg.gating.sparsity_target is not None:
            all_a = torch.cat([l.alpha() for l in model.feature_layers])
            sparsity = (all_a < 0.5).float().mean()
            diff = sparsity - cfg.gating.sparsity_target
            reg_loss = reg_loss + cfg.gating.sparsity_loss_weight * diff.abs()
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
    log_interval: int = 100
):
    model.train()
    running = {"loss": 0.0, "acc": 0.0, "n": 0}
    start = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if x.ndim == 4 and model.__class__.__name__ == "MLP":
            x = x.view(x.size(0), -1)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type="cuda", enabled=cfg.training.amp and device.startswith("cuda")):
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

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"Loss: {running['loss']/running['n']:.4f} "
                f"Acc: {running['acc']/running['n']:.2f} "
                f"Reg: {reg_loss.item():.4f}"
            )

    duration = time.time() - start
    return {
        "train_loss": running["loss"] / running["n"],
        "train_acc": running["acc"] / running["n"],
        "train_time_s": duration,
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
    
def save_checkpoint(model, optimizer, cfg, epoch, metrics, path):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg.to_json(),
        "epoch": epoch,
        "metrics": metrics
    }
    torch.save(ckpt, os.path.join(path, f"checkpoint_{epoch}.pt"))

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
