import torch
from types import SimpleNamespace

def build_param_groups(model, weight_decay: float):
    """Apply weight decay ONLY to conv/linear weights; not to BN/bias."""
    decay, no_decay = [], []
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            if m.weight is not None:
                decay.append(m.weight)
            if getattr(m, "bias", None) is not None:
                no_decay.append(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                no_decay.append(m.weight)
            if getattr(m, "bias", None) is not None:
                no_decay.append(m.bias)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def build_optimizer_and_scheduler(model, cfg):
    tr = getattr(cfg, "training", SimpleNamespace())
    model_name = getattr(cfg, "model", "")
    # Defaults: CIFAR-grade for cnn9_plain; otherwise moderate defaults
    lr = float(getattr(tr, "lr", 0.1 if model_name == "cnn9_plain" else 1e-3))
    momentum = float(getattr(tr, "momentum", 0.9))
    weight_decay = float(getattr(tr, "weight_decay", 5e-4 if model_name == "cnn9_plain" else 1e-2))
    opt_name = str(getattr(tr, "optimizer", "sgd" if model_name == "cnn9_plain" else "adamw")).lower()

    if opt_name == "sgd":
        param_groups = build_param_groups(model, weight_decay)
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=True)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(build_param_groups(model, weight_decay), lr=lr, momentum=momentum, nesterov=True)

    sch = getattr(tr, "scheduler", None)
    scheduler = None
    if isinstance(sch, (dict, SimpleNamespace)):
        name = getattr(sch, "name", sch["name"] if isinstance(sch, dict) else "step")
        if name == "step":
            milestones = getattr(sch, "milestones", sch.get("milestones", [100, 150]) if isinstance(sch, dict) else [100, 150])
            gamma = float(getattr(sch, "gamma", sch.get("gamma", 0.1) if isinstance(sch, dict) else 0.1))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)
        elif name == "cosine":
            epochs = int(getattr(tr, "epochs", 200))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler

def cross_entropy_loss(cfg):
    ls = float(getattr(getattr(cfg, "training", SimpleNamespace()), "label_smoothing", 0.0))
    return torch.nn.CrossEntropyLoss(label_smoothing=ls)