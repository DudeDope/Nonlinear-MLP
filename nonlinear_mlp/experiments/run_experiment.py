import argparse
import torch
from nonlinear_mlp.config import ExperimentConfig
from nonlinear_mlp.data.mnist import get_mnist_loaders
from nonlinear_mlp.data.cifar10 import get_cifar10_loaders
from nonlinear_mlp.data.tabular import load_adult
from nonlinear_mlp.models.mlp import MLP
from nonlinear_mlp.models.cifar_head import build_resnet18_with_mlp_head
from nonlinear_mlp.train import train_one_epoch, evaluate, save_checkpoint
from nonlinear_mlp.utils.profiler import count_params, model_linear_flops
from nonlinear_mlp.utils.metrics import measure_inference_latency, memory_usage_mb
from nonlinear_mlp.analysis.activation_stats import ActivationTracker
from nonlinear_mlp.pruning.post_training import decide_linearization, apply_layer_linearization
import json
import os
import random
import numpy as np
import time

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def build_model(cfg: ExperimentConfig, input_dim=None, num_classes=None):
    if cfg.model == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dims=cfg.hidden_dims,
            num_classes=num_classes or cfg.num_classes,
            approach=cfg.approach,
            fixed_cfg=cfg.fixed.__dict__,
            gating_cfg=cfg.gating.__dict__,
            layerwise_cfg=cfg.layerwise.__dict__
        )
    elif cfg.model == "resnet18_head":
        return build_resnet18_with_mlp_head(
            num_classes=num_classes or cfg.num_classes,
            head_hidden=cfg.hidden_dims,
            approach=cfg.approach,
            fixed_cfg=cfg.fixed.__dict__,
            gating_cfg=cfg.gating.__dict__,
            layerwise_cfg=cfg.layerwise.__dict__,
            pretrained=False,
            freeze_backbone=False
        )
    else:
        raise ValueError(f"Unknown model {cfg.model}")

def get_data(cfg: ExperimentConfig):
    if cfg.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=cfg.training.batch_size)
        input_dim = 28*28
        num_classes = 10
    elif cfg.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(batch_size=cfg.training.batch_size)
        input_dim = None  # images
        num_classes = 10
    elif cfg.dataset == "tabular_adult":
        train_loader, test_loader, in_dim, n_classes = load_adult(batch_size=cfg.training.batch_size)
        input_dim = in_dim
        num_classes = n_classes
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset}")
    return train_loader, test_loader, input_dim, num_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config (optional)")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--approach", type=str, default="fixed")
    parser.add_argument("--linear_ratio", type=float, default=0.5)
    parser.add_argument("--pattern", type=str, default="structured")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--run_name", type=str, default="exp")
    parser.add_argument("--gating", action="store_true")
    parser.add_argument("--pruning", action="store_true")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        cfg = ExperimentConfig(**cfg_dict)
    else:
        cfg = ExperimentConfig()
        cfg.dataset = args.dataset
        cfg.approach = "gating" if args.gating else args.approach
        cfg.fixed.linear_ratio = args.linear_ratio
        cfg.fixed.pattern = args.pattern
        cfg.training.epochs = args.epochs
        cfg.logging.run_name = args.run_name
        cfg.pruning.enabled = args.pruning

    set_seed(cfg.training.seed)
    device = cfg.training.device if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, input_dim, num_classes = get_data(cfg)
    model = build_model(cfg, input_dim=input_dim, num_classes=num_classes).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (device.startswith("cuda") and cfg.training.amp) else None

    meta = {
        "param_counts": count_params(model),
        "approx_linear_flops": model_linear_flops(model),
        "config": json.loads(cfg.to_json())
    }

    history = []
    best_val = -1
    patience_counter = 0

    for epoch in range(1, cfg.training.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optim, device, scaler, epoch, cfg, log_interval=cfg.logging.log_interval)
        val_stats = evaluate(model, test_loader, device, cfg)
        gating_stats = model.gather_gating_stats() if hasattr(model, "gather_gating_stats") else []
        record = {**train_stats, **val_stats, "epoch": epoch, "gating": gating_stats}
        history.append(record)
        print(f"[Epoch {epoch}] Val Acc: {val_stats['val_acc']:.2f}")
        if val_stats["val_acc"] > best_val:
            best_val = val_stats["val_acc"]
            patience_counter = 0
            if cfg.logging.save_checkpoints:
                save_checkpoint(model, optim, cfg, epoch, record, f"{cfg.logging.output_dir}/{cfg.logging.run_name}")
        else:
            patience_counter += 1
            if cfg.training.early_stop_patience and patience_counter >= cfg.training.early_stop_patience:
                print("Early stopping triggered.")
                break

    # Post-training gating harden
    if cfg.approach == "gating":
        print("Hardening gates...")
        model.harden_gates(threshold=cfg.gating.hard_threshold)
        hardened_eval = evaluate(model, test_loader, device, cfg)
        print("After harden val acc:", hardened_eval)

    # Post-training pruning
    if cfg.pruning.enabled:
        print("Collecting activation stats for pruning...")
        tracker = ActivationTracker(max_batches=50)
        tracker.register(model)
        stats = tracker.run(model, train_loader, device)
        tracker.remove()
        decisions = decide_linearization(
            stats,
            pos_thresh=cfg.pruning.activation_positive_ratio_threshold,
            neg_thresh=cfg.pruning.dead_negative_ratio_threshold,
            nonlinear_score_thresh=cfg.pruning.nonlinear_contribution_threshold
        )
        model = apply_layer_linearization(model, decisions)
        # Optional fine-tuning
        for ft_epoch in range(cfg.pruning.fine_tune_epochs):
            _ = train_one_epoch(model, train_loader, optim, device, scaler, ft_epoch, cfg)
        prune_eval = evaluate(model, test_loader, device, cfg)
        print("Prune evaluation:", prune_eval)

    # Latency measurement
    latency = measure_inference_latency(model, test_loader, device)
    meta["latency"] = latency
    meta["memory_mb"] = memory_usage_mb()

    os.makedirs(f"{cfg.logging.output_dir}/{cfg.logging.run_name}", exist_ok=True)
    with open(f"{cfg.logging.output_dir}/{cfg.logging.run_name}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{cfg.logging.output_dir}/{cfg.logging.run_name}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Experiment complete. Results saved.")

if __name__ == "__main__":
    main()