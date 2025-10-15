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

# Optional imports (guarded)
try:
    from nonlinear_mlp.data.imagenet import get_imagenet_loaders
    IMAGENET_AVAILABLE = True
except Exception:
    IMAGENET_AVAILABLE = False

try:
    from nonlinear_mlp.models.resnet50_head import build_resnet50_with_mlp_head
    RESNET50_AVAILABLE = True
except Exception:
    RESNET50_AVAILABLE = False

def _maybe_init_wandb(cfg: ExperimentConfig, out_dir: str):
    wb = {"enabled": False, "wandb": None}
    if not getattr(cfg.logging, "wandb_enabled", False):
        return wb
    try:
        import wandb
    except Exception as e:
        print(f"[W&B] wandb not installed; disable logging ({e}).")
        return wb

    # Determine dir for W&B files (per-run folder)
    wandb_dir = getattr(cfg.logging, "wandb_dir", None) or os.path.join(out_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    # Map settings
    project = getattr(cfg.logging, "wandb_project", "nonlinear-mlp")
    entity = getattr(cfg.logging, "wandb_entity", None)
    group = getattr(cfg.logging, "wandb_group", None)
    tags = getattr(cfg.logging, "wandb_tags", []) or []
    mode = getattr(cfg.logging, "wandb_mode", None)  # 'online'|'offline'|'disabled'

    # Initialize
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        tags=tags,
        name=cfg.logging.run_name,
        dir=wandb_dir,
        mode=mode,  # None -> default online
        config=json.loads(cfg.to_json()),
        reinit=True
    )
    wb["enabled"] = True
    wb["wandb"] = wandb
    wb["run"] = run
    wb["dir"] = wandb_dir
    print(f"[W&B] Initialized run at {wandb_dir} (project={project}, entity={entity}, group={group})")
    return wb

def _wandb_log_epoch(wb, epoch_record: dict, epoch_num: int):
    if not wb.get("enabled", False):
        return
    wandb = wb["wandb"]
    to_log = {
        "epoch": epoch_num,
        "train/loss": epoch_record.get("train_loss"),
        "train/acc": epoch_record.get("train_acc"),
        "val/loss": epoch_record.get("val_loss"),
        "val/acc": epoch_record.get("val_acc"),
        "time/train_s": epoch_record.get("train_time_s"),
    }
    wandb.log({k: v for k, v in to_log.items() if v is not None}, step=epoch_num)

    # Log gating summary if present
    gating_list = epoch_record.get("gating", [])
    if gating_list:
        # layer-wise scalar stats
        for g in gating_list:
            layer_idx = g.get("layer")
            for key in ["alpha_mean", "alpha_median", "alpha_min", "alpha_max", "alpha_lt_0.1", "alpha_lt_0.25", "alpha_gt_0.9"]:
                if key in g:
                    wandb.log({f"gating/{key}/layer_{layer_idx}": g[key]}, step=epoch_num)

def _wandb_log_alpha_histograms(wb, model, epoch_num: int, log_hist: bool = True):
    if not (wb.get("enabled", False) and log_hist):
        return
    try:
        from nonlinear_mlp.layers.gated import GatedActivationLayer
    except Exception:
        return
    import torch
    wandb = wb["wandb"]
    if getattr(model, "feature_layers", None) is None:
        return
    for i, layer in enumerate(model.feature_layers):
        if isinstance(layer, GatedActivationLayer):
            with torch.no_grad():
                a = layer.alpha().detach().float().cpu().numpy()
            # histogram per layer
            try:
                hist = wandb.Histogram(a)
                wandb.log({f"gating/alpha_hist_layer_{i}": hist}, step=epoch_num)
            except Exception:
                pass

def _wandb_log_meta(wb, meta: dict):
    if not wb.get("enabled", False):
        return
    wandb = wb["wandb"]
    # push meta summary fields
    lat = meta.get("latency", {})
    pc = meta.get("param_counts", {})
    for k, v in {
        "latency/mean_s": lat.get("mean_latency_s"),
        "latency/p50_s": lat.get("p50_latency_s"),
        "latency/p90_s": lat.get("p90_latency_s"),
        "throughput/samples_per_s": lat.get("samples_per_second"),
        "params/total": pc.get("total_params"),
        "params/trainable": pc.get("trainable_params"),
        "flops/approx_linear": meta.get("approx_linear_flops"),
        "memory/mb": meta.get("memory_mb"),
    }.items():
        if v is not None:
            wandb.run.summary[k] = v

def _wandb_finish(wb):
    if wb.get("enabled", False):
        try:
            wb["run"].finish()
        except Exception:
            pass
# ----------------------------------------------------------


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
    elif cfg.model == "resnet50_head":
        if not RESNET50_AVAILABLE:
            raise RuntimeError("resnet50_head requested but resnet50_head module not available.")
        return build_resnet50_with_mlp_head(
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
        input_dim = None
        num_classes = 10
    elif cfg.dataset == "imagenet":
        if not IMAGENET_AVAILABLE:
            raise RuntimeError("Imagenet dataset loader not available. Ensure imagenet.py exists.")
        train_loader, test_loader, num_classes = get_imagenet_loaders(
            root="data/imagenet",
            batch_size=cfg.training.batch_size
        )
        input_dim = None
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
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--approach", type=str, default="fixed")
    parser.add_argument("--linear_ratio", type=float, default=0.5)
    parser.add_argument("--pattern", type=str, default="structured")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--run_name", type=str, default="exp")
    parser.add_argument("--gating", action="store_true")
    parser.add_argument("--pruning", action="store_true")
    # Optional W&B CLI conveniences (override config if provided)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--wandb_mode", type=str, default=None, help="online|offline|disabled")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        cfg = ExperimentConfig(**cfg_dict)
        # Allow CLI override of run_name/model and W&B quick flags
        if args.run_name:
            cfg.logging.run_name = args.run_name
        if args.model:
            cfg.model = args.model
        if args.wandb:
            cfg.logging.wandb_enabled = True
        if args.wandb_project:
            cfg.logging.wandb_project = args.wandb_project
        if args.wandb_entity:
            cfg.logging.wandb_entity = args.wandb_entity
        if args.wandb_group:
            cfg.logging.wandb_group = args.wandb_group
        if args.wandb_tags:
            cfg.logging.wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        if args.wandb_mode:
            cfg.logging.wandb_mode = args.wandb_mode
    else:
        cfg = ExperimentConfig()
        cfg.dataset = args.dataset
        cfg.model = args.model
        cfg.approach = "gating" if args.gating else args.approach
        cfg.fixed.linear_ratio = args.linear_ratio
        cfg.fixed.pattern = args.pattern
        cfg.training.epochs = args.epochs
        cfg.logging.run_name = args.run_name
        cfg.pruning.enabled = args.pruning
        # W&B quick flags
        if args.wandb:
            cfg.logging.wandb_enabled = True
        if args.wandb_project:
            cfg.logging.wandb_project = args.wandb_project
        if args.wandb_entity:
            cfg.logging.wandb_entity = args.wandb_entity
        if args.wandb_group:
            cfg.logging.wandb_group = args.wandb_group
        if args.wandb_tags:
            cfg.logging.wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        if args.wandb_mode:
            cfg.logging.wandb_mode = args.wandb_mode

    # Seed & device
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    device = cfg.training.device if torch.cuda.is_available() else "cpu"

    # Data & model
    train_loader, test_loader, input_dim, num_classes = get_data(cfg)
    model = build_model(cfg, input_dim=input_dim, num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    # Modern AMP scaler (fallback to legacy if needed)
    try:
        scaler = torch.amp.GradScaler("cuda") if (device.startswith("cuda") and cfg.training.amp) else None
    except Exception:
        from torch.cuda.amp import GradScaler as _LegacyGradScaler
        scaler = _LegacyGradScaler() if (device.startswith("cuda") and cfg.training.amp) else None

    out_dir = f"{cfg.logging.output_dir}/{cfg.logging.run_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize W&B (per-run dir under runs/<run_name>/wandb)
    wb = _maybe_init_wandb(cfg, out_dir)

    meta = {
        "param_counts": count_params(model),
        "approx_linear_flops": model_linear_flops(model),
        "config": json.loads(cfg.to_json())
    }
    # If W&B enabled, store some meta in config/summary
    if wb.get("enabled", False):
        _wandb_log_meta(wb, {"param_counts": meta["param_counts"], "approx_linear_flops": meta["approx_linear_flops"]})

    history = []
    best_val = -1
    patience_counter = 0

    for epoch in range(1, cfg.training.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, scaler, epoch, cfg, log_interval=cfg.logging.log_interval)
        val_stats = evaluate(model, test_loader, device, cfg)
        gating_stats = model.gather_gating_stats() if hasattr(model, "gather_gating_stats") else []
        record = {**train_stats, **val_stats, "epoch": epoch, "gating": gating_stats}
        history.append(record)

        print(f"[Epoch {epoch}] Val Acc: {val_stats['val_acc']:.2f}")

        # W&B per-epoch logging
        _wandb_log_epoch(wb, record, epoch)
        _wandb_log_alpha_histograms(wb, model, epoch, log_hist=getattr(cfg.logging, "wandb_log_alpha_hist", True))

        if val_stats["val_acc"] > best_val:
            best_val = val_stats["val_acc"]
            patience_counter = 0
            if cfg.logging.save_checkpoints:
                save_checkpoint(model, optimizer, cfg, epoch, record, out_dir)
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
        history.append({"epoch": "hardened", **hardened_eval, "gating": model.gather_gating_stats()})
        # Log post-harden
        if wb.get("enabled", False):
            wb["wandb"].log({"val/acc_hardened": hardened_eval.get("val_acc")})

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
        model = apply_layer_linearization(model, decisions, verbose=True)
        # Optional fine-tune
        for ft_epoch in range(cfg.pruning.fine_tune_epochs):
            _ = train_one_epoch(model, train_loader, optimizer, device, scaler, ft_epoch, cfg, log_interval=cfg.logging.log_interval)
        prune_eval = evaluate(model, test_loader, device, cfg)
        print("Prune evaluation:", prune_eval)
        history.append({"epoch": "post_prune", **prune_eval})
        if wb.get("enabled", False):
            wb["wandb"].log({"val/acc_post_prune": prune_eval.get("val_acc")})

    # Latency measurement
    latency = measure_inference_latency(model, test_loader, device)
    meta["latency"] = latency
    meta["memory_mb"] = memory_usage_mb()
    _wandb_log_meta(wb, meta)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Experiment complete. Results saved.")

    _wandb_finish(wb)

if __name__ == "__main__":
    main()
