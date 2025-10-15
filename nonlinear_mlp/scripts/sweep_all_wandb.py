"""
Sweep runner that launches multiple experiments and logs to Weights & Biases.
Covers: Fixed ratios, Gating grid, Layerwise schedules, Pruning on MNIST and CIFAR-10 head.

Usage:
  python scripts/sweep_all_wandb.py --project nonlinear-mlp --entity <wandb_user_or_team> --mode online
"""
import argparse
import itertools
import json
import subprocess
from pathlib import Path
from datetime import datetime


def run_cmd(cmd):
    print("Running:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def write_config(path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def make_common_cfg(dataset, model, approach, run_name, **kwargs):
    cfg = {
        "dataset": dataset,
        "model": model,
        "approach": approach,
        "logging": {"run_name": run_name},
        "training": {"epochs": kwargs.get("epochs", 10), "batch_size": kwargs.get("batch_size", 128)},
    }
    if model == "mlp":
        cfg["hidden_dims"] = kwargs.get("hidden_dims", [512, 256, 128])
        cfg["num_classes"] = kwargs.get("num_classes", 10)
    elif model in ("resnet18_head", "resnet50_head"):
        cfg["hidden_dims"] = kwargs.get("hidden_dims", [512, 256])
        cfg["num_classes"] = kwargs.get("num_classes", 10)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True, help="wandb project")
    parser.add_argument("--entity", type=str, default=None, help="wandb entity (user/team)")
    parser.add_argument("--mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default="tmp_sweep_configs")
    parser.add_argument("--epochs_mnist", type=int, default=10)
    parser.add_argument("--epochs_cifar", type=int, default=60)
    args = parser.parse_args()

    base = Path(args.base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = args.group or f"sweep_{timestamp}"

    jobs = []

    # 1) MNIST Fixed ratios
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for pattern in ["structured"]:
            rn = f"mnist_fixed_{int(ratio*100)}_{pattern}"
            cfg = make_common_cfg("mnist", "mlp", "fixed", rn, epochs=args.epochs_mnist)
            cfg["fixed"] = {"linear_ratio": ratio, "pattern": pattern}
            path = base / f"{rn}.json"
            write_config(path, cfg)
            jobs.append([
                "python", "-m", "nonlinear_mlp.experiments.run_experiment",
                "--config", str(path),
                "--wandb",  # enable W&B
                "--wandb_project", args.project,
                "--wandb_entity", args.entity or "",
                "--wandb_mode", args.mode,
                "--wandb_group", group,
                "--wandb_tags", f"dataset:mnist,approach:fixed,pattern:{pattern}",
            ])

    # 2) MNIST Gating grid
    for init_alpha, entropy_reg, l1_reg in itertools.product([0.6, 0.8], [0.0, 0.002], [0.0, 0.01]):
        rn = f"mnist_gating_a{int(100*init_alpha)}_ent{entropy_reg}_l1{l1_reg}"
        cfg = make_common_cfg("mnist", "mlp", "gating", rn, epochs=args.epochs_mnist)
        cfg["gating"] = {
            "enabled": True,
            "init_alpha": init_alpha,
            "entropy_reg": entropy_reg,
            "l1_reg": l1_reg,
            "hard_threshold": 0.1,
            "temperature": 1.0,
            "clamp": True,
        }
        path = base / f"{rn}.json"
        write_config(path, cfg)
        jobs.append([
            "python", "-m", "nonlinear_mlp.experiments.run_experiment",
            "--config", str(path),
            "--wandb",
            "--wandb_project", args.project,
            "--wandb_entity", args.entity or "",
            "--wandb_mode", args.mode,
            "--wandb_group", group,
            "--wandb_tags", "dataset:mnist,approach:gating",
        ])

    # 3) MNIST Layerwise schedules
    layerwise_grid = [
        [0.2, 0.5, 0.8],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 0.5],
    ]
    for ratios in layerwise_grid:
        rn = "mnist_layerwise_" + "_".join(str(int(r * 100)) for r in ratios)
        cfg = make_common_cfg("mnist", "mlp", "layerwise", rn, epochs=args.epochs_mnist)
        cfg["fixed"] = {"linear_ratio": 0.5, "pattern": "structured", "per_layer": ratios}
        cfg["layerwise"] = {"enabled": True, "schedule": {str(i): r for i, r in enumerate(ratios)}}
        path = base / f"{rn}.json"
        write_config(path, cfg)
        jobs.append([
            "python", "-m", "nonlinear_mlp.experiments.run_experiment",
            "--config", str(path),
            "--wandb",
            "--wandb_project", args.project,
            "--wandb_entity", args.entity or "",
            "--wandb_mode", args.mode,
            "--wandb_group", group,
            "--wandb_tags", "dataset:mnist,approach:layerwise",
        ])

    # 4) MNIST Pruning from ReLU
    rn = "mnist_prune_from_relu"
    cfg = make_common_cfg("mnist", "mlp", "fixed", rn, epochs=args.epochs_mnist)
    cfg["fixed"] = {"linear_ratio": 0.0, "pattern": "structured"}
    cfg["pruning"] = {"enabled": True}
    path = base / f"{rn}.json"
    write_config(path, cfg)
    jobs.append([
        "python", "-m", "nonlinear_mlp.experiments.run_experiment",
        "--config", str(path),
        "--wandb",
        "--wandb_project", args.project,
        "--wandb_entity", args.entity or "",
        "--wandb_mode", args.mode,
        "--wandb_group", group,
        "--wandb_tags", "dataset:mnist,approach:pruning",
    ])

    # 5) CIFAR-10 Fixed ratios on ResNet-18 head
    for ratio in [0.25, 0.5, 0.75]:
        rn = f"cifar_head_fixed_{int(100*ratio)}"
        cfg = make_common_cfg("cifar10", "resnet18_head", "fixed", rn, epochs=args.epochs_cifar)
        cfg["fixed"] = {"linear_ratio": ratio, "pattern": "structured"}
        path = base / f"{rn}.json"
        write_config(path, cfg)
        jobs.append([
            "python", "-m", "nonlinear_mlp.experiments.run_experiment",
            "--config", str(path),
            "--wandb",
            "--wandb_project", args.project,
            "--wandb_entity", args.entity or "",
            "--wandb_mode", args.mode,
            "--wandb_group", group,
            "--wandb_tags", "dataset:cifar10,model:resnet18_head,approach:fixed",
        ])

    # 6) CIFAR-10 Gating on head
    rn = "cifar_head_gating"
    cfg = make_common_cfg("cifar10", "resnet18_head", "gating", rn, epochs=args.epochs_cifar)
    cfg["gating"] = {"enabled": True, "init_alpha": 0.8, "entropy_reg": 0.002, "l1_reg": 0.0, "hard_threshold": 0.1}
    path = base / f"{rn}.json"
    write_config(path, cfg)
    jobs.append([
        "python", "-m", "nonlinear_mlp.experiments.run_experiment",
        "--config", str(path),
        "--wandb",
        "--wandb_project", args.project,
        "--wandb_entity", args.entity or "",
        "--wandb_mode", args.mode,
        "--wandb_group", group,
        "--wandb_tags", "dataset:cifar10,model:resnet18_head,approach:gating",
    ])

    # 7) CIFAR-10 Pruning from ReLU on head
    rn = "cifar_head_prune_from_relu"
    cfg = make_common_cfg("cifar10", "resnet18_head", "fixed", rn, epochs=args.epochs_cifar)
    cfg["fixed"] = {"linear_ratio": 0.0, "pattern": "structured"}
    cfg["pruning"] = {"enabled": True}
    path = base / f"{rn}.json"
    write_config(path, cfg)
    jobs.append([
        "python", "-m", "nonlinear_mlp.experiments.run_experiment",
        "--config", str(path),
        "--wandb",
        "--wandb_project", args.project,
        "--wandb_entity", args.entity or "",
        "--wandb_mode", args.mode,
        "--wandb_group", group,
        "--wandb_tags", "dataset:cifar10,model:resnet18_head,approach:pruning",
    ])

    for cmd in jobs:
        run_cmd(cmd)

    print("Sweep complete.")


if __name__ == "__main__":
    main()
