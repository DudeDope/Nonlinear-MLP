#!/usr/bin/env python
"""
Simple sweep runner (idempotent) that mirrors the MNIST ratio loop
and extends to CIFAR-10 head and Tabular Adult.

Examples:
  python nonlinear_mlp/scripts/simple_sweep.py --wandb_project nonlinear-mlp --mode online
  python nonlinear_mlp/scripts/simple_sweep.py --include mnist_fixed,cifar_fixed --skip_existing  # default

What it does:
- MNIST MLP: fixed ratios, gating, pruning
- CIFAR-10 ResNet-18 head: fixed ratios, gating, pruning
- Tabular Adult MLP (optional): fixed + gating

It skips any run whose runs/<run_name>/meta.json already exists.
"""

import argparse
import os
import subprocess
from datetime import datetime

def run_once(run_name: str, cmd: list, skip_existing: bool = True) -> bool:
    out_dir = os.path.join("runs", run_name)
    meta_path = os.path.join(out_dir, "meta.json")
    if skip_existing and os.path.exists(meta_path):
        print(f"[SKIP] {run_name} (found {meta_path})")
        return True
    print("Running:", " ".join(str(x) for x in cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {run_name} (exit={e.returncode})")
        return False

def add_wandb_flags(cmd: list, project: str | None, mode: str, entity: str | None, group: str | None, tags: str | None, enable: bool) -> None:
    if not enable or not project:
        return
    cmd.extend(["--wandb", "--wandb_project", project, "--wandb_mode", mode])
    if entity:
        cmd.extend(["--wandb_entity", entity])
    if group:
        cmd.extend(["--wandb_group", group])
    if tags:
        cmd.extend(["--wandb_tags", tags])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project (enable logging if provided)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (user/team)")
    parser.add_argument("--mode", type=str, default="disabled", choices=["online", "offline", "disabled"], help="W&B mode")
    parser.add_argument("--group", type=str, default=None, help="W&B group name")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip runs with existing meta.json")
    parser.add_argument("--include", type=str, default="mnist_fixed,mnist_gating,mnist_pruning,cifar_fixed,cifar_gating,cifar_pruning,adult_fixed,adult_gating",
                        help="Comma list of jobs to include")
    parser.add_argument("--epochs_mnist", type=int, default=10)
    parser.add_argument("--epochs_cifar", type=int, default=60)
    parser.add_argument("--epochs_adult", type=int, default=30)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = args.group or f"simple_sweep_{timestamp}"
    enable_wandb = args.wandb_project is not None and args.mode != "disabled"

    # Base runner
    base_cmd = ["python", "-m", "nonlinear_mlp.experiments.run_experiment"]

    # Parse which sets to include
    include = {t.strip() for t in args.include.split(",") if t.strip()}

    failures = []

    # 1) MNIST MLP: fixed ratios
    if "mnist_fixed" in include:
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        for r in ratios:
            run_name = f"mnist_fixed_{int(r*100)}"
            cmd = base_cmd + [
                "--dataset", "mnist",
                "--model", "mlp",
                "--approach", "fixed",
                "--linear_ratio", str(r),
                "--pattern", "structured",
                "--epochs", str(args.epochs_mnist),
                "--run_name", run_name
            ]
            add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                            f"dataset:mnist,approach:fixed,ratio:{r}", enable_wandb)
            if not run_once(run_name, cmd, args.skip_existing):
                failures.append(run_name)

    # 2) MNIST MLP: gating (single, simple)
    if "mnist_gating" in include:
        run_name = "mnist_gating"
        cmd = base_cmd + [
            "--dataset", "mnist",
            "--model", "mlp",
            "--gating",
            "--epochs", str(args.epochs_mnist),
            "--run_name", run_name
        ]
        add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                        "dataset:mnist,approach:gating", enable_wandb)
        if not run_once(run_name, cmd, args.skip_existing):
            failures.append(run_name)

    # 3) MNIST MLP: pruning from ReLU (post-training)
    if "mnist_pruning" in include:
        run_name = "mnist_prune_from_relu"
        cmd = base_cmd + [
            "--dataset", "mnist",
            "--model", "mlp",
            "--approach", "fixed",
            "--linear_ratio", "0.0",
            "--pruning",
            "--epochs", str(args.epochs_mnist),
            "--run_name", run_name
        ]
        add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                        "dataset:mnist,approach:pruning", enable_wandb)
        if not run_once(run_name, cmd, args.skip_existing):
            failures.append(run_name)

    # 4) CIFAR-10 ResNet-18 head: fixed ratios
    if "cifar_fixed" in include:
        ratios = [0.25, 0.5, 0.75]
        for r in ratios:
            run_name = f"cifar_head_fixed_{int(r*100)}"
            cmd = base_cmd + [
                "--dataset", "cifar10",
                "--model", "resnet18_head",
                "--approach", "fixed",
                "--linear_ratio", str(r),
                "--pattern", "structured",
                "--epochs", str(args.epochs_cifar),
                "--run_name", run_name
            ]
            add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                            f"dataset:cifar10,model:resnet18_head,approach:fixed,ratio:{r}", enable_wandb)
            if not run_once(run_name, cmd, args.skip_existing):
                failures.append(run_name)

    # 5) CIFAR-10 ResNet-18 head: gating
    if "cifar_gating" in include:
        run_name = "cifar_head_gating"
        cmd = base_cmd + [
            "--dataset", "cifar10",
            "--model", "resnet18_head",
            "--gating",
            "--epochs", str(args.epochs_cifar),
            "--run_name", run_name
        ]
        add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                        "dataset:cifar10,model:resnet18_head,approach:gating", enable_wandb)
        if not run_once(run_name, cmd, args.skip_existing):
            failures.append(run_name)

    # 6) CIFAR-10 ResNet-18 head: pruning from ReLU
    if "cifar_pruning" in include:
        run_name = "cifar_head_prune_from_relu"
        cmd = base_cmd + [
            "--dataset", "cifar10",
            "--model", "resnet18_head",
            "--approach", "fixed",
            "--linear_ratio", "0.0",
            "--pruning",
            "--epochs", str(args.epochs_cifar),
            "--run_name", run_name
        ]
        add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                        "dataset:cifar10,model:resnet18_head,approach:pruning", enable_wandb)
        if not run_once(run_name, cmd, args.skip_existing):
            failures.append(run_name)

    # 7) Tabular Adult MLP: fixed + gating (optional small runs)
    if "adult_fixed" in include:
        run_name = "adult_fixed_50"
        cmd = base_cmd + [
            "--dataset", "tabular_adult",
            "--model", "mlp",
            "--approach", "fixed",
            "--linear_ratio", "0.5",
            "--epochs", str(args.epochs_adult),
            "--run_name", run_name
        ]
        add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                        "dataset:adult,approach:fixed,ratio:0.5", enable_wandb)
        if not run_once(run_name, cmd, args.skip_existing):
            failures.append(run_name)

    if "adult_gating" in include:
        run_name = "adult_gating"
        cmd = base_cmd + [
            "--dataset", "tabular_adult",
            "--model", "mlp",
            "--gating",
            "--epochs", str(args.epochs_adult),
            "--run_name", run_name
        ]
        add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                        "dataset:adult,approach:gating", enable_wandb)
        if not run_once(run_name, cmd, args.skip_existing):
            failures.append(run_name)

    if failures:
        print("\nSome runs failed:")
        for r in failures:
            print(" -", r)
    else:
        print("\nAll selected runs completed or were skipped (idempotent).")

if __name__ == "__main__":
    main()
