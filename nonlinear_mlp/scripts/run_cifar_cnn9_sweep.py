#!/usr/bin/env python
"""
Simple sweep for PlainCNN9 on CIFAR-10:
- N% ∈ {0, 25, 50, 75, 100} linear_ratio (structured removal of ReLUs from deeper layers)
- 200 epochs, SGD(momentum=0.9), lr=0.1, weight_decay=5e-4, batch=128 (these are defaults in runner for cnn9_plain)
- Idempotent: skips if runs/<run_name>/meta.json exists

Usage:
  python -m nonlinear_mlp.scripts.run_cifar_cnn9_sweep --wandb_project nonlinear-mlp --mode online
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
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = args.group or f"cnn9_sweep_{timestamp}"
    enable_wandb = args.wandb_project is not None and args.mode != "disabled"

    base_cmd = ["python", "-m", "nonlinear_mlp.experiments.run_experiment"]

    failures = []
    for r in [0.0, 0.25, 0.5, 0.75, 1.0]:
        run_name = f"cifar_cnn9_fixed_{int(r*100)}"
        cmd = base_cmd + [
            "--dataset", "cifar10",
            "--model", "cnn9_plain",
            "--approach", "fixed",
            "--linear_ratio", str(r),
            "--pattern", "structured",
            "--epochs", str(args.epochs),
            "--run_name", run_name,
        ]
        # Make sure batch size is set via config: we use cfg.training.batch_size; if your config dataclass
        # already has a default of 128, this is not necessary. If not, add env var override or adjust config file.
        # Here we rely on existing defaults in your config to keep the script simple.

        add_wandb_flags(cmd, args.wandb_project, args.mode, args.wandb_entity, group,
                        f"dataset:cifar10,model:cnn9_plain,approach:fixed,ratio:{r}", enable_wandb)
        if not run_once(run_name, cmd, args.skip_existing):
            failures.append(run_name)

    if failures:
        print("\nSome runs failed:")
        for rn in failures:
            print(" -", rn)
    else:
        print("\nAll runs completed or were skipped (idempotent).")

if __name__ == "__main__":
    main()