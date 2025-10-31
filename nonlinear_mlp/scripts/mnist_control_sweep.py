#!/usr/bin/env python
"""
Sweep control MLPs (structural width reduction) on MNIST.

For a target nonlinearity fraction K% per layer, we build a control model that keeps K% of neurons
structurally (width-reduced layers). We map this to the existing 'fixed.linear_ratio' flag by
setting linear_ratio = 1 - K.

Defaults:
- Ratios kept K ∈ {25, 50, 75, 100} (%)
- 20 epochs (override with --epochs)
- Idempotent: skips runs with runs/<run_name>/meta.json

Usage:
  python -m nonlinear_mlp.scripts.mnist_control_sweep --wandb_project nonlinear-mlp --mode online
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
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = args.group or f"mnist_mlp_control_{timestamp}"
    enable_wandb = args.wandb_project is not None and args.mode != "disabled"

    base_cmd = ["python", "-m", "nonlinear_mlp.experiments.run_experiment"]

    # Keep (nonlinearity) percentages K to structurally retain per hidden layer
    keeps = [25, 50, 75, 100]  # exclude 0
    failures = []
    for K in keeps:
        keep_frac = K / 100.0
        # Map to fixed.linear_ratio (fraction linear) so the control model can compute keep = 1 - linear_ratio
        linear_ratio = 1.0 - keep_frac
        run_name = f"mnist_control_{K}"
        cmd = base_cmd + [
            "--dataset", "mnist",
            "--model", "mlp_control",
            "--approach", "fixed",                     # approach is unused by mlp_control, but harmless to set
            "--linear_ratio", str(linear_ratio),       # used by control model to compute keep ratio
            "--pattern", "structured",                 # unused here, for consistency with other runs
            "--epochs", str(args.epochs),
            "--run_name", run_name,
        ]
        add_wandb_flags(
            cmd,
            args.wandb_project,
            args.mode,
            args.wandb_entity,
            group,
            f"dataset:mnist,model:mlp_control,keep_nonlin:{K}",
            enable_wandb,
        )
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
