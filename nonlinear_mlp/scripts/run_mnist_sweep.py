import subprocess, json, os, time

ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
base_cmd = ["python", "-m", "nonlinear_mlp.experiments.run_experiment"]

failures = []

for r in ratios:
    run_name = f"mnist_fixed_{int(r*100)}"
    out_dir = os.path.join("runs", run_name)
    # Skip if the run already exists and has meta.json (idempotent)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        print(f"Skipping existing run: {run_name}")
        continue

    cmd = base_cmd + [
        "--dataset", "mnist",
        "--approach", "fixed",
        "--linear_ratio", str(r),
        "--pattern", "structured",
        "--epochs", "20",
        "--run_name", run_name,
        "--wandb",
        "--wandb_project", "nonlinear-mlp",
    ]
    print("Running:", " ".join(cmd))
    try:
        # You can add 'capture_output=True, text=True' if you want to collect logs
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed: {run_name} (exit code {e.returncode})")
        failures.append(run_name)

print("Sweep complete.")
if failures:
    print("Failures:", failures)
