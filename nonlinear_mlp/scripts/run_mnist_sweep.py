import subprocess, json, os

ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
for r in ratios:
    run_name = f"mnist_fixed_{int(r*100)}"
    cmd = [
        "python", "-m", "nonlinear_mlp.experiments.run_experiment",
        "--dataset", "mnist",
        "--approach", "fixed",
        "--linear_ratio", str(r),
        "--pattern", "structured",
        "--epochs", "10",
        "--run_name", run_name
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
print("Sweep complete.")