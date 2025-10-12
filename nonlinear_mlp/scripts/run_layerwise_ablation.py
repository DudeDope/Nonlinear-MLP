import json, subprocess, itertools, os

# Example layer-wise ratios for a 3-layer MLP
layer_ratios_grid = [
    [0.2, 0.5, 0.8],
    [0.0, 0.5, 1.0],
    [0.5, 0.5, 0.5],
]

for ratios in layer_ratios_grid:
    cfg = {
        "dataset": "mnist",
        "approach": "layerwise",
        "fixed": {
            "linear_ratio": 0.5,
            "pattern": "structured",
            "per_layer": ratios
        },
        "layerwise": { "enabled": True, "schedule": { str(i): r for i, r in enumerate(ratios) } },
        "training": { "epochs": 10 },
        "logging": { "run_name": f"mnist_layerwise_{'_'.join(str(int(r*100)) for r in ratios)}" }
    }
    cfg_path = f"tmp_layerwise_{'_'.join(str(int(r*100)) for r in ratios)}.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    cmd = ["python", "-m", "nonlinear_mlp.experiments.run_experiment", "--config", cfg_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)