# Controlled Experiments with Mixed-Activation Neural Networks

This repository studies neural networks whose hidden units can use different
activation families. It provides controlled experiment entry points for fixed,
mixed, gated, and nonlinearity-dropout variants on image and tabular datasets.
A concise research summary is available on my
[portfolio](https://dudedope.github.io/projects/nonlinear-mlp/).

## Evidence status

This is an ongoing experimental codebase. The repository contains training
infrastructure, sweep scripts, metadata, checkpoints, and metric histories; it
does not yet contain a stable cross-dataset result table supporting a broad
generalization claim. Output schemas and plotting code document how evidence
is collected, not what the final finding is.

## Implemented scope

- MNIST, CIFAR-10, CIFAR-100, ImageNet-style, and tabular data loaders.
- Fixed, learned-mixture, gated, control, and nonlinearity-dropout models.
- Optional post-training pruning and activation-statistics utilities.
- Repeated sweeps for MNIST and CIFAR-family experiments.
- JSON metadata, per-epoch histories, checkpoints, and optional Weights &
  Biases logging.

## Installation

```bash
git clone https://github.com/DudeDope/Nonlinear-MLP.git
cd Nonlinear-MLP
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

## Minimal controlled comparison

Run a baseline and a mixed-activation model with the same dataset, epoch budget,
and default seed:

```bash
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --approach fixed --linear_ratio 0.0 --epochs 1 --run_name mnist_relu_smoke
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --approach fixed --linear_ratio 0.5 --epochs 1 --run_name mnist_mixed_smoke
```

Each run writes its configuration, history, and checkpoints beneath the output
directory chosen by the experiment runner. A meaningful comparison should hold
the data split, preprocessing, optimizer, budget, and seed fixed and should be
repeated across several seeds; the two smoke commands above only verify the
pipeline.

## Other entry points

- `nonlinear_mlp/scripts/run_mnist_sweep.py` — MNIST ratio sweep.
- `nonlinear_mlp/scripts/run_mnist_nldropout_sweep.py` — nonlinearity-dropout
  sweep.
- `nonlinear_mlp/scripts/run_cifar_cnn9_sweep.py` and
  `cifar100_cnn9_sweep.py` — CIFAR CNN sweeps.
- `nonlinear_mlp/scripts/run_layerwise_ablation.py` — layerwise ablation.
- `nonlinear_mlp/scripts/plot_results.py` — plots from recorded run outputs.
- `nonlinear_mlp/scripts/sweep_all_wandb.py` — optional W&B orchestration; see
  `W&B.md`.

## Reproducibility and interpretation

The experiment configuration defaults to seed 42. GPU kernels and data-loader
behavior can still introduce nondeterminism, so report software/hardware
versions and multiple seeded runs. Comparisons should separate accuracy,
calibration, parameter count, latency, and pruning effects. Do not interpret a
single dataset/run as evidence that mixed activations are universally better.

## Repository map

- `nonlinear_mlp/experiments/` — primary CLI experiment runner.
- `nonlinear_mlp/models/` and `nonlinear_mlp/layers/` — architectures and
  activation modules.
- `nonlinear_mlp/data/` — dataset loaders.
- `nonlinear_mlp/training/` — optimization and training utilities.
- `nonlinear_mlp/analysis/` and `nonlinear_mlp/pruning/` — diagnostics and
  pruning.
- `nonlinear_mlp/scripts/` — sweeps, ablations, and plotting.
