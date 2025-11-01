# Nonlinear-MLP Experimentation Guide (Updated)

A complete, practical guide for running, extending, and interpreting experiments to answer the core research questions:

> 1) How much nonlinearity is actually necessary?  
> 2) Does this hold across architectures (MLPs, CNNs, ResNet heads, tabular MLPs)?  
> 3) What is the performance vs efficiency trade-off?  
> 4) Which neurons should be linear vs nonlinear (random vs structured vs learned)?  

This document tells you:
- Exactly how to start each task and sweeps (MNIST, CIFAR-10/100, ImageNet).
- What to look for in logs and outputs beyond W&B charts.
- How to interpret gating and activation statistics.
- How to run the analysis pipeline over runs/ and produce ablations.
- How to add “control” models that structurally reduce neurons to match effective nonlinearity.

---

## 1) Repository Structure (Key Additions)

| Path | Purpose |
|------|---------|
| `nonlinear_mlp/config.py` | Experiment configuration dataclasses. |
| `nonlinear_mlp/layers/mixed.py` | Fixed per-neuron Identity vs ReLU (MLP blocks). |
| `nonlinear_mlp/layers/gated.py` | Gated per-neuron α between Identity/ReLU (MLP blocks). |
| `nonlinear_mlp/layers/nl_dropout_layer.py` | Deterministic per-neuron zeroing (new “nl_dropout”). |
| `nonlinear_mlp/layers/mixed2d.py` | Channel-wise Identity vs ReLU for Conv2d (CNN control). |
| `nonlinear_mlp/models/mlp.py` | MLP with approaches: fixed, gating, nl_dropout. |
| `nonlinear_mlp/models/mlp_control.py` | Control MLP that structurally reduces layer widths (new). |
| `nonlinear_mlp/models/cnn_plain.py` | 9-layer plain CNN with channel-wise nonlinearity control (new). |
| `nonlinear_mlp/models/cifar_head.py` | ResNet-18 + MLP head wrapper. |
| `nonlinear_mlp/models/resnet50_head.py` | ResNet-50 + MLP head wrapper (optional if present). |
| `nonlinear_mlp/data/mnist.py` | MNIST loaders. |
| `nonlinear_mlp/data/cifar10.py` | CIFAR-10 loaders. |
| `nonlinear_mlp/data/cifar100.py` | CIFAR-100 loaders (new). |
| `nonlinear_mlp/data/imagenet.py` | ImageNet loaders via ImageFolder (new). |
| `nonlinear_mlp/experiments/run_experiment.py` | Unified CLI runner, per-step W&B logging, pruning hooks. |
| `nonlinear_mlp/analysis/activation_stats.py` | Robust activation tracker (Linear and optional Conv). |
| `nonlinear_mlp/analysis/collect_runs.py` | Consolidate runs/* into a single CSV. |
| `nonlinear_mlp/analysis/ablation.py` | Ablation plots (accuracy vs nonlinearity, Pareto, gaps). |
| `nonlinear_mlp/analysis/evaluate_run.py` | Re-evaluate checkpoints (ACC, NLL, ECE; supports MLP flattening). |
| `nonlinear_mlp/analysis/merge_analysis_tables.py` | Merge summary.csv with extra_eval.csv. |
| `nonlinear_mlp/pruning/post_training.py` | Heuristics to linearize/prune layers post-training. |
| `nonlinear_mlp/utils/metrics.py` | Accuracy, latency, throughput; safe MLP flattening in latency. |
| `nonlinear_mlp/scripts/run_mnist_sweep.py` | MNIST fixed ratio sweep. |
| `nonlinear_mlp/scripts/run_mnist_nldropout.py` | MNIST deterministic zeroing sweep (new). |
| `nonlinear_mlp/scripts/run_mnist_mlp_control_sweep.py` | Control MLP sweep (structural width) (new). |
| `nonlinear_mlp/scripts/run_cifar_cnn9_sweep.py` | CIFAR-10 9-layer CNN channel-wise sweep (new). |
| `nonlinear_mlp/scripts/run_cifar100_sweep.py` | CIFAR-100 ResNet18-head sweep (new). |
| `nonlinear_mlp/scripts/run_imagenet_head_baseline.py` | ImageNet head baseline runner (new). |
| `nonlinear_mlp/scripts/evaluate_all_runs.py` | Batch evaluate every run (ACC/NLL/ECE) (new). |
| `runs/<run_name>/` | Output per experiment: history.json, meta.json, checkpoints, extra_eval.json. |

---

## 2) Installation & Environment

```bash
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate             # Windows

pip install -r requirements.txt
# torchvision/torchaudio should match your CUDA/PyTorch install

# Optional (Transformers later):
# pip install transformers accelerate tokenizers
```

GPU check:
```bash
python -c "import torch; print(torch.cuda.is_available()); import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

---

## 3) Core Concepts (Updated)

| Concept | Meaning | Why It Matters |
|---------|--------|----------------|
| Linear ratio | Fraction of hidden units that bypass ReLU (Identity) | Primary axis for “minimum nonlinearity”. |
| fixed (MLP) | Per-neuron deterministic mix via MixedActivationLayer | Baseline nonlinearity control. |
| gating (MLP) | Learn α per neuron (Identity↔ReLU), harden later | Lets model discover needed nonlinearity. |
| nl_dropout (MLP) | Deterministic per-neuron zeroing of N% (new) | Control variant: remove neurons by output=0. |
| control MLP | Structurally reduce layer width to match nonlinear fraction (new) | Parameter-matched control against “fixed” runs. |
| CNN channel-wise | Per-channel Identity vs ReLU after each Conv (new) | Neuron-like control in CNNs. |
| Activation stats | positive/negative fractions + nonlinear_score | Diagnostics for where ReLU actually changes outputs. |
| Pruning | Decide post-hoc linearization using activation stats | Validate necessity of nonlinearity ex-post. |
| Per-step W&B | Log loss/acc per batch, epoch aggregates | Inspect convergence and stability. |

Datasets supported: MNIST, CIFAR-10, CIFAR-100 (new), ImageNet (new), Adult tabular.

---

## 4) Quick Starts

### 4.1 MNIST MLP (fixed)

Baseline ReLU:
```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset mnist --model mlp --approach fixed \
  --linear_ratio 0.0 --epochs 20 --run_name mnist_fixed_0 --wandb --wandb_project nonlinear-mlp
```

Sweep (0,25,50,75,100):
```bash
python nonlinear_mlp/scripts/run_mnist_sweep.py
```

### 4.2 MNIST MLP (gating)

```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset mnist --model mlp --gating \
  --epochs 20 --run_name mnist_gating --wandb --wandb_project nonlinear-mlp
```

### 4.3 MNIST MLP (deterministic zeroing, nl_dropout)

- Exactly N% of neurons per hidden layer output 0 forever; rest use ReLU.
- pattern=structured zeros last K per layer; pattern=random uses a fixed seed.

Single run (50% zeroed):
```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset mnist --model mlp --approach nl_dropout \
  --linear_ratio 0.5 --pattern structured \
  --epochs 20 --run_name mnist_mlp_nldropout_50 --wandb --wandb_project nonlinear-mlp
```

Sweep (0,25,50,75):
```bash
python -m nonlinear_mlp.scripts.run_mnist_nldropout --wandb_project nonlinear-mlp --mode online
```

### 4.4 MNIST MLP (control, structural width reduction)

- Keeps K% of hidden neurons structurally (layer widths reduced).
- Map desired K% to `linear_ratio = 1 - K` to reuse config plumbing.

Example (keep 50%):
```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset mnist --model mlp_control \
  --linear_ratio 0.5 --epochs 20 \
  --run_name mnist_mlp_control_nonlin_50 --wandb --wandb_project nonlinear-mlp
```

Sweep:
```bash
python -m nonlinear_mlp.scripts.run_mnist_mlp_control_sweep --wandb_project nonlinear-mlp --mode online
```

### 4.5 CIFAR-10 (plain CNN-9, channel-wise nonlinearity)

- 9 conv layers with ReLU-per-channel control (N% linear per conv).
- Defaults: 200 epochs, SGD 0.1, mom 0.9, wd 5e-4, bs 128.

Example (50% linear channels):
```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset cifar10 --model cnn9_plain --approach fixed \
  --linear_ratio 0.5 --pattern structured \
  --epochs 200 --run_name cifar_cnn9_fixed_50 --wandb --wandb_project nonlinear-mlp
```

Sweep:
```bash
python -m nonlinear_mlp.scripts.run_cifar_cnn9_sweep --wandb_project nonlinear-mlp --mode online
```

### 4.6 CIFAR-10 (ResNet18 head)

```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset cifar10 --model resnet18_head --approach fixed \
  --linear_ratio 0.5 --epochs 60 --run_name cifar_head_fixed_50 --wandb --wandb_project nonlinear-mlp
```

### 4.7 CIFAR-100 (ResNet18 head)

```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset cifar100 --model resnet18_head --approach fixed \
  --linear_ratio 0.0 --epochs 100 --run_name cifar100_r18_baseline --wandb --wandb_project nonlinear-mlp
```

Sweep:
```bash
python -m nonlinear_mlp.scripts.run_cifar100_sweep --wandb_project nonlinear-mlp --mode online
```

### 4.8 ImageNet (ResNet head baseline)

- Expects `data/imagenet/{train,val}/class/*`.
- Defaults: 224px crops; adjust root via a config if needed.

```bash
python -m nonlinear_mlp.scripts.run_imagenet_head_baseline \
  --model resnet18_head --epochs 90 --wandb_project nonlinear-mlp --mode online
```

---

## 5) W&B Logging (Per-step + Per-epoch)

- runner (`experiments/run_experiment.py`) logs:
  - Per-step: loss, acc, reg, batch size, lr.
  - Per-epoch: train/val metrics, timings.
  - Summary: params, throughput/latency, approx flops, memory.
- Enable with `--wandb --wandb_project <name>` (+ optional entity, group, tags).

---

## 6) Run Artifacts

Each run writes:
```
runs/<run_name>/history.json
runs/<run_name>/meta.json
runs/<run_name>/checkpoint_<epoch>.pt (if save_checkpoints)
runs/<run_name>/extra_eval.json (after evaluation step)
```

### history.json (epoch array)
```json
{
  "epoch": 10,
  "train_loss": ...,
  "train_acc": ...,
  "val_loss": ...,
  "val_acc": ...,
  "gating": [
    {"layer": 0, "alpha_mean": 0.73, "alpha_lt_0.1": 0.05, ...}, ...
  ]
}
```
Special rows:
- `"epoch": "hardened"` after gating harden.
- `"epoch": "post_prune"` after pruning fine-tune.

### meta.json
```json
{
  "param_counts": {"total_params": ..., "trainable_params": ...},
  "approx_linear_flops": ...,
  "latency": {
    "mean_latency_s": ...,
    "p50_latency_s": ...,
    "samples_per_second": ...
  },
  "memory_mb": ...,
  "config": { ...full config blob... }
}
```

---

## 7) Analysis Pipeline (from runs/)

1) Collect all runs into one CSV:
```bash
python -m nonlinear_mlp.analysis.collect_runs --runs_dir runs
# -> runs/_analysis/summary.csv
```

2) Evaluate all checkpoints for ACC/NLL/ECE (clean & noisy):
```bash
python -m nonlinear_mlp.scripts.evaluate_all_runs --runs_dir runs --noise_std 0.1 --force
# -> runs/<run>/extra_eval.json and runs/_analysis/extra_eval.csv
```
Notes:
- Evaluator auto-flattens images for MLPs.
- Robust to dict configs via namespace wrapping.

3) Merge tables:
```bash
python -m nonlinear_mlp.analysis.merge_analysis_tables --runs_dir runs
# -> runs/_analysis/summary_merged.csv
```

4) Plot ablations:
```bash
python -m nonlinear_mlp.analysis.ablation --runs_dir runs
# -> runs/_analysis/figures/*.png
```

Figures produced:
- `acc_vs_nonlin.png` — accuracy vs effective nonlinearity (all models).
- `pareto_acc_latency.png` — accuracy vs throughput/latency.
- `train_val_gap.png` — generalization gap vs nonlinearity.
- `gating_soft_vs_hardened.png` — soft vs hardened (if present).

---

## 8) Activation Statistics (Improved)

`analysis/activation_stats.py`:
- Hooks `nn.Linear` as `linear_0, linear_1, ...` (stable naming).
- Optional `include_conv=True` to hook `nn.Conv2d` (`conv_0..`).
- Metrics:
  - `positive_frac` = fraction z>0 (low clipping → identity candidate).
  - `negative_frac` = fraction z<0 (dead-ish).
  - `nonlinear_score` = E[|min(0,z)| / (|z| + eps)] (how much ReLU truncates).
- Safe image flattening for MLPs during stats pass.

Heuristic thresholds (tune empirically):
| Feature | Threshold | Action |
|---------|-----------|--------|
| positive_frac > 0.95 & nonlinear_score < 0.05 | Mark linear |
| negative_frac > 0.95 | Consider pruning neuron |
| nonlinear_score > 0.2 | Likely influential nonlinearity |

---

## 9) Models & Approaches (What they do)

- MLP (fixed): per-neuron identity vs ReLU via MixedActivationLayer.
- MLP (gating): per-neuron learnable α; supports `harden_gates(threshold)`.
- MLP (nl_dropout): deterministically zero N% per layer outputs, keep ReLU for the rest.
- MLP (control): shrink widths to match desired nonlinearity fraction structurally.
- CNN-9 (channel-wise): per-channel identity vs ReLU masks after each conv.
- ResNet heads: standard backbones + MLP head with nonlinearity in head.

---

## 10) Sweeps (Scripts Catalog)

- MNIST fixed: `nonlinear_mlp/scripts/run_mnist_sweep.py`
- MNIST nl_dropout: `nonlinear_mlp/scripts/run_mnist_nldropout.py`
- MNIST control (structural): `nonlinear_mlp/scripts/run_mnist_mlp_control_sweep.py`
- CIFAR-10 CNN-9 channel-wise: `nonlinear_mlp/scripts/run_cifar_cnn9_sweep.py`
- CIFAR-100 ResNet18 head: `nonlinear_mlp/scripts/run_cifar100_sweep.py`
- ImageNet head baseline: `nonlinear_mlp/scripts/run_imagenet_head_baseline.py`
- Batch evaluate all runs: `nonlinear_mlp/scripts/evaluate_all_runs.py`

All scripts are idempotent: they skip when `runs/<run_name>/meta.json` exists (toggle with `--no_skip_existing` where supported).

---

## 11) Interpreting Gating

| Metric | Insight |
|--------|---------|
| `alpha_mean` ↓ | Layer trending more linear. |
| `alpha_lt_0.1` ↑ | Large fraction near identity behavior. |
| `alpha_gt_0.9` stable | Persistent nonlinear core. |
| Layerwise shifts | Early vs late layer nonlinearity demand. |

Rule of thumb: if final `alpha_mean < 0.3` consistently → candidate for forced linearization or structural reduction.

---

## 12) Performance vs Efficiency (What to plot)

- Accuracy vs effective nonlinearity fraction (all models).
- Pareto: Accuracy vs throughput (or inverse latency).
- Train−Val gap vs nonlinearity (implicit regularization).
- Gating soft vs hardened (stability of discrete assignment).

Use:
```bash
python -m nonlinear_mlp.analysis.collect_runs --runs_dir runs
python -m nonlinear_mlp.analysis.ablation --runs_dir runs
```

---

## 13) Dataset Notes

- MNIST: flattens to 784 for MLPs (auto-handled in training & eval).
- CIFAR-10/100: std aug (crop+flip) and normalization; channel-wise CNN experiment for CIFAR-10.
- ImageNet: folder layout `data/imagenet/train|val`; std 224 crops and normalization.

---

## 14) Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| AttributeError on cfg.training.seed | Nested config stored as dict | Runner/evaluator coerce to namespaces; update files already included. |
| mat1×mat2 shape error on MNIST eval | MLP received 4D tensor | Evaluator now flattens for MLP; ensure updated `evaluate_run.py`. |
| No W&B logs | Missing flags or API key | Use `--wandb --wandb_project ...` and `wandb login` or set `WANDB_API_KEY`. |
| ImageNet path error | Wrong root path | Use default `data/imagenet` structure or provide config with `data.imagenet_root`. |
| CNN-9 poor accuracy at high linear ratios | Too few nonlinear channels | Try lower linear_ratio or selective layer schedules. |

---

## 15) Estimated GPU Times (RTX 3090, rough)

| Task | Setting | Est. Time per Run |
|------|---------|--------------------|
| MNIST MLP (fixed/nl_dropout/control) | 20 epochs, bs 128 | 0.3–1.5 min |
| CIFAR-10 ResNet18 head | 60 epochs, bs 128 | 45–90 min |
| CIFAR-10 CNN-9 channel-wise | 200 epochs, bs 128 | 30–60 min |
| CIFAR-100 ResNet18 head | 100 epochs, bs 128 | 2–3 hrs |
| ImageNet ResNet18 head | 90 epochs, 224px, bs 256 | 18–36 hrs |
| Analysis (collect+ablation) | N/A | < 1 min for ~100 runs |
| Evaluate all runs (ACC/NLL/ECE) | noise_std=0.1 | ~1–3 s per run |

Notes:
- Times vary by aug, IO, cudnn bench, precision (AMP), and whether backbones are frozen.
- For faster sanity checks: reduce epochs, use subsets, or increase batch size.

---

## 16) Minimal Commands Cheat Sheet

```bash
# Baseline MNIST ReLU
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --approach fixed --linear_ratio 0.0 --epochs 20 --run_name mnist_fixed_0

# MNIST 50% linear (fixed)
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --approach fixed --linear_ratio 0.5 --epochs 20 --run_name mnist_fixed_50

# MNIST gating
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --gating --epochs 20 --run_name mnist_gating

# MNIST deterministic zeroing (50%)
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --approach nl_dropout --linear_ratio 0.5 --epochs 20 --run_name mnist_mlp_nldropout_50

# MNIST control model (keep 50% structurally)
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp_control --linear_ratio 0.5 --epochs 20 --run_name mnist_mlp_control_nonlin_50

# CIFAR-10 CNN-9 channel-wise (50% linear channels)
python -m nonlinear_mlp.experiments.run_experiment --dataset cifar10 --model cnn9_plain --approach fixed --linear_ratio 0.5 --epochs 200 --run_name cifar_cnn9_fixed_50

# CIFAR-10 ResNet18 head 50%
python -m nonlinear_mlp.experiments.run_experiment --dataset cifar10 --model resnet18_head --approach fixed --linear_ratio 0.5 --epochs 60 --run_name cifar_head_fixed_50

# CIFAR-100 ResNet18 head baseline
python -m nonlinear_mlp.experiments.run_experiment --dataset cifar100 --model resnet18_head --approach fixed --linear_ratio 0.0 --epochs 100 --run_name cifar100_r18_baseline

# ImageNet ResNet18 head baseline
python -m nonlinear_mlp.scripts.run_imagenet_head_baseline --model resnet18_head --epochs 90

# Collect + Ablation
python -m nonlinear_mlp.analysis.collect_runs --runs_dir runs
python -m nonlinear_mlp.analysis.ablation --runs_dir runs

# Evaluate all + merge tables
python -m nonlinear_mlp.scripts.evaluate_all_runs --runs_dir runs --noise_std 0.1 --force
python -m nonlinear_mlp.analysis.merge_analysis_tables --runs_dir runs
```

---

## 17) Final Checklist (Before Claiming Findings)

| Item | Verified? |
|------|-----------|
| Accuracy vs ratio curves stable across random seeds |
| Gating alpha distributions converge (not noise) |
| Latency/throughput measured with consistent batch sizes |
| Activation stats collected on representative subset |
| Layerwise behavior consistent (not artifact) |
| Cross-domain comparative table prepared (MNIST, CIFAR-10/100, ImageNet head) |
| Harden step doesn’t collapse accuracy |
| Control model comparisons included (parameter-matched) |
| Analysis figures saved to `runs/_analysis/figures/` |

---

## 18) Next Steps / Extensions

- Per-neuron pruning (structural) for MLP and per-channel for CNNs, to hard-match activation stats.
- Energy logging via `nvidia-smi` polling for efficiency claims.
- Robustness (FGSM/PGD) vs nonlinearity curves.
- Transformer FFN gating experiments (SST-2), reusing gating/mixed modules.

Happy experimenting!
