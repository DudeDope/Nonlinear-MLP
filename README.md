# Nonlinear-MLP Experimentation Guide

A complete, practical guide for running, extending, and interpreting experiments to answer the core research questions:

> 1. How much nonlinearity is actually necessary?  
> 2. Does this hold across architectures (MLPs, CNN heads, tabular MLPs, Transformer FFNs)?  
> 3. What is the performance vs efficiency trade-off?  
> 4. Which neurons should be linear vs nonlinear (random vs structured vs learned)?  

This document tells you:
- Exactly how to start each task.
- What to look for in logs and outputs.
- How to interpret gating and activation statistics.
- How to build evidence for each primary question.
- How to scale from MNIST → CIFAR-10 → Tabular → ImageNet → Transformers.

---

## 1. Repository Structure (Key Directories)

| Path | Purpose |
|------|---------|
| `nonlinear_mlp/config.py` | Experiment configuration dataclasses. |
| `nonlinear_mlp/layers/` | Mixed (fixed ratio) and gated activation modules. |
| `nonlinear_mlp/models/` | MLP, ResNet head wrappers, (optionally Transformers). |
| `nonlinear_mlp/data/` | Dataset loaders (MNIST, CIFAR-10, Adult, ImageNet). |
| `nonlinear_mlp/analysis/activation_stats.py` | Collect activation statistics (for pruning / interpretability). |
| `nonlinear_mlp/pruning/post_training.py` | Heuristics to linearize/prune layers post-training. |
| `nonlinear_mlp/utils/metrics.py` | Accuracy and latency measurement utilities. |
| `nonlinear_mlp/experiments/run_experiment.py` | Unified CLI runner for all models/datasets. |
| `nonlinear_mlp/scripts/` | Predefined sweeps (MNIST ratio sweep, layerwise ablation). |
| `nonlinear_mlp/notebooks/nonlinearity_analysis.ipynb` | Central analysis & visualization notebook. |
| `runs/<run_name>/` | Output directory per experiment (history, metadata, checkpoints). |

---

## 2. Installation & Environment

```bash
python -m venv .venv
source .venv/bin/activate            # (Linux/macOS)
# or .venv\Scripts\activate          # (Windows)

pip install -r requirements.txt

# Optional for Transformers experiments:
pip install transformers accelerate tokenizers
```

**GPU Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 3. Core Concepts

| Concept | Meaning | Why It Matters |
|---------|--------|----------------|
| Linear ratio | Fraction of hidden neurons that skip ReLU (identity) | Direct axis for "minimum nonlinearity" investigation |
| Approach 1 | Fixed ratio (structured / random / alternating) | Baseline control of nonlinearity |
| Approach 2 | Learned gating (alpha blend) | Lets network *discover* required nonlinearity |
| Approach 3 | Post-training pruning / linearization | Derive minimal nonlinear set after a standard train |
| Approach 4 | Train from scratch with mixed activations | End-to-end optimization under hybrid structure |
| Approach 5 | Layer-wise adaptation schedule | Tests if different layers need different nonlinear density |
| Activation stats | Fraction positive / negative + nonlinear contribution | Diagnostic for where ReLU actually changes outputs |
| Gating alphas | Learned per-layer (or per-neuron) nonlinearity weights | Interpret what the model deems necessary |
| Harden phase | Converts soft gating into discrete identity/ReLU behavior | Stabilizes final evaluation state |

---

## 4. Quick Start (Task 1 – MNIST MLP)

### 4.1 Baseline ReLU
```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset mnist --model mlp --approach fixed \
  --linear_ratio 0.0 --epochs 10 --run_name mnist_relu_baseline
```

### 4.2 Sweep Linear Ratios
Use included script (adjust epochs if needed):
```bash
python nonlinear_mlp/scripts/run_mnist_sweep.py
```
Generates runs like:
```
runs/
  mnist_fixed_0/
  mnist_fixed_25/
  mnist_fixed_50/
  ...
```

### 4.3 Gating Run
```bash
python -m nonlinear_mlp.experiments.run_experiment \
  --dataset mnist --model mlp --gating \
  --epochs 12 --run_name mnist_gating
```

### 4.4 Analyze
Open the notebook:
```
jupyter notebook nonlinear_mlp/notebooks/nonlinearity_analysis.ipynb
```
Look at:
- Accuracy vs linear ratio curve.
- Gating alpha evolution.
- Generalization gap (train vs val).
- Latency vs accuracy scatter.

---

## 5. Run Directory Artifacts

Each experiment creates:
```
runs/<run_name>/history.json
runs/<run_name>/meta.json
runs/<run_name>/checkpoint_<epoch>.pt (if enabled)
```

### 5.1 `history.json`
Array of epoch records:
```json
{
  "epoch": 10,
  "train_loss": ...,
  "train_acc": ...,
  "val_loss": ...,
  "val_acc": ...,
  "gating": [
     {"layer": 0, "alpha_mean": 0.73, "alpha_lt_0.1": 0.05, ...},
     ...
  ]
}
```
Special entries:
- `"epoch": "hardened"` – after gating thresholding.
- `"epoch": "post_prune"` – after pruning fine-tune (if used).

### 5.2 `meta.json`
Contains:
```json
{
  "param_counts": {"total_params": ..., "trainable_params": ...},
  "approx_linear_flops": ...,
  "latency": {
     "mean_latency_s": ...,
     "p50_latency_s": ...,
     "samples_per_second": ...,
     ...
  },
  "memory_mb": ...,
  "config": { ... full config blob ... }
}
```

---

## 6. How to Answer the Primary Questions

### Q1: “How much nonlinearity is actually necessary?”
Steps:
1. Generate accuracy vs linear ratio plot (MNIST first).
2. Identify smallest ratio with < target accuracy drop (e.g. <1–2% absolute).
3. For gating runs, compute effective linear ratio: `1 - mean_alpha` and evaluate accuracy.
4. Report threshold (e.g., “50% linear retains 98% of baseline accuracy”).

Artifacts:
- Notebook: “Accuracy vs Linear Neuron Ratio”
- `summary_df` (`analysis_summary.csv` after saving from notebook)
- Gating alpha histogram / distribution table

Metrics to extract:
- `val_acc` at each ratio
- `linear_ratio_est` (fixed or inferred)
- Possibly plot “delta accuracy vs ratio” for clarity.

### Q2: “Does this work across architectures?”
Do same process for:
- MNIST MLP → Foundational
- CIFAR ResNet-18 head → Vision mid-scale
- Tabular Adult → Non-vision
- (Later) ImageNet ResNet-50 head → Large-scale
- (Later) Transformers (BERT FFN gating) → NLP

Compare curves:
- Normalize performance drop at 25%, 50%, 75% linear vs baseline.
- Summarize in a table:

| Architecture | 25% Linear Drop | 50% Linear Drop | 75% Linear Drop | Gated Effective Ratio |
|--------------|-----------------|-----------------|-----------------|-----------------------|

### Q3: “Performance vs Efficiency trade-off?”
Gather:
- `val_acc` (accuracy)
- `latency.mean_latency_s`
- `samples_per_second`
- Optionally param counts (same) & theoretical `approx_linear_flops`

Plot:
- Latency vs Accuracy (Pareto frontier)
- Accuracy vs Mean Latency (annotated with ratio)
- Efficiency metric: `accuracy / latency` ranking

Interpretation:
- Identify sweet spot where marginal accuracy loss buys disproportionate latency gain.

### Q4: “Which neurons should be linear vs nonlinear?”
Approaches:
1. Fixed structured vs random vs alternating → Compare accuracy variance.
2. Gating alpha distribution:
   - Examine `alpha_lt_0.1` fraction per layer over epochs.
   - Identify if early or late layers trend toward linear.
3. Activation stats (Approach 3):
   - Positive fraction close to 1.0 means ReLU ineffective → linear candidate.
   - “Nonlinear contribution” score low (<0.05) → safe linearization.
4. Pruning decisions:
   - Count layers flagged for linearization in decisions object.
   - Evaluate effect on accuracy before & after fine-tuning.

Report:
- Layer-by-layer table with columns: `layer_idx | pos_frac | nonlinear_score | gating_alpha_mean | linearized?`

---

## 7. Secondary Question Workflows

| Question | Evidence / Procedure |
|----------|---------------------|
| Does optimal ratio change across layers? | Use layerwise schedule + gating alpha stats. Plot alpha_mean vs layer. |
| Dependence on dataset complexity? | Compare threshold ratio across MNIST vs CIFAR vs ImageNet subset vs Tabular. |
| Generalization insight | Compare generalization gap vs ratio (does reducing nonlinearity regularize?). |
| Training dynamics | Plot convergence speed (# epochs to reach X% baseline). |
| Robustness (future) | Add adversarial eval (FGSM / PGD) for different ratios. |

---

## 8. Experimental Task Guide

### Task 1 (MNIST)
Goal: Rapid sensitivity check.
- Run ratio sweep + gating.
- Target: 50% linear ≤ 1–3% accuracy loss.

### Task 2 (CIFAR-10 ResNet-18 Head)
Goal: Confirm idea in harder visuals.
- Start with baseline (0% linear).
- Try 25/50/75%.
- Add gating and pruning run.
- Compare inference latency (note: head-only speedup modest).

### Task 3 (Tabular)
Goal: Generalization beyond vision.
- Use best approach (gating or fixed 50%).
- Compare ratio threshold shift vs vision.

### Task 4 (ImageNet Subset/full)
Goal: Scale evidence.
- Use gating on ResNet-50 head.
- Add top-5 accuracy metric (modify evaluate).
- Evaluate cost-benefit at 25–50% linear in head.

### Task 5 (Transformers)
Goal: NLP domain validation.
- Patch FFN intermediate with gating/mixed module.
- Fine-tune on a small GLUE task (e.g., SST-2).
- Track alpha collapse per layer (are higher layers more linear?).

### Task 6 (Ablations)
Goal: Mechanistic understanding.
- Layerwise ratio schedules: low→high, high→low, uniform.
- Compare gating vs fixed with same effective ratio.
- Try pruning after gating vs pruning after pure ReLU training.

---

## 9. Interpreting Gating (Approach 2)

| Metric | Insight |
|--------|---------|
| `alpha_mean` ↓ | Layer becoming more linear overall. |
| `alpha_lt_0.1` ↑ | Large chunk of neurons act linear. |
| `alpha_gt_0.9` stable | Core nonlinear subset persistent. |
| Shift early vs late | Structural / representational economy differences. |

**Decision:** If a layer’s `alpha_mean < 0.3` across final epochs → candidate for forced linearization in future architecture simplification.

---

## 10. Activation Stats (Approach 3)

Collected statistics per `Linear`:
- `positive_frac`: If >0.95, ReLU rarely clips → identity safe.
- `negative_frac`: If >0.95, neuron outputs near zero → “dead.”
- `nonlinear_score`: Mean fraction of magnitude below zero (ReLU-suppressed region). Lower means minimal nonlinear action.

Heuristic thresholds (tune with pilot):
| Feature | Threshold | Action |
|---------|-----------|--------|
| positive_frac > 0.95 & nonlinear_score < 0.05 | Mark linear |
| negative_frac > 0.95 | Consider pruning or ignoring |
| nonlinear_score > 0.2 | Likely influential nonlinearity |

---

## 11. Layerwise Schedule Exploration

Use JSON config:
```json
{
  "dataset": "mnist",
  "approach": "layerwise",
  "fixed": { "linear_ratio": 0.5, "pattern": "structured" },
  "layerwise": {
    "enabled": true,
    "schedule": { "0": 0.2, "1": 0.5, "2": 0.8 }
  },
  "training": { "epochs": 12 },
  "logging": { "run_name": "mnist_layerwise_gradient" }
}
```
Run:
```bash
python -m nonlinear_mlp.experiments.run_experiment --config config.json
```

Plot (in notebook) accuracy vs (average ratio) and inspect per-layer gating alphas (if gating approach used).

---

## 12. Metrics Reference

| Metric | File Source | Meaning | Use |
|--------|-------------|---------|-----|
| `train_acc` / `val_acc` | `history.json` | Performance | Primary axis |
| `train_loss` / `val_loss` | `history.json` | Convergence / overfit gap | Generalization analysis |
| `gating.alpha_mean` | `history.json` | Nonlinearity density per layer | Approach 2 insights |
| `latency.mean_latency_s` | `meta.json` | Wall-clock per batch | Efficiency |
| `samples_per_second` | `meta.json` | Throughput | Practical speed |
| `approx_linear_flops` | `meta.json` | Theoretical linear-layer FLOPs | Rough cost baseline |
| `memory_mb` | `meta.json` | Host memory RSS | Resource footprint |
| `param_counts.trainable_params` | `meta.json` | Model complexity | Cross-run normalization |

---

## 13. Building Your Final Report

Structure suggestion:

### 1. Executive Summary
Short narrative: “We found that ~50% linear neurons retain ≥98% MNIST accuracy, ~50% yields <2% drop on CIFAR head…”

### 2. Methods
Outline Approaches 1–5 with diagrams (optional).

### 3. Results
- Accuracy vs Linear Ratio (plots per dataset)
- Latency vs Accuracy (Pareto)
- Gating Alpha Distributions (heatmaps)
- Layerwise sensitivity

### 4. Cross-Domain Consistency
Table summarizing thresholds for MNIST / CIFAR / Tabular / ImageNet / BERT.

### 5. Interpretation
- Linear neurons handle pass-through & amplitude features.
- Nonlinear subset concentrated in earlier or middle layers (if observed).

### 6. Limitations
- No real compute reduction yet (no sparse kernel).
- Pruning heuristic coarse (layer-level).
- Transformer gating scalar vs per-neuron (if not extended).

### 7. Future Work
- Per-neuron structural pruning.
- Energy and robustness evaluation.
- Adaptive ratio scheduling during training.

---

## 14. Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| Accuracy very low across all ratios | Data loader normalization mismatch | Check mean/std |
| Gating alpha stays ~initial | Regularization too strong or lr too low | Reduce entropy/L1 or raise lr |
| No latency difference | Head-only modification or GPU overhead dominates | Increase batch size or move linearization deeper |
| Pruning causes large accuracy crash | Thresholds too aggressive | Relax positive_frac / nonlinear_score cutoffs |
| Notebook errors on epoch | Post-prune/harden epochs are strings | Coerce or filter non-integer epochs |

---

## 15. Extensions (Optional)

| Extension | Description |
|-----------|-------------|
| Per-neuron pruning | Rebuild layers excluding “dead” units. |
| Adaptive gating annealing | Lower temperature over epochs. |
| Robustness tests | FGSM / PGD across ratios. |
| Energy measurement | Log GPU power (poll `nvidia-smi`). |
| Alpha clustering | Analyze if “important nonlinearity” forms functional groups. |
| Mixed activation in attention projections | Extend beyond FFN to MHA linear projections. |

---

## 16. Suggested Execution Order (Time-Efficient Path)

1. MNIST sweep (establish curve shape).
2. CIFAR-10 head fixed + gating runs.
3. Add pruning pass to CIFAR results.
4. Tabular (Adult) confirm portability.
5. ImageNet subset (e.g., 50–100 classes) for scaling trend.
6. Transformer FFN gating (SST-2).
7. Layerwise ablations (only after gating insights).
8. Compile cross-domain summary.

---

## 17. Time Budget Guidance

| Phase | Est. Time (GPU modest) |
|-------|------------------------|
| MNIST full sweep | 30–60 min |
| CIFAR head 4–6 runs | 4–6 hrs |
| Gating CIFAR (longer) | 4–8 hrs |
| Tabular trio | 1–2 hrs |
| ImageNet subset (10–20 epochs) | 1–2 days |
| Transformer small GLUE | 4–8 hrs |
| Ablations + analysis | Parallel / incremental |

---

## 18. Minimal Commands Cheat Sheet

```bash
# Baseline
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --approach fixed --linear_ratio 0.0 --run_name mnist_relu

# 50% linear
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --approach fixed --linear_ratio 0.5 --run_name mnist_lin50

# Gating
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --model mlp --gating --run_name mnist_gating

# CIFAR-10 head 50%
python -m nonlinear_mlp.experiments.run_experiment --dataset cifar10 --model resnet18_head --approach fixed --linear_ratio 0.5 --epochs 60 --run_name cifar_head_50

# CIFAR-10 gating
python -m nonlinear_mlp.experiments.run_experiment --dataset cifar10 --model resnet18_head --gating --epochs 60 --run_name cifar_head_gating

# CIFAR-10 pruning (start all ReLU)
python -m nonlinear_mlp.experiments.run_experiment --dataset cifar10 --model resnet18_head --approach fixed --linear_ratio 0.0 --pruning --epochs 60 --run_name cifar_head_prune

# Layerwise schedule example
python -m nonlinear_mlp.experiments.run_experiment --config configs/mnist_layerwise.json --run_name mnist_layerwise_demo
```

---

## 19. Final Checklist Before Claiming Findings

| Item | Verified? |
|------|-----------|
| Accuracy vs ratio curves stable across random seeds |
| Gating alpha distributions converge (not random noise) |
| Latency gains measured with consistent batch size |
| Activation stats collected on representative data subset |
| Layerwise behavior consistent across runs (not artifact) |
| Report includes cross-domain comparative table |
| Hardening step doesn’t cause unexpected accuracy collapse |
| Pruning changes documented with before/after metrics |

---

## 20. Contact / Next Steps

If you need:
- A GLUE fine-tuning script,
- Per-neuron pruning implementation,
- Energy measurement integration,
- Automated summary report generator,

Request these incrementally to maintain clarity of experimental control.

---

### TL;DR Flow

1. Run MNIST sweep → see elbow.  
2. Run CIFAR head fixed + gating → confirm elbow similar.  
3. Run pruning → evaluate post-hoc linearization viability.  
4. Expand to tabular + ImageNet head → cross-domain consistency.  
5. Patch Transformer FFN → NLP confirmation.  
6. Layerwise schedules + gating stats → mechanistic story.  
7. Assemble report with accuracy/latency/gating/activation evidence.  

Good experimenting!
