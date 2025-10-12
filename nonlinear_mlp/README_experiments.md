# Nonlinearity Reduction Experiments

Implements experimental framework for exploring how many nonlinear neurons are *actually necessary* across tasks and architectures.

## Approaches Implemented

| Approach | Description | File(s) |
|----------|-------------|---------|
| 1 | Fixed % linear vs ReLU neurons | `layers/mixed.py` |
| 2 | Learned soft gating (alpha blend) | `layers/gated.py` |
| 3 | Post-training pruning (activation stats + heuristics) | `analysis/activation_stats.py`, `pruning/post_training.py` |
| 4 | Training from scratch w/ mixed activations | Combine approaches in config |
| 5 | Layer-wise ratio adaptation | Use `layerwise.schedule` in config |

## Quick Start (MNIST)

```bash
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --approach fixed --linear_ratio 0.5 --pattern structured --epochs 10 --run_name mnist_fixed_50
```

Patterns:
- `structured`: first N% linear
- `random`
- `alternating`

Ratios: test 0.0, 0.25, 0.5, 0.75, 1.0 (where 1.0 = all linear, 0.0 = all ReLU)

## Learned Gating

```bash
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --gating --epochs 15 --run_name mnist_gating
```

Adjust in `config.py` (or JSON):
- `gating.init_alpha`
- `gating.entropy_reg`
- `gating.l1_reg`
- `gating.hard_threshold`

After training, model gates are hardened (alpha -> {0,1}) and re-evaluated.

## CIFAR-10 with ResNet-18 Head

Modify only MLP head:

```bash
python -m nonlinear_mlp.experiments.run_experiment --dataset cifar10 --approach fixed --linear_ratio 0.5 --run_name cifar_head_50
```

Learned gating on head:

```bash
python -m nonlinear_mlp.experiments.run_experiment --dataset cifar10 --gating --run_name cifar_head_gating
```

## Tabular (Adult) (Requires Preprocessed CSV)

Place `data/adult.csv` with numeric features + `label` column:

```bash
python -m nonlinear_mlp.experiments.run_experiment --dataset tabular_adult --approach fixed --linear_ratio 0.5 --run_name adult_50
```

## Layer-wise Schedules

Create a JSON config:

```json
{
  "dataset": "mnist",
  "approach": "layerwise",
  "fixed": {
    "linear_ratio": 0.5,
    "pattern": "structured"
  },
  "layerwise": {
    "enabled": true,
    "schedule": {
      "0": 0.2,
      "1": 0.5,
      "2": 0.8
    }
  },
  "training": { "epochs": 12 }
}
```

Run:

```bash
python -m nonlinear_mlp.experiments.run_experiment --config path/to/config.json --run_name layerwise_test
```

## Post-Training Pruning (Approach 3)

Enable pruning:

```bash
python -m nonlinear_mlp.experiments.run_experiment --dataset mnist --approach fixed --linear_ratio 0.0 --pruning --epochs 10 --run_name mnist_prune_from_relu
```

Workflow:
1. Train standard model (all ReLU or mixed)
2. Collect activation stats on training subset
3. Decide linearizable or dead neurons/layers
4. (Optional) Fine-tune

Heuristics adjustable in `config.pruning`.

## Metrics Logged

- Training: loss, accuracy, time
- Validation: loss, accuracy
- Gating stats (alpha distribution per layer)
- Inference latency (mean, p50, p90)
- Memory usage (RSS)
- Param counts, approximate linear-layer FLOPs
- History stored as JSON

## Recommended MNIST Sweep

| Ratio | Command |
|-------|---------|
| 0% linear | `--linear_ratio 0.0` |
| 25% | `--linear_ratio 0.25` |
| 50% | `--linear_ratio 0.5` |
| 75% | `--linear_ratio 0.75` |
| 100% linear | `--linear_ratio 1.0` |

## Gating Regularization Guidelines

| Goal | Setting |
|------|---------|
| Encourage binary alpha | Increase `entropy_reg` (e.g. 0.005) |
| Reduce nonlinear usage | Increase `l1_reg` (e.g. 0.01) |
| Target ratio explicitly | Set `sparsity_target` + `sparsity_loss_weight` |

Inspect distributions in `history.json`.

## Next Steps (Not Yet Implemented Here)

- ImageNet: replace dataset loader, reuse ResNet head logic
- Transformer FFN modification: wrap feed-forward (GELU) sublayers with gating/mixed modules
- Energy measurement: integrate NVIDIA SMI polling
- Adversarial robustness: add FGSM / PGD evaluation script

## Interpreting Results

1. Plot accuracy vs. % linear neurons
2. Track gating alpha histograms over epochs
3. Compare latency vs. accuracy trade-off
4. Examine which layers tolerate more linearization (layer-wise schedule)
5. Correlate activation statistics with gating outcomes

## Citation / Related Work (Add Later)

Add references to pruning, MoE, MobileNetV2, GLU, etc.

---

Happy experimenting!