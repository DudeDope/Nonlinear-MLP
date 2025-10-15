# W&B Logging (Optional)

This repo can log every run to Weights & Biases and store all W&B files inside each run’s folder.

## Prerequisites
- Install: `pip install wandb`
- Set API key:
  - Locally: `export WANDB_API_KEY=YOUR_KEY`
  - Colab: 
    ```python
    import wandb
    wandb.login()  # or set os.environ["WANDB_API_KEY"]="YOUR_KEY"
    ```

## Enable W&B from CLI
- Minimal:
  ```bash
  python -m nonlinear_mlp.experiments.run_experiment \
    --dataset mnist --model mlp --approach fixed --linear_ratio 0.5 \
    --epochs 5 --run_name mnist_lin50 --wandb \
    --wandb_project nonlinear-mlp --wandb_entity YOUR_ENTITY
  ```
- Comma-separated tags and optional group:
  ```bash
  --wandb_group sweep_mnist --wandb_tags ratio,baseline
  ```

## Where logs are written
- JSON logs: `runs/<run_name>/history.json`, `runs/<run_name>/meta.json`
- Checkpoints: `runs/<run_name>/checkpoint_<epoch>.pt`
- W&B files: `runs/<run_name>/wandb/` (local copy of W&B run data)

## What gets logged to W&B
- Config snapshot (all hyperparams)
- Per-epoch metrics: `train/loss`, `train/acc`, `val/loss`, `val/acc`, `time/train_s`
- Gating stats (if approach=gating):
  - Scalars per layer: `gating/alpha_mean|alpha_median|.../layer_i`
  - Histograms per layer: `gating/alpha_hist_layer_i` (toggle via `logging.wandb_log_alpha_hist`)
- Summary after evaluation:
  - Latency and throughput
  - Param counts, approx linear FLOPs
  - Memory usage
  - Post-harden and post-prune accuracy (if applicable)

## Enable W&B via JSON config (alternative)
Add to the `logging` section:
```json
"logging": {
  "run_name": "cifar_head_50",
  "output_dir": "runs",
  "wandb_enabled": true,
  "wandb_project": "nonlinear-mlp",
  "wandb_entity": "YOUR_ENTITY",
  "wandb_group": "cifar_sweep",
  "wandb_tags": ["ratio:0.5", "fixed"],
  "wandb_mode": "online"
}
```
Then run:
```bash
python -m nonlinear_mlp.experiments.run_experiment --config path/to/config.json
```

Tip: Each run will create its own `runs/<run_name>/wandb` folder so your W&B logs are separated by run.
