from typing import Any, Dict, Optional

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class WandbLogger:
    def __init__(self):
        self.run = None
        self.enabled = False

    def start(
        self,
        project: Optional[str],
        entity: Optional[str],
        group: Optional[str],
        mode: str = "disabled",
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if wandb is None or mode == "disabled" or not project:
            self.enabled = False
            return
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags or [],
            mode=mode,
            config=config or {},
            settings=wandb.Settings(start_method="thread"),
        )
        self.enabled = True

    def log_epoch(self, record: Dict[str, Any]):
        if not self.enabled:
            return
        gating = record.get("gating", [])
        log_dict = {
            "epoch": record.get("epoch"),
            "train/acc": record.get("train_acc"),
            "train/loss": record.get("train_loss"),
            "train/time_s": record.get("train_time_s"),
            "val/acc": record.get("val_acc"),
            "val/loss": record.get("val_loss"),
        }
        for item in gating:
            layer = item.get("layer")
            if layer is None:
                continue
            prefix = f"gating/layer{layer}"
            for k in ("alpha_mean", "alpha_median", "alpha_lt_0.1", "alpha_gt_0.9"):
                if k in item:
                    log_dict[f"{prefix}/{k}"] = item[k]
        wandb.log(log_dict)

    def log_event(self, name: str, data: Dict[str, Any]):
        if not self.enabled:
            return
        ev = {f"event/{name}/{k}": v for k, v in data.items()}
        wandb.log(ev)

    def log_meta(self, meta: Dict[str, Any]):
        if not self.enabled:
            return
        summ = {}
        latency = meta.get("latency") or {}
        for k, v in latency.items():
            summ[f"latency/{k}"] = v
        params = (meta.get("param_counts") or {})
        for k, v in params.items():
            summ[f"params/{k}"] = v
        mem = meta.get("memory_mb")
        if mem is not None:
            summ["system/memory_mb"] = mem
        flops = meta.get("approx_linear_flops")
        if flops is not None:
            summ["compute/approx_linear_flops"] = flops
        if "config" in meta:
            try:
                wandb.config.update(meta["config"], allow_val_change=True)
            except Exception:
                pass
        self.run.summary.update(summ)

    def finish(self):
        if not self.enabled:
            return
        self.run.finish()
        self.enabled = False