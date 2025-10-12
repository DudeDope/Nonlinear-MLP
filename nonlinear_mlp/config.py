from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Callable
import json
import math

ActivationPattern = Literal["structured", "random", "alternating"]

@dataclass
class FixedRatioConfig:
    linear_ratio: float = 0.5
    pattern: ActivationPattern = "structured"  # structured | random | alternating
    per_layer: Optional[List[float]] = None  # overrides linear_ratio if provided

@dataclass
class GatingConfig:
    enabled: bool = False
    init_alpha: float = 0.75          # initial alpha (prob of nonlinear)
    temperature: float = 1.0
    entropy_reg: float = 0.001        # encourages alpha -> {0,1}
    l1_reg: float = 0.0               # pushes alpha toward 0 (less nonlinear)
    sparsity_target: Optional[float] = None
    sparsity_loss_weight: float = 0.01
    hard_threshold: float = 0.1       # after training alpha<th -> linear
    clamp: bool = True

@dataclass
class PruningConfig:
    enabled: bool = False
    activation_positive_ratio_threshold: float = 0.95
    dead_negative_ratio_threshold: float = 0.95
    nonlinear_contribution_threshold: float = 0.05
    impact_eval_batches: int = 5
    fine_tune_epochs: int = 3

@dataclass
class LayerwiseConfig:
    enabled: bool = False
    schedule: Optional[Dict[int, float]] = None  # layer_idx -> linear_ratio

@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    early_stop_patience: Optional[int] = None

@dataclass
class LoggingConfig:
    log_interval: int = 100
    save_checkpoints: bool = True
    output_dir: str = "runs"
    run_name: str = "exp"

@dataclass
class ExperimentConfig:
    dataset: str = "mnist"  # mnist | cifar10 | tabular_adult | tabular_credit | tabular_wine
    model: str = "mlp"      # mlp | resnet18_head | tabular_mlp
    approach: Literal["fixed", "gating", "pruning", "mixed", "layerwise"] = "fixed"
    fixed: FixedRatioConfig = field(default_factory=FixedRatioConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    layerwise: LayerwiseConfig = field(default_factory=LayerwiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10
    input_dim: Optional[int] = None  # for tabular
    resume: Optional[str] = None

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)