import torch
import copy

def decide_linearization(stats: dict,
                         pos_thresh=0.95,
                         neg_thresh=0.95,
                         nonlinear_score_thresh=0.05):
    """
    Returns a dict layer_name -> decision:
      {
        layer_name: {
          neuron_indices_to_linear: tensor(bool),
          neuron_indices_to_remove: tensor(bool)
        }
      }
    For simplicity we only mark entire layer as having linearizable neurons if positive_frac high.
    More granular per-neuron analysis would require storing per-neuron stats.
    """
    decisions = {}
    for layer, st in stats.items():
        positive_frac = st["positive_frac"]
        negative_frac = st["negative_frac"]
        nonlinear_score = st["nonlinear_score"]
        decision = {}
        # Heuristic: if positive_frac high and nonlinear_score low -> safe to linearize whole layer
        decision["make_layer_linear"] = (positive_frac >= pos_thresh and nonlinear_score <= nonlinear_score_thresh)
        decision["dead_layer"] = (negative_frac >= neg_thresh)
        decisions[layer] = decision
    return decisions

def apply_layer_linearization(model, decisions):
    """
    Replaces ReLU effect by identity in eligible layers
    (For MixedActivationLayer you'd adjust mask; for GatedActivationLayer set alpha=0.)
    Here we approximate by setting weights unchanged and assuming upstream code will treat it linearly.
    """
    for layer_name, decision in decisions.items():
        if decision.get("make_layer_linear"):
            # If gating layer, set alpha->0
            # If mixed, set all to linear; we search modules
            idx = int(layer_name.split("_")[-1])
            # naive iteration
            linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
            if idx < len(linear_layers):
                # No direct change needed for plain Linear; if Mixed/Gated used, user must adapt integration.
                pass
    return model