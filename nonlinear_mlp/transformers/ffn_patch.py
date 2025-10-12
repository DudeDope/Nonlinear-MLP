import torch
import torch.nn as nn
from typing import Literal, Optional
from nonlinear_mlp.layers.gated import GatedActivationLayer
from nonlinear_mlp.layers.mixed import MixedActivationLayer

"""
Usage:
    from transformers import AutoModel
    model = AutoModel.from_pretrained('bert-base-uncased')
    patch_transformer_ffn(
        model,
        approach='gating',
        gating_kwargs={'init_alpha':0.7, 'temperature':1.0},
        fixed_kwargs={'linear_ratio':0.5, 'pattern':'structured'}
    )
"""

def patch_transformer_ffn(
    model: nn.Module,
    approach: Literal['gating','fixed'] = 'gating',
    gating_kwargs: Optional[dict] = None,
    fixed_kwargs: Optional[dict] = None,
    target_module_names=('intermediate',),  # typical for BERT
    verbose: bool = True
):
    gating_kwargs = gating_kwargs or {}
    fixed_kwargs = fixed_kwargs or {}

    replaced = 0
    for name, module in model.named_modules():
        # Heuristic: locate submodules that have a dense -> activation -> dense pattern
        # BERT intermediate: module.intermediate.dense (Linear) + activation (GELU)
        if any(tn in name for tn in target_module_names):
            # Try to identify a Linear followed by activation
            # In BERT: BertIntermediate(dense=Linear, intermediate_act_fn=GELU)
            if hasattr(module, 'dense'):
                linear = module.dense
                in_f = linear.in_features
                out_f = linear.out_features
                if approach == 'gating':
                    new_layer = GatedActivationLayer(
                        in_f, out_f,
                        init_alpha=gating_kwargs.get('init_alpha', 0.75),
                        temperature=gating_kwargs.get('temperature', 1.0),
                        clamp=gating_kwargs.get('clamp', True)
                    )
                    # Copy weights
                    new_layer.linear.weight.data.copy_(linear.weight.data)
                    if linear.bias is not None:
                        new_layer.linear.bias.data.copy_(linear.bias.data)
                    module.dense = new_layer.linear  # keep shape
                    module.custom_gate = new_layer  # store gate separately
                    module.intermediate_act_fn = lambda x: new_layer(x)  # override path
                elif approach == 'fixed':
                    ratio = fixed_kwargs.get('linear_ratio', 0.5)
                    pattern = fixed_kwargs.get('pattern', 'structured')
                    new_layer = MixedActivationLayer(
                        in_f, out_f, linear_ratio=ratio, pattern=pattern
                    )
                    new_layer.linear.weight.data.copy_(linear.weight.data)
                    if linear.bias is not None:
                        new_layer.linear.bias.data.copy_(linear.bias.data)
                    module.dense = new_layer.linear
                    module.custom_mixed = new_layer
                    module.intermediate_act_fn = lambda x: new_layer.linear(x) if ratio == 1.0 else new_layer(x)
                replaced += 1
    if verbose:
        print(f"[FFN Patch] Replaced {replaced} intermediate projections with {approach} approach.")
    return model

def extract_gate_stats(model):
    stats = []
    for name, module in model.named_modules():
        if hasattr(module, 'custom_gate'):
            gate = module.custom_gate
            a = gate.alpha().detach()
            stats.append({
                'module': name,
                'alpha_mean': float(a.mean()),
                'alpha_median': float(a.median()),
                'alpha_lt_0.1': float((a < 0.1).float().mean())
            })
    return stats