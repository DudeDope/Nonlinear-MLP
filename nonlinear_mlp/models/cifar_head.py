import torch
import torch.nn as nn
from torchvision.models import resnet18
from nonlinear_mlp.models.mlp import MLP

def build_resnet18_with_mlp_head(
    num_classes: int,
    head_hidden: list = [512, 256],
    approach="fixed",
    fixed_cfg=None,
    gating_cfg=None,
    layerwise_cfg=None,
    pretrained: bool = False,
    freeze_backbone: bool = False,
):
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.fc.in_features
    backbone = nn.Sequential(*(list(model.children())[:-1]))  # remove original fc
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    head_mlp = MLP(
        input_dim=in_features,
        hidden_dims=head_hidden,
        num_classes=num_classes,
        approach=approach,
        fixed_cfg=fixed_cfg or {},
        gating_cfg=gating_cfg or {},
        layerwise_cfg=layerwise_cfg or {},
    )
    wrapper = ResNetHeadWrapper(backbone, head_mlp)
    return wrapper

class ResNetHeadWrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.head(x)