import torch
import torch.nn as nn
from torchvision.models import resnet50
from nonlinear_mlp.models.mlp import MLP

def build_resnet50_with_mlp_head(
    num_classes: int,
    head_hidden: list = [2048, 512],
    approach: str = "fixed",
    fixed_cfg=None,
    gating_cfg=None,
    layerwise_cfg=None,
    pretrained: bool = False,
    freeze_backbone: bool = False,
):
    model = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    in_features = model.fc.in_features  # typically 2048
    backbone = nn.Sequential(*(list(model.children())[:-1]))
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
    return ResNet50HeadWrapper(backbone, head_mlp)

class ResNet50HeadWrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.head(x)