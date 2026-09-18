"""
Backbone registry for the representation-transfer benchmark.

Every backbone yields a callable  x -> {"fine": (B,C1,H,W), "coarse": (B,C2,H',W')}.
CNNs read two stages (stride 8 and 16). ViTs read two intermediate blocks at the
same resolution (224/14 = 16x16 for DINOv2). The scoring code never changes.

Add a backbone with @register("name"); the function returns either
  (torch.nn.Module, fine_node, coarse_node)   -> wrapped via create_feature_extractor
or a torch.nn.Module whose forward already returns the {"fine","coarse"} dict.
"""
from typing import Callable

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

REGISTRY: dict[str, Callable[[], object]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


# ---------------- CNNs ----------------

@register("resnet50")
def _resnet50():
    m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    return m, "layer2", "layer3"            # 512@28x28, 1024@14x14


@register("wide_resnet50")
def _wide_resnet50():
    m = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V2")
    return m, "layer2", "layer3"


@register("convnext_tiny")
def _convnext_tiny():
    m = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
    return m, "features.3", "features.5"    # 192@28x28, 384@14x14


# ---------------- DINOv2 (ViT) ----------------

class DinoV2Wrapper(torch.nn.Module):
    """Read two intermediate blocks of a DINOv2 ViT as fine/coarse patch-token grids."""

    def __init__(self, hub_name: str, fine_block: int, coarse_block: int):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", hub_name)
        self.blocks = [fine_block, coarse_block]

    def forward(self, x):
        # get_intermediate_layers with reshape=True -> list of (B, C, H/14, W/14), CLS token dropped
        fine, coarse = self.model.get_intermediate_layers(x, n=self.blocks, reshape=True)
        return {"fine": fine, "coarse": coarse}


@register("dinov2_vits14")
def _dinov2_s():
    return DinoV2Wrapper("dinov2_vits14", fine_block=5, coarse_block=8)    # 384-d, 12 blocks total


@register("dinov2_vitb14")
def _dinov2_b():
    return DinoV2Wrapper("dinov2_vitb14", fine_block=5, coarse_block=8)    # 768-d, 12 blocks total


# ---------------- builder ----------------

def build(name: str) -> torch.nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; available: {sorted(REGISTRY)}")
    out = REGISTRY[name]()
    if isinstance(out, tuple):
        model, fine, coarse = out
        model = create_feature_extractor(model, return_nodes={fine: "fine", coarse: "coarse"})
    else:
        model = out
    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
