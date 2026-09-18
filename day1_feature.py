"""
Day 1: load a handful of MVTec AD images, push them through a pretrained
ResNet-50, print the intermediate feature-map shapes.

Usage:
    python day1_features.py --root /path/to/mvtec_ad --category bottle

If --root is omitted, the script fabricates 4 random images so it still
runs end to end. That's fine for day one.
"""
import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor

IMG_SIZE = 224
MAX_IMAGES = 4

# PatchCore uses mid-level layers (layer2 + layer3). Grab those.
RETURN_NODES = {"layer2": "layer2", "layer3": "layer3"}


def load_images(root: str | None, category: str) -> torch.Tensor:
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if root is None:
        print("No --root given; using random tensors.")
        return torch.randn(MAX_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    good_dir = Path(root) / category / "train" / "good"
    paths = sorted(good_dir.glob("*.png"))[:MAX_IMAGES]
    if not paths:
        raise FileNotFoundError(f"No PNGs found in {good_dir}")
    print(f"Loaded {len(paths)} images from {good_dir}")
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="MVTec AD root folder")
    ap.add_argument("--category", default="bottle")
    args = ap.parse_args()

    x = load_images(args.root, args.category)

    backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
    extractor = create_feature_extractor(backbone, return_nodes=RETURN_NODES)

    with torch.no_grad():
        feats = extractor(x)

    print(f"Input: {tuple(x.shape)}")
    for name, t in feats.items():
        print(f"{name}: {tuple(t.shape)}")


if __name__ == "__main__":
    main()
