"""
Day 2: minimal PatchCore-style anomaly scoring on one MVTec AD category.

  1. Extract layer2 + layer3 patch features from train/good  -> memory bank
  2. Extract the same from test/*                              -> query patches
  3. Image score = max over patches of (distance to nearest bank patch)
  4. Report image-level AUROC

Usage:
    python day2_patchcore.py --root data/mvtec_ad --category bottle

Deliberately simple: no coreset subsampling yet (random subsample instead),
no pixel-level scores, no fancy backbones. Those are later days.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.models.feature_extraction import create_feature_extractor

IMG_SIZE = 224
BATCH = 16
BANK_MAX = 20_000  # cap memory bank size; random subsample above this
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

TF = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_extractor():
    backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval().to(DEVICE)
    return create_feature_extractor(backbone, return_nodes={"layer2": "l2", "layer3": "l3"})


@torch.no_grad()
def patch_features(extractor, x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) -> (B, P, D) patch descriptors, P = 28*28 for 224 input."""
    f = extractor(x.to(DEVICE))
    l2, l3 = f["l2"], f["l3"]                                  # (B,512,28,28), (B,1024,14,14)
    l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
    feat = torch.cat([l2, l3], dim=1)                          # (B,1536,28,28)
    feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)  # local neighbourhood aggregation
    return feat.flatten(2).transpose(1, 2).cpu()               # (B, 784, 1536)


def load_dir(paths: list[Path]) -> torch.Tensor:
    return torch.stack([TF(Image.open(p).convert("RGB")) for p in paths])


def extract_all(extractor, paths: list[Path]) -> torch.Tensor:
    out = []
    for i in range(0, len(paths), BATCH):
        out.append(patch_features(extractor, load_dir(paths[i:i + BATCH])))
        print(f"\r  {min(i + BATCH, len(paths))}/{len(paths)}", end="")
    print()
    return torch.cat(out)  # (N, P, D)


@torch.no_grad()
def nn_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """For each query row, L2 distance to its nearest bank row. query (Q,D), bank (M,D) -> (Q,)"""
    bank = bank.to(DEVICE)
    out = []
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk].to(DEVICE)
        d = torch.cdist(q, bank)          # (chunk, M)
        out.append(d.min(dim=1).values.cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()

    cat_dir = Path(args.root) / args.category
    cache = Path(args.cache) / args.category
    cache.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    extractor = build_extractor()
    print(f"device: {DEVICE}")

    # --- memory bank from train/good ---
    bank_path = cache / "bank_resnet50.pt"
    if bank_path.exists():
        bank = torch.load(bank_path)
        print(f"loaded bank {tuple(bank.shape)}")
    else:
        train_paths = sorted((cat_dir / "train" / "good").glob("*.png"))
        print(f"train/good: {len(train_paths)} images")
        feats = extract_all(extractor, train_paths)             # (N, P, D)
        bank = feats.reshape(-1, feats.shape[-1])               # (N*P, D)
        if len(bank) > BANK_MAX:
            idx = torch.randperm(len(bank))[:BANK_MAX]
            bank = bank[idx]
        torch.save(bank, bank_path)
        print(f"bank {tuple(bank.shape)} saved")

    # --- test set ---
    test_paths, labels = [], []
    for sub in sorted((cat_dir / "test").iterdir()):
        for p in sorted(sub.glob("*.png")):
            test_paths.append(p)
            labels.append(0 if sub.name == "good" else 1)
    print(f"test: {len(test_paths)} images ({sum(labels)} defective)")

    q = extract_all(extractor, test_paths)                      # (T, P, D)
    T, P, D = q.shape
    d = nn_distance(q.reshape(-1, D), bank).reshape(T, P)       # (T, P)
    image_scores = d.max(dim=1).values

    auroc = roc_auc_score(labels, image_scores.numpy())
    print(f"\n{args.category} | resnet50 | image AUROC = {auroc:.4f}")


if __name__ == "__main__":
    main()
