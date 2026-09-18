"""
Benchmark runner: PatchCore-style kNN scoring, any backbone from backbones.py,
any set of MVTec AD categories.

Usage:
    python run_benchmark.py --root data/mvtec_ad --backbone convnext_tiny \
        --categories bottle screw capsule hazelnut zipper

    python run_benchmark.py --table        # print backbone x category table from results/

Banks are cached per (backbone, category) under cache/. Results go to
results/<backbone>.csv (overwritten per category on rerun).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score

import backbones

IMG_SIZE = 224
BATCH = 16
BANK_MAX = 20_000
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

TF = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def patch_features(extractor, x: torch.Tensor) -> torch.Tensor:
    f = extractor(x.to(DEVICE))
    fine, coarse = f["fine"], f["coarse"]
    coarse = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
    feat = torch.cat([fine, coarse], dim=1)
    feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
    return feat.flatten(2).transpose(1, 2).cpu()          # (B, P, D)


def extract_all(extractor, paths: list[Path]) -> torch.Tensor:
    out = []
    for i in range(0, len(paths), BATCH):
        x = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i + BATCH]])
        out.append(patch_features(extractor, x))
        print(f"\r    {min(i + BATCH, len(paths))}/{len(paths)}", end="")
    print()
    return torch.cat(out)


@torch.no_grad()
def nn_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    bank = bank.to(DEVICE)
    out = []
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i:i + chunk].to(DEVICE), bank)
        out.append(d.min(dim=1).values.cpu())
    return torch.cat(out)


def run_category(extractor, name: str, root: Path, cat: str, cache: Path) -> float:
    cat_dir = root / cat
    bank_path = cache / cat / f"bank_{name}.pt"
    bank_path.parent.mkdir(parents=True, exist_ok=True)

    if bank_path.exists():
        bank = torch.load(bank_path)
    else:
        train_paths = sorted((cat_dir / "train" / "good").glob("*.png"))
        feats = extract_all(extractor, train_paths)
        bank = feats.reshape(-1, feats.shape[-1])
        if len(bank) > BANK_MAX:
            bank = bank[torch.randperm(len(bank))[:BANK_MAX]]
        torch.save(bank, bank_path)

    test_paths, labels = [], []
    for sub in sorted((cat_dir / "test").iterdir()):
        for p in sorted(sub.glob("*.png")):
            test_paths.append(p)
            labels.append(0 if sub.name == "good" else 1)

    q = extract_all(extractor, test_paths)
    T, P, D = q.shape
    d = nn_distance(q.reshape(-1, D), bank).reshape(T, P)
    return roc_auc_score(labels, d.max(dim=1).values.numpy())


def save_results(path: Path, name: str, results: dict[str, float]) -> None:
    """Merge into results/<backbone>.csv, replacing rows for categories rerun."""
    rows = {}
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                rows[r["category"]] = r["image_auroc"]
    for cat, a in results.items():
        rows[cat] = f"{a:.4f}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "category", "image_auroc"])
        for cat in sorted(rows):
            w.writerow([name, cat, rows[cat]])


def print_table(results_dir: Path) -> None:
    table: dict[str, dict[str, float]] = defaultdict(dict)   # cat -> backbone -> auroc
    names = []
    for p in sorted(results_dir.glob("*.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                table[r["category"]][r["backbone"]] = float(r["image_auroc"])
                if r["backbone"] not in names:
                    names.append(r["backbone"])
    if not table:
        print("no results yet")
        return
    print("| category | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    means = defaultdict(list)
    for cat in sorted(table):
        cells = []
        for n in names:
            a = table[cat].get(n)
            cells.append(f"{a:.4f}" if a is not None else "—")
            if a is not None:
                means[n].append(a)
        print(f"| {cat} | " + " | ".join(cells) + " |")
    print("| **mean** | " + " | ".join(f"**{sum(means[n]) / len(means[n]):.4f}**" for n in names) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mvtec_ad")
    ap.add_argument("--backbone", default="resnet50", choices=sorted(backbones.REGISTRY))
    ap.add_argument("--categories", nargs="+", default=["bottle"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results")
    ap.add_argument("--table", action="store_true", help="print combined table and exit")
    args = ap.parse_args()

    if args.table:
        print_table(Path(args.results))
        return

    torch.manual_seed(0)
    extractor = backbones.build(args.backbone).to(DEVICE)
    print(f"backbone: {args.backbone} | device: {DEVICE}\n")

    results = {}
    for cat in args.categories:
        print(f"== {cat} ==")
        results[cat] = run_category(extractor, args.backbone, Path(args.root), cat, Path(args.cache))
        print(f"    image AUROC = {results[cat]:.4f}\n")

    save_results(Path(args.results) / f"{args.backbone}.csv", args.backbone, results)
    print_table(Path(args.results))


if __name__ == "__main__":
    main()
