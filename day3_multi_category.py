"""
Day 3: run the PatchCore baseline over several MVTec AD categories and
write a results table.

Usage:
    python day3_multi_category.py --root data/mvtec_ad \
        --categories bottle screw capsule hazelnut zipper

Results are appended to results/resnet50.csv and printed as a markdown
table you can paste straight into the README.
"""
import argparse
import csv
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

from day2_patchcore import (BANK_MAX, DEVICE, build_extractor, extract_all,
                            nn_distance)


def run_category(extractor, root: Path, cat: str, cache: Path) -> float:
    cat_dir = root / cat
    cat_cache = cache / cat
    cat_cache.mkdir(parents=True, exist_ok=True)

    bank_path = cat_cache / "bank_resnet50.pt"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results/resnet50.csv")
    args = ap.parse_args()

    torch.manual_seed(0)
    extractor = build_extractor()
    print(f"device: {DEVICE}\n")

    results = {}
    for cat in args.categories:
        print(f"== {cat} ==")
        auroc = run_category(extractor, Path(args.root), cat, Path(args.cache))
        results[cat] = auroc
        print(f"   image AUROC = {auroc:.4f}\n")

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if out.stat().st_size == 0:
            w.writerow(["backbone", "category", "image_auroc"])
        for cat, a in results.items():
            w.writerow(["resnet50", cat, f"{a:.4f}"])

    print("| category | image AUROC |")
    print("|---|---|")
    for cat, a in results.items():
        print(f"| {cat} | {a:.4f} |")
    mean = sum(results.values()) / len(results)
    print(f"| **mean** | **{mean:.4f}** |")


if __name__ == "__main__":
    main()
