# defect-transfer-bench

A small, honest benchmark of how well pretrained vision backbones transfer to
industrial anomaly detection without fine-tuning.

The method is fixed and deliberately simple (PatchCore-style kNN on patch
features). Only the backbone changes. The question is: **which pretrained
representations carry the local structure that defect detection needs?**

## Method

For each backbone, two intermediate feature maps (a fine and a coarse one) are
concatenated per spatial position, locally averaged (3×3), and treated as patch
descriptors. Patches from the *good* training images form a memory bank. A test
image's anomaly score is the largest nearest-neighbour distance across its
patches. Metric is image-level AUROC.

| | fine / coarse | descriptor dim | grid @224 |
|---|---|---|---|
| resnet50 | layer2 / layer3 | 1536 | 28×28 |
| wide_resnet50 | layer2 / layer3 | 1536 | 28×28 |
| convnext_tiny | stage2 / stage3 | 576 | 28×28 |
| dinov2_vits14 | block 5 / block 8 | 768 | 16×16 |

Current settings: 224×224 input, random 20k-patch memory bank (no coreset yet),
ImageNet normalization. These are known to be sub-optimal; the point at this
stage is a controlled comparison, not a leaderboard.

## Results — MVTec AD, image AUROC

| category | convnext_tiny | dinov2_vits14 | resnet50 | wide_resnet50 |
|---|---|---|---|---|
| bottle | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| capsule | 0.7890 | 0.8851 | 0.9071 | 0.9055 |
| hazelnut | 1.0000 | 0.9993 | 1.0000 | 1.0000 |
| screw | 0.7237 | 0.6628 | 0.8262 | 0.7518 |
| zipper | 0.8803 | 0.9601 | 0.9569 | 0.9617 |
| **mean** | **0.8786** | **0.9015** | **0.9380** | **0.9238** |

## Findings so far

- **bottle** and **hazelnut** are saturated for every backbone; **screw** and
  **capsule** are where backbones actually differ. Future comparisons should
  weight those.
- **ConvNeXt-T** underperforms ResNet-50 by ~6 points mean and ~10 on screw,
  despite better ImageNet accuracy. Narrower stage widths and a smoother,
  more classification-tuned feature space are the likely causes.
- **DINOv2-S** is the strongest on zipper and near-best on capsule, but the
  worst on screw. At 224 input a ViT-S/14 token covers a 14-pixel patch on a
  16×16 grid, so small defects are averaged away. This is a **granularity
  confound**, not (only) a representation difference — resolving it is the
  next experiment.
- **Wide-ResNet-50** does not beat ResNet-50 here, which suggests the random
  20k bank is a bottleneck: the wider net produces more useful patches than
  the bank keeps.

## Roadmap

- [ ] Resolution ablation (DINOv2 at 448, CNNs at 320) to separate granularity
      from representation
- [ ] Greedy coreset subsampling instead of random
- [ ] Pixel-level AUROC / anomaly maps
- [ ] Remaining MVTec AD categories; VisA, DAGM, NEU-DET
- [ ] CLIP and YOLOX backbones

## Setup

```
pip install torch torchvision scikit-learn pillow
python download_mvtec.py --out data/mvtec_ad --categories bottle screw capsule hazelnut zipper
python run_benchmark.py --backbone resnet50 --categories bottle screw capsule hazelnut zipper
python run_benchmark.py --table
```

Backbones are registered in `backbones.py`; adding one is a ~5-line function.
Memory banks are cached under `cache/` (git-ignored). Results land in
`results/<backbone>.csv`.

## References

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), CVPR 2022

MVTec AD is CC BY-NC-SA 4.0; this repo does not redistribute it.
