"""
Download and extract MVTec AD categories from the official MVTec links.

Usage:
    python download_mvtec.py --out data/mvtec_ad --categories bottle
    python download_mvtec.py --out data/mvtec_ad --categories bottle screw zipper
    python download_mvtec.py --out data/mvtec_ad --all

Each category lands at <out>/<category>/{train,test,ground_truth}/...
Already-extracted categories are skipped. Partial downloads resume.

MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial). Fine for a public
benchmark; cite Bergmann et al., CVPR 2019.
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://www.mydrive.ch/shares"
URLS = {
    "bottle":     f"{BASE}/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz",
    "cable":      f"{BASE}/150453/10f960e07fec2838b2cd512586633a32/download/420937413-1629958794/cable.tar.xz",
    "capsule":    f"{BASE}/150454/e0ce6dd74eb150f46c0d98131b3703f2/download/420937454-1629958872/capsule.tar.xz",
    "carpet":     f"{BASE}/150455/eac7fbce84d93a5094e13f391170eca4/download/420937484-1629959013/carpet.tar.xz",
    "grid":       f"{BASE}/150456/bb0b2e3dc804ccb8b4485e01f8e4493b/download/420937487-1629959044/grid.tar.xz",
    "hazelnut":   f"{BASE}/150457/51d49f65e84bdc100ff8035d7dd783ea/download/420937545-1629959162/hazelnut.tar.xz",
    "leather":    f"{BASE}/150458/923030ce14e10a7d147e95d0f8885f6b/download/420937607-1629959262/leather.tar.xz",
    "metal_nut":  f"{BASE}/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz",
    "pill":       f"{BASE}/150460/d4f1c04da67034ccc8d8b5fa3e73f244/download/420938129-1629960351/pill.tar.xz",
    "screw":      f"{BASE}/150461/242f454cc6385e5693c4fd4b94567d1e/download/420938130-1629960389/screw.tar.xz",
    "tile":       f"{BASE}/150462/5479f0fdc97bc6fa16eab0cb0cf0109f/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": f"{BASE}/150463/895a1a5fb84b958f07417784b060434b/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": f"{BASE}/150464/99b8ea0332438d64fcc2475ee73f9c29/download/420938166-1629960554/transistor.tar.xz",
    "wood":       f"{BASE}/150465/d5b4115b720cdb54d217e75636e6e374/download/420938383-1629960649/wood.tar.xz",
    "zipper":     f"{BASE}/150466/bd155b557520edaf692d9bfdb915c24a/download/420938385-1629960680/zipper.tar.xz",
}

CHUNK = 1 << 20  # 1 MB


def download(url: str, dest: Path) -> None:
    """Stream to disk with resume support and a simple progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored Range; start over
        total = have + int(resp.headers.get("Content-Length", 0))
        mode = "ab" if have else "wb"
        with open(dest, mode) as f:
            done = have
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:6.1f} MB", end="")
    print()


def extract(archive: Path, out: Path) -> None:
    print(f"  extracting {archive.name} ...")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(out)


def fetch_category(name: str, out: Path, keep_archive: bool) -> None:
    if (out / name / "train").exists():
        print(f"[{name}] already extracted, skipping")
        return
    archive = out / "_archives" / f"{name}.tar.xz"
    print(f"[{name}] downloading")
    download(URLS[name], archive)
    extract(archive, out)
    if not keep_archive:
        archive.unlink()
    print(f"[{name}] done -> {out / name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mvtec_ad")
    ap.add_argument("--categories", nargs="*", default=["bottle"], choices=sorted(URLS))
    ap.add_argument("--all", action="store_true", help="download all 15 categories (~4.9 GB)")
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    cats = sorted(URLS) if args.all else args.categories
    out = Path(args.out)
    for c in cats:
        fetch_category(c, out, args.keep_archive)


if __name__ == "__main__":
    main()
