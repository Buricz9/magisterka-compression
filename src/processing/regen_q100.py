"""Regenerate JPEG2000 and AVIF compressed images for Q=100 only.

After fixing the `compression_ratio <= 1.0` branch in compress_images.py, the
existing Q=100 JP2/AVIF files (which are bit-exact lossless copies of the PNG)
need to be replaced with the correctly lossy versions produced by binary search.
Q < 100 is unaffected — those compressions ran through the (correct) binary
search path and don't need regeneration.

Usage:
    python -m src.processing.regen_q100
"""
import sys
import json
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.processing.compress_images import (
    compress_image_jpeg2000,
    compress_image_avif,
    get_extension,
    get_cr_map_path,
)


def regen_split(task: str, split: str, fmt: str, quality: int = 100):
    source_dir = config.get_data_path(task, split, quality=None)
    if not source_dir.exists():
        print(f"  [skip] {source_dir} not found")
        return

    image_files = sorted(source_dir.glob("*.png"))
    if not image_files:
        print(f"  [skip] no PNGs in {source_dir}")
        return

    cr_map_path = get_cr_map_path(task, split)
    if not cr_map_path.exists():
        print(f"  [error] CR map missing: {cr_map_path}")
        return
    with open(cr_map_path) as f:
        cr_map = json.load(f)
    quality_cr = cr_map.get(str(quality), {})
    if not quality_cr:
        print(f"  [error] no CR entries for Q={quality} in {cr_map_path}")
        return

    out_dir = config.get_data_path(task, split, quality=quality, format=fmt)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = get_extension(fmt)
    compress_fn = compress_image_jpeg2000 if fmt == "jpeg2000" else compress_image_avif

    ok = 0
    fail = 0
    for img_path in tqdm(image_files, desc=f"{fmt} {split} Q={quality}"):
        cr = quality_cr.get(img_path.stem)
        if cr is None:
            fail += 1
            continue
        out_path = out_dir / f"{img_path.stem}{ext}"
        # Force overwrite — the existing files are the broken lossless ones.
        if compress_fn(img_path, out_path, cr):
            ok += 1
        else:
            fail += 1
    print(f"  {fmt} {split} Q={quality}: ok={ok} fail={fail}")


def main():
    task = "syntax"
    splits = ["train", "val", "test"]
    formats = ["jpeg2000", "avif"]
    quality = 100

    print("=" * 70)
    print(f"Regenerating Q={quality} for {formats} on task={task}")
    print("=" * 70)

    for fmt in formats:
        print(f"\n=== {fmt.upper()} ===")
        for split in splits:
            regen_split(task, split, fmt, quality=quality)

    print("\nDone.")


if __name__ == "__main__":
    main()
