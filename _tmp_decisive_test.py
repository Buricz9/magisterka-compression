"""Decisive test: Agent A vs Agent C dispute.

Test: take 3 images, compress with JPEG Q=10 to get target CR.
Then compress with JPEG2000 to MATCH that CR.
Check:
- Is JP2 file size close to target (±5%)?
- Is PSNR sensibly low (~25-30 dB)?
- Did binary search converge?
"""
import sys
from pathlib import Path
import io
import math
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"C:\Uczelnia\Magisterka")
sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.compress_images import (
    compress_image_jpeg,
    compress_image_jpeg2000,
)


def psnr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def test_one(img_path, jpeg_q, tmpdir):
    """For one image: compress JPEG Q, get CR, then JP2 matched CR."""
    orig_size = img_path.stat().st_size
    png_arr = np.array(Image.open(img_path).convert("RGB"))

    # 1) JPEG at quality Q
    jpg_path = tmpdir / f"{img_path.stem}_q{jpeg_q}.jpg"
    compress_image_jpeg(img_path, jpg_path, jpeg_q)
    jpg_size = jpg_path.stat().st_size
    jpeg_cr = orig_size / jpg_size
    jpg_arr = np.array(Image.open(jpg_path).convert("RGB"))
    jpeg_psnr = psnr(png_arr, jpg_arr)

    # 2) JPEG2000 matched to that CR
    jp2_path = tmpdir / f"{img_path.stem}_q{jpeg_q}_matched.jp2"
    compress_image_jpeg2000(img_path, jp2_path, jpeg_cr)
    jp2_size = jp2_path.stat().st_size
    jp2_cr = orig_size / jp2_size
    jp2_arr = np.array(Image.open(jp2_path).convert("RGB"))
    jp2_psnr = psnr(png_arr, jp2_arr)

    target_size = orig_size / jpeg_cr
    diff_pct = abs(jp2_size - target_size) / target_size * 100

    return {
        "image": img_path.name,
        "orig_size": orig_size,
        "jpeg_q": jpeg_q,
        "jpeg_size": jpg_size,
        "jpeg_cr": jpeg_cr,
        "jpeg_psnr": jpeg_psnr,
        "target_jp2_size": target_size,
        "jp2_size": jp2_size,
        "jp2_cr": jp2_cr,
        "jp2_psnr": jp2_psnr,
        "size_diff_pct_vs_target": diff_pct,
    }


def main():
    test_dir = PROJECT_ROOT / "dataset" / "arcade" / "syntax" / "test" / "images"
    images = sorted(test_dir.glob("*.png"))
    # Pick 3 "random" but deterministic
    rng = np.random.default_rng(42)
    chosen_idx = rng.choice(len(images), size=3, replace=False)
    chosen = [images[i] for i in chosen_idx]

    tmpdir = PROJECT_ROOT / "_tmp_jp2_test"
    tmpdir.mkdir(exist_ok=True)

    print("=" * 100)
    for q in [70, 50, 30, 10]:
        print(f"\n--- JPEG Q = {q} (so JP2 must match the resulting CR) ---")
        for img in chosen:
            r = test_one(img, q, tmpdir)
            print(
                f"  {r['image']:>8s}  orig={r['orig_size']:>7d}  "
                f"jpeg_size={r['jpeg_size']:>7d}  jpeg_CR={r['jpeg_cr']:>6.2f}  "
                f"jpeg_PSNR={r['jpeg_psnr']:>5.2f}  "
                f"-> target_jp2={int(r['target_jp2_size']):>7d}  "
                f"actual_jp2={r['jp2_size']:>7d}  jp2_CR={r['jp2_cr']:>6.2f}  "
                f"jp2_PSNR={r['jp2_psnr']:>5.2f}  "
                f"size_diff={r['size_diff_pct_vs_target']:>5.1f}%"
            )


if __name__ == "__main__":
    main()
