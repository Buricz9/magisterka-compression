"""Smoke test the patched compress_image_jpeg2000 / compress_image_avif on one image.

Verifies that the previously-buggy CR<=1.0 case (Q=100 on a typical ARCADE PNG)
now produces lossy output, not bit-exact lossless.
"""
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.processing.compress_images import (
    compress_image_jpeg2000,
    compress_image_avif,
)

# Pick a test image known to have CR<1 for Q=100 (test split, 99.7%).
test_dir = config.get_data_path("syntax", "test", quality=None)
img_path = sorted(test_dir.glob("*.png"))[0]
png_size = img_path.stat().st_size
png_arr = np.array(Image.open(img_path).convert("RGB"))

# Load actual CR for Q=100 from cr_map (will be < 1)
cr_map = json.loads((PROJECT_ROOT / "dataset" / "cr_maps" / "syntax_test_cr_map.json").read_text())
cr_q100 = cr_map["100"][img_path.stem]
print(f"Image: {img_path.name}")
print(f"PNG size: {png_size}")
print(f"CR (PNG/JPEG_q100) = {cr_q100:.3f}  -> target_size = {png_size/cr_q100:.0f}")
print()

out_dir = PROJECT_ROOT / "src" / "processing" / "_smoke_out"
out_dir.mkdir(exist_ok=True)

# --- JPEG2000 with the buggy-but-now-fixed CR<=1 path ---
jp2_path = out_dir / "test.jp2"
compress_image_jpeg2000(img_path, jp2_path, cr_q100)
jp2_arr = np.array(Image.open(jp2_path).convert("RGB"))
jp2_size = jp2_path.stat().st_size
jp2_psnr = psnr(png_arr, jp2_arr, data_range=255)
jp2_bitexact = np.array_equal(png_arr, jp2_arr)
print(f"JPEG2000: size={jp2_size}  PSNR={jp2_psnr:.2f}dB  bit_exact={jp2_bitexact}")

# --- AVIF ditto ---
avif_path = out_dir / "test.avif"
compress_image_avif(img_path, avif_path, cr_q100)
avif_arr = np.array(Image.open(avif_path).convert("RGB"))
avif_size = avif_path.stat().st_size
avif_psnr = psnr(png_arr, avif_arr, data_range=255)
avif_bitexact = np.array_equal(png_arr, avif_arr)
print(f"AVIF:     size={avif_size}  PSNR={avif_psnr:.2f}dB  bit_exact={avif_bitexact}")
print()

if not jp2_bitexact and not avif_bitexact and np.isfinite(jp2_psnr) and np.isfinite(avif_psnr):
    print("PASS: oba formaty zapisane stratnie (jak chcielismy).")
else:
    print("FAIL: bug nadal sie wyzwala dla CR<=1.0 (cos jest nie tak z patchem).")
