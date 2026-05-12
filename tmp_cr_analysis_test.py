"""
Empiryczny test wpływu zmiany definicji CR na binary search i JP2 direct encoding.
Uruchom: & C:\Uczelnia\Magisterka\venv\Scripts\python.exe C:\Uczelnia\Magisterka\tmp_cr_analysis_test.py
"""
import sys
from pathlib import Path
from PIL import Image
import tempfile
import math

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

import numpy as np

PROJECT_ROOT = Path("C:/Uczelnia/Magisterka")
img_path = PROJECT_ROOT / "dataset/arcade/syntax/test/images/1.png"


def psnr(original: Image.Image, compressed: Image.Image) -> float:
    a = np.array(original, dtype=np.float64)
    b = np.array(compressed, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def compress_jp2_search(img: Image.Image, out_path: Path, target_size: float) -> int:
    """Binary search — current production logic."""
    lo, hi = 1.0, 1000.0
    best_rate = (lo + hi) / 2.0
    best_size = None
    best_diff = float("inf")
    tolerance = 0.05 * target_size

    for _ in range(20):
        mid = (lo + hi) / 2.0
        img.save(
            out_path, "JPEG2000",
            irreversible=True, quality_mode="rates",
            quality_layers=[mid], mct=1,
        )
        actual = out_path.stat().st_size
        diff = abs(actual - target_size)
        if diff < best_diff:
            best_diff = diff
            best_rate = mid
            best_size = actual
        if diff < tolerance:
            break
        if actual > target_size:
            lo = mid
        else:
            hi = mid

    if best_size is None or out_path.stat().st_size != best_size:
        img.save(
            out_path, "JPEG2000",
            irreversible=True, quality_mode="rates",
            quality_layers=[best_rate], mct=1,
        )
    return out_path.stat().st_size


def main():
    assert img_path.exists(), f"Image not found: {img_path}"
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size
    png_size = img_path.stat().st_size
    RAW = H * W * 3

    print(f"Image: {img_path.name}  size={img.size}  mode={img.mode}")
    print(f"PNG size  : {png_size:>10,} B")
    print(f"RAW size  : {RAW:>10,} B  (H={H} x W={W} x 3)")
    print(f"RAW/PNG   : {RAW/png_size:.4f}")
    print()

    tmpdir = Path(tempfile.gettempdir())
    rows = []

    for Q in [100, 70, 30]:
        # --- JPEG reference ---
        jpeg_path = tmpdir / f"arcade1_Q{Q}.jpg"
        img.save(jpeg_path, "JPEG", quality=Q, subsampling=0, optimize=True)
        jpeg_size = jpeg_path.stat().st_size

        CR_PNG = png_size / jpeg_size   # current definition
        CR_RAW = RAW / jpeg_size        # proposed definition

        # target_size is identical under both definitions:
        #   current : PNG_size / CR_PNG = PNG_size / (PNG_size/JPEG_size) = JPEG_size
        #   proposed: RAW_size / CR_RAW = RAW_size / (RAW_size/JPEG_size) = JPEG_size
        target_size = float(jpeg_size)

        # --- JP2 direct (no binary search): rate = CR_RAW passed directly ---
        jp2_dir_path = tmpdir / f"arcade1_Q{Q}_direct.jp2"
        img.save(
            jp2_dir_path, "JPEG2000",
            irreversible=True, quality_mode="rates",
            quality_layers=[CR_RAW], mct=1,
        )
        jp2_dir_size = jp2_dir_path.stat().st_size

        # --- JP2 direct with CR_PNG (to show the difference) ---
        jp2_dir_png_path = tmpdir / f"arcade1_Q{Q}_direct_crpng.jp2"
        img.save(
            jp2_dir_png_path, "JPEG2000",
            irreversible=True, quality_mode="rates",
            quality_layers=[CR_PNG], mct=1,
        )
        jp2_dir_png_size = jp2_dir_png_path.stat().st_size

        # --- JP2 binary search with target = JPEG_size ---
        jp2_srch_path = tmpdir / f"arcade1_Q{Q}_search.jp2"
        jp2_srch_size = compress_jp2_search(img, jp2_srch_path, target_size)

        # --- PSNR ---
        ref = img
        psnr_direct = psnr(ref, Image.open(jp2_dir_path).convert("RGB"))
        psnr_direct_png = psnr(ref, Image.open(jp2_dir_png_path).convert("RGB"))
        psnr_search = psnr(ref, Image.open(jp2_srch_path).convert("RGB"))

        rows.append(dict(
            Q=Q,
            jpeg_size=jpeg_size,
            CR_PNG=CR_PNG,
            CR_RAW=CR_RAW,
            target=int(target_size),
            dir_raw=jp2_dir_size,
            dir_png=jp2_dir_png_size,
            srch=jp2_srch_size,
            psnr_dir_raw=psnr_direct,
            psnr_dir_png=psnr_direct_png,
            psnr_srch=psnr_search,
        ))

    # ---- Print table ----
    hdr = (f"{'Q':>5} | {'JPEG_B':>8} | {'CR_PNG':>7} | {'CR_RAW':>7} | "
           f"{'target_B':>9} | {'dir_RAW_B':>10} | {'err_dR%':>8} | "
           f"{'dir_PNG_B':>10} | {'err_dP%':>8} | "
           f"{'search_B':>9} | {'err_s%':>7} | "
           f"{'PSNR_dR':>8} | {'PSNR_dP':>8} | {'PSNR_s':>8}")
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        def err(got, tgt): return (got - tgt) / tgt * 100
        print(
            f"{r['Q']:>5} | {r['jpeg_size']:>8,} | {r['CR_PNG']:>7.3f} | {r['CR_RAW']:>7.3f} | "
            f"{r['target']:>9,} | {r['dir_raw']:>10,} | {err(r['dir_raw'],r['target']):>+7.1f}% | "
            f"{r['dir_png']:>10,} | {err(r['dir_png'],r['target']):>+7.1f}% | "
            f"{r['srch']:>9,} | {err(r['srch'],r['target']):>+6.1f}% | "
            f"{r['psnr_dir_raw']:>8.2f} | {r['psnr_dir_png']:>8.2f} | {r['psnr_srch']:>8.2f}"
        )

    print()
    print("Legenda:")
    print("  dir_RAW  = JP2 bezposrednio z quality_layers=[CR_RAW], bez binary search")
    print("  dir_PNG  = JP2 bezposrednio z quality_layers=[CR_PNG], bez binary search")
    print("  search   = JP2 z binary search (aktualna logika produkcyjna)")
    print("  err_%    = (JP2_size - target_size) / target_size * 100%")
    print("  target_B = JPEG_size (identyczny dla obu definicji CR)")


if __name__ == "__main__":
    main()
