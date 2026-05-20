"""Compress ARCADE dataset images to various formats (JPEG, JPEG2000, AVIF).

Matched compression ratio approach:
- JPEG is compressed first using quality factor Q.
- For each JPEG image, the achieved compression ratio (CR) is measured.
- JPEG2000 and AVIF are then compressed to match the same CR per image.

CR map is saved to disk to avoid re-compressing JPEG multiple times.
"""
import sys
import json
from pathlib import Path
from PIL import Image
try:
    import pillow_avif  # noqa: F401  # registers AVIF plugin in Pillow
except ImportError:
    pass
import argparse
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def get_cr_map_path(task, split):
    """Get path to CR map file for caching."""
    return PROJECT_ROOT / "dataset" / "cr_maps" / f"{task}_{split}_cr_map.json"


def compress_image_jpeg(input_path, output_path, quality):
    """Compress to JPEG format using quality factor.

    Uses subsampling=0 (4:4:4) to preserve chroma information,
    which is important for medical images where color may be diagnostically relevant.
    """
    try:
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # subsampling=0 (4:4:4) preserves full chroma resolution for medical images
        img.save(output_path, "JPEG", quality=quality, subsampling=0, optimize=True)
        return True
    except Exception as e:
        print(f"Error (JPEG): {input_path.name}: {e}")
        return False


def compress_image_jpeg2000(input_path, output_path, compression_ratio):
    """Compress to JPEG2000 to match a target compression ratio.

    `compression_ratio` is interpreted as RAW-based CR (raw_pixel_bytes / file_bytes),
    matching JPEG2000 / DICOM / AVIF literature convention. Target file size is
    therefore `raw_size / compression_ratio`, which equals the corresponding JPEG
    file size for the matched-CR experimental setup.

    Binary-searches OpenJPEG's `quality_layers` (in `quality_mode="rates"`) against
    the resulting on-disk size. Always uses `irreversible=True` (9/7 wavelet, lossy)
    and `mct=1` (Multi-Component Transform — RGB → YCbCr in the wavelet domain),
    matching how Kakadu and other reference encoders treat RGB.
    """
    try:
        img = Image.open(input_path)
        # Read channel count BEFORE convert("RGB") so raw_size matches
        # get_compression_ratio's definition (which uses the source mode).
        source_channels = len(img.getbands())
        if img.mode != "RGB":
            img = img.convert("RGB")

        raw_size = img.width * img.height * source_channels
        target_size = max(1.0, raw_size / compression_ratio)

        # Search rate in [1, 1000]. Higher rate -> smaller file (more compression).
        # rate=1.0 (lower bound) is the least-compressed lossy 9/7 wavelet output.
        # With RAW-based CR the target_size is always smaller than raw_size, so
        # the search always converges into the lossy regime.
        lo, hi = 1.0, 1000.0
        best_rate = (lo + hi) / 2.0
        best_size = None
        best_diff = float("inf")
        max_iterations = 20
        tolerance = 0.05 * target_size

        for _ in range(max_iterations):
            mid = (lo + hi) / 2.0
            img.save(
                output_path,
                "JPEG2000",
                irreversible=True,
                quality_mode="rates",
                quality_layers=[mid],
                mct=1,
            )
            actual_size = output_path.stat().st_size
            diff = abs(actual_size - target_size)

            if diff < best_diff:
                best_diff = diff
                best_rate = mid
                best_size = actual_size

            if diff < tolerance:
                break

            if actual_size > target_size:
                lo = mid  # need more compression -> higher rate
            else:
                hi = mid

        # Re-save with best rate only if the last iteration wasn't the best,
        # otherwise the on-disk file already corresponds to best_rate.
        if best_size is None or output_path.stat().st_size != best_size:
            img.save(
                output_path,
                "JPEG2000",
                irreversible=True,
                quality_mode="rates",
                quality_layers=[best_rate],
                mct=1,
            )

        if best_diff > tolerance:
            # `best_size` is the size measured for `best_rate` during the
            # search; the file on disk corresponds to `best_rate` after the
            # re-save above. Use it directly instead of re-reading the file,
            # which would assume OpenJPEG re-encodes byte-deterministically.
            actual_size = best_size if best_size is not None else output_path.stat().st_size
            # best_rate pinned near `lo=1.0`: at OpenJPEG's lossy minimum,
            # the target CR is below the format's achievable floor.
            # best_rate pinned near `hi=1000.0`: target CR is above what the
            # search range can reach (file still too large at max rate).
            if best_rate < 1.5:
                hint = " [JP2 lossy floor — target CR unreachable, file too large]"
            elif best_rate > 999.0:
                hint = " [JP2 rate ceiling — target CR unreachable, file too small]"
            else:
                hint = ""
            print(
                f"Warning (JPEG2000): {input_path.name}: tolerance not reached "
                f"(target={target_size:.0f}B, actual={actual_size}B, "
                f"diff={best_diff:.0f}B, rate={best_rate:.3f}){hint}"
            )

        return True
    except Exception as e:
        print(f"Error (JPEG2000): {input_path.name}: {e}")
        return False


def compress_image_avif(input_path, output_path, compression_ratio):
    """Compress to AVIF format matching a target compression ratio.

    `compression_ratio` is interpreted as RAW-based CR (raw_pixel_bytes / file_bytes),
    matching JPEG2000 / DICOM / AVIF literature convention. Target file size is
    therefore `raw_size / compression_ratio`, which equals the corresponding JPEG
    file size for the matched-CR experimental setup.

    Binary-searches `quality` against the resulting on-disk size. Notes:
      * `quality=100` in pillow-avif-plugin activates an internal lossless path
        (bit-exact equal to source), so the search is capped at quality=99.
      * `subsampling="4:4:4"` matches JPEG's `subsampling=0`, keeping the chroma
        policy consistent across formats.
      * `speed=6` and `range="full"` are pinned to make the comparison
        reproducible across libavif versions.
    """
    try:
        img = Image.open(input_path)
        # Read channel count BEFORE convert("RGB") so raw_size matches
        # get_compression_ratio's definition (which uses the source mode).
        source_channels = len(img.getbands())
        if img.mode != "RGB":
            img = img.convert("RGB")

        raw_size = img.width * img.height * source_channels
        target_size = max(1.0, raw_size / compression_ratio)

        avif_kwargs = {
            "subsampling": "4:4:4",
            "speed": 6,
            "range": "full",
        }

        # Binary search for the quality that gives closest CR.
        # hi=99 (not 100) because quality=100 in pillow-avif-plugin is lossless;
        # we want every comparison point to be genuinely lossy.
        lo, hi = 1, 99
        best_quality = (lo + hi) // 2
        best_size = None
        best_diff = float("inf")
        max_iterations = 20  # Limit iterations for efficiency
        tolerance = 0.05 * target_size  # Stop if within 5% of target

        iterations = 0
        while lo <= hi and iterations < max_iterations:
            mid = (lo + hi) // 2
            img.save(output_path, "AVIF", quality=mid, **avif_kwargs)
            actual_size = output_path.stat().st_size
            diff = abs(actual_size - target_size)

            if diff < best_diff:
                best_diff = diff
                best_quality = mid
                best_size = actual_size

            # Early stopping if close enough
            if diff < tolerance:
                break

            if actual_size > target_size:
                hi = mid - 1  # need more compression -> lower quality
            else:
                lo = mid + 1  # can afford less compression -> higher quality

            iterations += 1

        # Note: best_quality is already bounded to [lo=1, hi=99] by the search,
        # so no extra min(..., 99) clamp is needed here.
        if best_size is None or output_path.stat().st_size != best_size:
            img.save(output_path, "AVIF", quality=best_quality, **avif_kwargs)

        if best_diff > tolerance:
            # Use the measured `best_size` for `best_quality` rather than
            # re-reading the file (which assumes deterministic re-encoding).
            actual_size = best_size if best_size is not None else output_path.stat().st_size
            print(
                f"Warning (AVIF): {input_path.name}: tolerance not reached "
                f"(target={target_size:.0f}B, actual={actual_size}B, "
                f"diff={best_diff:.0f}B, quality={best_quality})"
            )

        return True
    except Exception as e:
        print(f"Error (AVIF): {input_path.name}: {e}")
        return False


def get_extension(fmt):
    """Get file extension for format."""
    extensions = {
        "jpeg": ".jpg",
        "jpeg2000": ".jp2",
        "avif": ".avif",
    }
    return extensions.get(fmt, ".jpg")


def get_compression_ratio(original_path, compressed_path):
    """Calculate compression ratio = raw_size / compressed_size.

    Follows the convention used in JPEG2000 reference (ISO 15444), DICOM/medical
    imaging literature, and AVIF/AV1 benchmarks: the numerator is the size of
    the uncompressed pixel buffer (H x W x C x bytes_per_sample), NOT the size
    of an already-compressed reference file (PNG). This keeps CR independent of
    the source-file encoder's efficiency and guarantees CR > 1 for genuinely
    lossy compression.

    Number of channels is read from the source image (1 for mode L, 3 for RGB,
    4 for RGBA, …) rather than hardcoded — ARCADE mixes mode-L and grayscale-as-RGB.
    Assumes 8-bit samples (1 byte/sample).
    """
    with Image.open(original_path) as img:
        w, h = img.size
        c = len(img.getbands())
        mode = img.mode
    # raw_size assumes 8-bit samples (1 byte/sample). Assert it explicitly:
    # a 16-bit source (mode I;16, I, F) would silently make CR 2x too low.
    # Kept consistent with measure_quality.calculate_metrics.
    if mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        raise ValueError(
            f"{original_path.name}: non-8-bit source mode '{mode}'. "
            f"CR formula assumes 8-bit samples; add a bits/8 factor "
            f"(here and in measure_quality.calculate_metrics) and "
            f"recompress before measuring."
        )
    raw_size = w * h * c
    compressed_size = compressed_path.stat().st_size
    if compressed_size <= 0:
        return float("nan")
    return raw_size / compressed_size


def compress_dataset_jpeg(task, split, quality_levels, force=False):
    """Compress dataset to JPEG and return per-image compression ratios.

    Args:
        force: If True, re-compress even if CR map exists.

    The CR map is persisted to disk AFTER EACH quality level, so a crash
    mid-loop (e.g. at Q=70 when 8/13 done) keeps the work for q-levels that
    already finished. A later run with the same `quality_levels` will skip
    those entries unless `force=True`.
    """
    source_dir = config.get_data_path(task, split, quality=None)
    if not source_dir.exists():
        print(f"Error: {source_dir} not found")
        return {}

    image_files = sorted(list(source_dir.glob("*.png")))
    print(f"\n{task}/{split}: {len(image_files)} images")

    cr_map_path = get_cr_map_path(task, split)
    cr_map_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing CR map if present; only re-do quality levels that are
    # missing or incomplete (count mismatch).
    cr_map = {}
    if cr_map_path.exists() and not force:
        print(f"Loading existing CR map from {cr_map_path}")
        with open(cr_map_path, 'r') as f:
            cr_map = json.load(f)

    def _is_complete(q):
        return str(q) in cr_map and len(cr_map[str(q)]) == len(image_files)

    todo_q = [q for q in quality_levels if not _is_complete(q)]
    if not todo_q:
        print(f"CR map complete for quality levels {quality_levels}")
        return cr_map

    print(f"Format: JPEG (reference); compressing Q levels: {todo_q}")

    for quality in todo_q:
        output_dir = config.get_data_path(task, split, quality=quality, format="jpeg")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Preserve any partial results for this quality level (e.g. a previous
        # run that crashed after some images) instead of resetting to {}.
        # This gives per-image skip: a level that failed on one image no
        # longer re-compresses every image from scratch (mirrors the
        # per-file skip in compress_dataset_matched).
        cr_map.setdefault(str(quality), {})

        for img_path in tqdm(image_files, desc=f"JPEG Q={quality}"):
            output_path = output_dir / f"{img_path.stem}.jpg"
            # Skip images that already have a CR entry and a file on disk,
            # unless forcing.
            if (not force
                    and img_path.stem in cr_map[str(quality)]
                    and output_path.exists()):
                continue
            success = compress_image_jpeg(img_path, output_path, quality)
            if success:
                cr_map[str(quality)][img_path.stem] = get_compression_ratio(
                    img_path, output_path
                )

        # Persist after every quality level so a crash doesn't lose progress.
        with open(cr_map_path, 'w') as f:
            json.dump(cr_map, f)
    print(f"Saved CR map to {cr_map_path}")
    return cr_map


def compress_dataset_matched(task, split, quality_levels, fmt, cr_map, force=False):
    """Compress dataset to JPEG2000 or AVIF matching JPEG compression ratios.

    Args:
        force: If True, re-compress even if output files exist
    """
    source_dir = config.get_data_path(task, split, quality=None)
    if not source_dir.exists():
        print(f"Error: {source_dir} not found")
        return

    image_files = sorted(list(source_dir.glob("*.png")))
    print(f"\n{task}/{split}: {len(image_files)} images")
    print(f"Format: {fmt.upper()} (matched CR from JPEG)")

    extension = get_extension(fmt)
    compress_fn = (
        compress_image_jpeg2000 if fmt == "jpeg2000" else compress_image_avif
    )

    for quality in quality_levels:
        output_dir = config.get_data_path(task, split, quality=quality, format=fmt)
        output_dir.mkdir(parents=True, exist_ok=True)

        quality_cr = cr_map.get(str(quality), {})
        if not quality_cr:
            print(f"Warning: no CR data for Q={quality}, skipping")
            continue

        for img_path in tqdm(image_files, desc=f"{fmt.upper()} Q={quality} (matched CR)"):
            stem = img_path.stem
            output_path = output_dir / f"{stem}{extension}"

            # Skip if exists and not forcing
            if not force and output_path.exists():
                continue

            cr = quality_cr.get(stem)
            if cr is None:
                print(f"Warning: no JPEG CR for {stem} at Q={quality}, skipping")
                continue

            compress_fn(img_path, output_path, cr)


def main():
    parser = argparse.ArgumentParser(
        description="Compress images to JPEG, JPEG2000, or AVIF (matched CR)"
    )
    parser.add_argument("--task", default="syntax",
                       choices=["syntax", "all"],
                       help="Task to compress (default: syntax)")
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all"
    )
    parser.add_argument(
        "--format",
        choices=["jpeg", "jpeg2000", "avif", "all"],
        default="all",
        help="Compression format (JPEG is always run first as reference)",
    )
    parser.add_argument("--mvp", action="store_true", help="Use MVP quality levels")
    parser.add_argument("--force", action="store_true",
                       help="Re-compress even if files exist")
    args = parser.parse_args()

    quality_levels = config.QUALITY_LEVELS_MVP if args.mvp else config.QUALITY_LEVELS
    tasks = config.TASKS if args.task == "all" else [args.task]
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    formats = config.COMPRESSION_FORMATS if args.format == "all" else [args.format]

    print("=" * 80)
    print("ARCADE Dataset Compression (Matched Compression Ratio)")
    print("=" * 80)
    print(f"Formats: {formats}")
    print(f"Quality levels: {quality_levels}")
    print(f"Tasks: {tasks}")
    print(f"Splits: {splits}")
    print("=" * 80)

    for task in tasks:
        for split in splits:
            # Step 1: Compress JPEG first to get reference CRs (uses cached CR map if exists)
            if "jpeg" in formats or "jpeg2000" in formats or "avif" in formats:
                cr_map = compress_dataset_jpeg(task, split, quality_levels, force=args.force)

            # Step 2: Compress other formats using matched CRs
            for fmt in formats:
                if fmt == "jpeg":
                    continue  # already done
                compress_dataset_matched(task, split, quality_levels, fmt, cr_map, force=args.force)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
