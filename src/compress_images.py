"""Compress ARCADE dataset images to various formats (JPEG, JPEG2000, AVIF).

Matched compression ratio approach:
- JPEG is compressed first using quality factor Q.
- For each JPEG image, the achieved compression ratio (CR) is measured.
- JPEG2000 and AVIF are then compressed to match the same CR per image.
"""
import sys
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


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
    """Compress to JPEG2000 format using target compression ratio.

    Converts compression ratio to bits per pixel (bpp) for quality_layers.
    For RGB images: bpp = 24 / compression_ratio (where 24 = 8 bits * 3 channels)

    Args:
        input_path: Path to input image
        output_path: Path for output JPEG2000 file
        compression_ratio: Target compression ratio (original_size / compressed_size)
    """
    try:
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if compression_ratio <= 1.0:
            # CR <= 1 means no compression or file grew; use lossless
            img.save(output_path, "JPEG2000")
        else:
            # Convert compression ratio to bits per pixel (bpp)
            # For RGB: 24 bpp uncompressed (8 bits * 3 channels)
            # bpp = 24 / CR
            bpp = 24.0 / compression_ratio
            img.save(
                output_path,
                "JPEG2000",
                irreversible=True,
                quality_mode="rates",
                quality_layers=[bpp],
            )
        return True
    except Exception as e:
        print(f"Error (JPEG2000): {input_path.name}: {e}")
        return False


def compress_image_avif(input_path, output_path, compression_ratio):
    """Compress to AVIF format matching a target compression ratio.

    AVIF (via Pillow) only supports a quality parameter (0-100), so we use
    binary search to find the quality value that produces the closest CR.
    """
    try:
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        original_size = input_path.stat().st_size

        if compression_ratio <= 1.0:
            img.save(output_path, "AVIF", quality=100)
            return True

        target_size = original_size / compression_ratio

        # Binary search for the quality that gives closest CR
        # Limit iterations to prevent infinite loops
        lo, hi = 1, 100
        best_quality = 50
        best_diff = float("inf")
        max_iterations = 20  # Limit iterations for efficiency
        tolerance = 0.05 * target_size  # Stop if within 5% of target

        iterations = 0
        while lo <= hi and iterations < max_iterations:
            mid = (lo + hi) // 2
            img.save(output_path, "AVIF", quality=mid)
            actual_size = output_path.stat().st_size
            diff = abs(actual_size - target_size)

            if diff < best_diff:
                best_diff = diff
                best_quality = mid

            # Early stopping if close enough
            if diff < tolerance:
                break

            if actual_size > target_size:
                hi = mid - 1  # need more compression -> lower quality
            else:
                lo = mid + 1  # can afford less compression -> higher quality

            iterations += 1

        # Save with the best quality found
        img.save(output_path, "AVIF", quality=best_quality)
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
    """Calculate compression ratio = original_size / compressed_size."""
    original_size = original_path.stat().st_size
    compressed_size = compressed_path.stat().st_size
    if compressed_size == 0:
        return 0.0
    return original_size / compressed_size


def compress_dataset_jpeg(task, split, quality_levels):
    """Compress dataset to JPEG and return per-image compression ratios."""
    source_dir = config.get_data_path(task, split, quality=None)
    if not source_dir.exists():
        print(f"Error: {source_dir} not found")
        return {}

    image_files = sorted(list(source_dir.glob("*.png")))
    print(f"\n{task}/{split}: {len(image_files)} images")
    print("Format: JPEG (reference)")

    # {quality: {image_stem: compression_ratio}}
    cr_map = {}

    for quality in quality_levels:
        output_dir = config.get_data_path(task, split, quality=quality, format="jpeg")
        output_dir.mkdir(parents=True, exist_ok=True)
        cr_map[quality] = {}

        for img_path in tqdm(image_files, desc=f"JPEG Q={quality}"):
            output_path = output_dir / f"{img_path.stem}.jpg"
            success = compress_image_jpeg(img_path, output_path, quality)
            if success:
                cr_map[quality][img_path.stem] = get_compression_ratio(
                    img_path, output_path
                )

    return cr_map


def compress_dataset_matched(task, split, quality_levels, fmt, cr_map):
    """Compress dataset to JPEG2000 or AVIF matching JPEG compression ratios."""
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

        quality_cr = cr_map.get(quality, {})

        for img_path in tqdm(image_files, desc=f"{fmt.upper()} Q={quality} (matched CR)"):
            stem = img_path.stem
            cr = quality_cr.get(stem)
            if cr is None:
                print(f"Warning: no JPEG CR for {stem} at Q={quality}, skipping")
                continue

            output_path = output_dir / f"{stem}{extension}"
            compress_fn(img_path, output_path, cr)


def main():
    parser = argparse.ArgumentParser(
        description="Compress images to JPEG, JPEG2000, or AVIF (matched CR)"
    )
    parser.add_argument("--task", choices=["syntax", "stenosis", "all"], default="all")
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
            # Step 1: Always compress JPEG first to get reference CRs
            if "jpeg" in formats or "jpeg2000" in formats or "avif" in formats:
                cr_map = compress_dataset_jpeg(task, split, quality_levels)

            # Step 2: Compress other formats using matched CRs
            for fmt in formats:
                if fmt == "jpeg":
                    continue  # already done
                compress_dataset_matched(task, split, quality_levels, fmt, cr_map)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
