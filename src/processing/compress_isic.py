"""
Compression script for ISIC 2019 dataset.

Compresses images using JPEG, JPEG2000, and AVIF at various quality levels.
Uses matched compression ratio approach for fair comparison between formats.
"""
import sys
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.processing.compress_images import compress_image_jpeg, compress_image_jpeg2000, compress_image_avif


def compress_isic_dataset(
    quality_levels: list,
    formats: list,
    splits: list = ['train', 'val', 'test'],
    input_root: Path = None,
    output_root: Path = None,
    matched_cr: bool = True
):
    """
    Compress ISIC 2019 dataset.

    Args:
        quality_levels: List of quality levels (e.g., [100, 85, 70, 50, 30, 10])
        formats: List of formats (e.g., ['jpeg', 'jpeg2000', 'avif'])
        splits: Dataset splits to compress
        input_root: Input directory (default: dataset/isic_2019)
        output_root: Output directory (default: dataset/compressed_isic)
        matched_cr: Use matched compression ratio approach
    """
    if input_root is None:
        input_root = config.DATASET_ROOT / 'isic_2019'
    if output_root is None:
        output_root = config.DATASET_ROOT / 'compressed_isic'

    input_root = Path(input_root)
    output_root = Path(output_root)

    print(f"Compressing ISIC 2019 dataset")
    print(f"  Input: {input_root}")
    print(f"  Output: {output_root}")
    print(f"  Quality levels: {quality_levels}")
    print(f"  Formats: {formats}")
    print(f"  Matched CR: {matched_cr}")

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        split_dir = input_root / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping")
            continue

        # Get all images
        image_files = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png'))
        print(f"Found {len(image_files)} images")

        if matched_cr:
            # Matched compression ratio approach
            for quality in quality_levels:
                print(f"\nQuality Q={quality}")

                for fmt in formats:
                    fmt_dir = output_root / fmt / f'q{quality}' / split
                    fmt_dir.mkdir(parents=True, exist_ok=True)

                    print(f"  {fmt.upper()}...", end='', flush=True)
                    success = 0

                    for img_path in tqdm(image_files, desc=f"{fmt} q{quality}"):
                        out_path = fmt_dir / f"{img_path.stem}.jpg" if fmt == 'jpeg' else \
                                   fmt_dir / f"{img_path.stem}.jp2" if fmt == 'jpeg2000' else \
                                   fmt_dir / f"{img_path.stem}.avif"

                        if out_path.exists():
                            success += 1
                            continue

                        try:
                            if fmt == 'jpeg':
                                compress_image_jpeg(img_path, out_path, quality)
                            elif fmt == 'jpeg2000':
                                # Calculate target CR from JPEG
                                jpeg_path = output_root / 'jpeg' / f'q{quality}' / split / f"{img_path.stem}.jpg"
                                if jpeg_path.exists():
                                    target_cr = img_path.stat().st_size / jpeg_path.stat().st_size
                                else:
                                    target_cr = 10  # Default
                                compress_image_jpeg2000(img_path, out_path, target_cr)
                            elif fmt == 'avif':
                                # Calculate target CR from JPEG
                                jpeg_path = output_root / 'jpeg' / f'q{quality}' / split / f"{img_path.stem}.jpg"
                                if jpeg_path.exists():
                                    target_cr = img_path.stat().st_size / jpeg_path.stat().st_size
                                else:
                                    target_cr = 10  # Default
                                compress_image_avif(img_path, out_path, target_cr)

                            success += 1
                        except Exception as e:
                            print(f"\n    Error: {img_path.name}: {e}")

                    print(f" Done ({success}/{len(image_files)})")

    print(f"\n[OK] Compression complete!")
    print(f"Output saved to: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Compress ISIC 2019 dataset")
    parser.add_argument(
        "--format",
        type=str,
        nargs='+',
        default=['jpeg', 'jpeg2000', 'avif'],
        choices=['jpeg', 'jpeg2000', 'avif', 'all'],
        help="Compression format(s)"
    )
    parser.add_argument(
        "--quality-levels",
        type=int,
        nargs='+',
        default=config.QUALITY_LEVELS_MVP,
        help="Quality levels"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help="Dataset splits to compress"
    )
    parser.add_argument(
        "--mvp",
        action="store_true",
        help="Use MVP quality levels (100, 85, 70, 50, 30, 10)"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Input directory (default: dataset/isic_2019)"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output directory (default: dataset/compressed_isic)"
    )

    args = parser.parse_args()

    # Handle 'all' format
    formats = args.format
    if 'all' in formats:
        formats = ['jpeg', 'jpeg2000', 'avif']

    # Handle MVP flag
    quality_levels = config.QUALITY_LEVELS_MVP if args.mvp else args.quality_levels

    compress_isic_dataset(
        quality_levels=quality_levels,
        formats=formats,
        splits=args.splits,
        input_root=Path(args.input_root) if args.input_root else None,
        output_root=Path(args.output_root) if args.output_root else None
    )


if __name__ == "__main__":
    main()
