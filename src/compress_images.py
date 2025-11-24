"""Compress ARCADE dataset images to various formats (JPEG, JPEG2000, AVIF)."""
import sys
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def compress_image_jpeg(input_path, output_path, quality):
    """Compress to JPEG format."""
    try:
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=quality, subsampling=2, optimize=True)
        return True
    except Exception as e:
        print(f"Error (JPEG): {input_path.name}: {e}")
        return False


def compress_image_jpeg2000(input_path, output_path, quality):
    """Compress to JPEG2000 format."""
    try:
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # JPEG2000 quality is different - use quality_layers parameter
        # Lower quality = more compression
        quality_layers = [100 - quality]  # Invert: Q10 = 90% compression, Q100 = 0% compression
        img.save(output_path, "JPEG2000", quality_layers=quality_layers)
        return True
    except Exception as e:
        print(f"Error (JPEG2000): {input_path.name}: {e}")
        return False


def compress_image_avif(input_path, output_path, quality):
    """Compress to AVIF format."""
    try:
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # AVIF quality: 0-100 (higher = better quality)
        img.save(output_path, "AVIF", quality=quality)
        return True
    except Exception as e:
        print(f"Error (AVIF): {input_path.name}: {e}")
        print(f"Note: AVIF requires 'pillow-avif-plugin'. Install with: pip install pillow-avif-plugin")
        return False


def compress_image(input_path, output_path, quality, format='jpeg'):
    """Compress single image to specified format."""
    if format == 'jpeg':
        return compress_image_jpeg(input_path, output_path, quality)
    elif format == 'jpeg2000':
        return compress_image_jpeg2000(input_path, output_path, quality)
    elif format == 'avif':
        return compress_image_avif(input_path, output_path, quality)
    else:
        print(f"Unknown format: {format}")
        return False


def get_extension(format):
    """Get file extension for format."""
    extensions = {
        'jpeg': '.jpg',
        'jpeg2000': '.jp2',
        'avif': '.avif'
    }
    return extensions.get(format, '.jpg')


def compress_dataset(task, split, quality_levels, format='jpeg'):
    """Compress entire dataset split."""
    source_dir = config.get_data_path(task, split, quality=None)
    if not source_dir.exists():
        print(f"Error: {source_dir} not found")
        return

    image_files = sorted(list(source_dir.glob("*.png")))
    print(f"\n{task}/{split}: {len(image_files)} images")
    print(f"Format: {format.upper()}")

    for quality in quality_levels:
        output_dir = config.get_data_path(task, split, quality=quality, format=format)
        output_dir.mkdir(parents=True, exist_ok=True)

        extension = get_extension(format)
        for img_path in tqdm(image_files, desc=f"{format.upper()} Q={quality}"):
            output_path = output_dir / f"{img_path.stem}{extension}"
            compress_image(img_path, output_path, quality, format)


def main():
    parser = argparse.ArgumentParser(description="Compress images to JPEG, JPEG2000, or AVIF")
    parser.add_argument("--task", choices=["syntax", "stenosis", "all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--format", choices=["jpeg", "jpeg2000", "avif", "all"], default="jpeg",
                       help="Compression format")
    parser.add_argument("--mvp", action="store_true", help="Use MVP quality levels")
    args = parser.parse_args()

    quality_levels = config.QUALITY_LEVELS_MVP if args.mvp else config.QUALITY_LEVELS
    tasks = config.TASKS if args.task == "all" else [args.task]
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    formats = config.COMPRESSION_FORMATS if args.format == "all" else [args.format]

    print("="*80)
    print("ARCADE Dataset Compression")
    print("="*80)
    print(f"Formats: {formats}")
    print(f"Quality levels: {quality_levels}")
    print(f"Tasks: {tasks}")
    print(f"Splits: {splits}")
    print("="*80)

    for format in formats:
        print(f"\n>>> Processing format: {format.upper()}")
        for task in tasks:
            for split in splits:
                compress_dataset(task, split, quality_levels, format)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
