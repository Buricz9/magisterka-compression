"""Measure image quality metrics (PSNR, SSIM) for compressed images."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def load_image_as_array(image_path):
    """Load image and convert to numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def calculate_metrics(baseline_path, compressed_path):
    """Calculate quality metrics between baseline and compressed image."""
    try:
        baseline = load_image_as_array(baseline_path)
        compressed = load_image_as_array(compressed_path)

        if baseline.shape != compressed.shape:
            compressed_img = Image.open(compressed_path).resize(
                (baseline.shape[1], baseline.shape[0]), Image.LANCZOS
            )
            compressed = np.array(compressed_img)

        psnr_value = psnr(baseline, compressed, data_range=255)
        ssim_value = ssim(baseline, compressed, data_range=255, channel_axis=2, win_size=7)

        baseline_size = baseline_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        compression_ratio = baseline_size / compressed_size if compressed_size > 0 else 0

        return {
            "psnr": psnr_value,
            "ssim": ssim_value,
            "compression_ratio": compression_ratio,
        }
    except Exception as e:
        print(f"Error: {baseline_path.name}: {e}")
        return None


def measure_quality(task, split, quality_levels, format='jpeg'):
    """Measure quality metrics for all images in a split."""
    baseline_dir = config.get_data_path(task, split, quality=None)
    baseline_images = sorted(list(baseline_dir.glob("*.png")))

    # Get file extension for format
    extensions = {'jpeg': '.jpg', 'jpeg2000': '.jp2', 'avif': '.avif'}
    ext = extensions.get(format, '.jpg')

    results = []
    for quality in quality_levels:
        compressed_dir = config.get_data_path(task, split, quality=quality, format=format)

        for baseline_path in tqdm(baseline_images, desc=f"{format.upper()} Q={quality}"):
            compressed_path = compressed_dir / f"{baseline_path.stem}{ext}"
            if not compressed_path.exists():
                continue

            metrics = calculate_metrics(baseline_path, compressed_path)
            if metrics:
                results.append({
                    "task": task,
                    "split": split,
                    "format": format,
                    "quality": quality,
                    "image": baseline_path.name,
                    **metrics
                })

    # Save to CSV
    df = pd.DataFrame(results)
    output_file = config.RESULTS_ROOT / "metrics" / f"quality_{task}_{split}_{format}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["syntax", "stenosis", "all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--format", choices=["jpeg", "jpeg2000", "avif", "all"], default="jpeg")
    parser.add_argument("--mvp", action="store_true")
    args = parser.parse_args()

    quality_levels = config.QUALITY_LEVELS_MVP if args.mvp else config.QUALITY_LEVELS
    tasks = config.TASKS if args.task == "all" else [args.task]
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    formats = config.COMPRESSION_FORMATS if args.format == "all" else [args.format]

    for format in formats:
        print(f"\n>>> Measuring quality for format: {format.upper()}")
        for task in tasks:
            for split in splits:
                measure_quality(task, split, quality_levels, format)

    print("\nDone!")


if __name__ == "__main__":
    main()
