"""Measure image quality metrics (PSNR, SSIM) for compressed images."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
try:
    import pillow_avif  # noqa: F401  # registers AVIF plugin; matches encoder side
except ImportError:
    pass
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def load_image_as_array(image_path):
    """Load image and convert to numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def calculate_metrics(baseline_path, compressed_path):
    """Calculate quality metrics between baseline and compressed image.

    Note: Images must have the same dimensions for accurate PSNR/SSIM.
    Resize would change pixel values and invalidate the metrics.
    """
    try:
        baseline = load_image_as_array(baseline_path)
        compressed = load_image_as_array(compressed_path)

        # Check dimensions - do NOT resize as it changes pixel values
        # and invalidates PSNR/SSIM measurements
        if baseline.shape != compressed.shape:
            print(f"Warning: Dimension mismatch for {baseline_path.name}: "
                  f"{baseline.shape} vs {compressed.shape}. Skipping.")
            return None

        psnr_value = psnr(baseline, compressed, data_range=255)
        # PSNR=inf means MSE=0, i.e. the "compressed" image is bit-exact
        # identical to the baseline. For lossy formats (JPEG/JP2/AVIF) this
        # should NEVER happen and indicates a real bug (e.g. an encoder
        # silently falling back to a lossless path). Do NOT mask it with a
        # fake dB value — that would poison groupby().mean() and hide the
        # regression. Store NaN instead: pandas mean() skips NaN by default,
        # so the average stays honest and the missing point is visible in
        # plots. The printed warning surfaces the offending file.
        if not np.isfinite(psnr_value):
            print(f"Warning: PSNR=inf (MSE=0, bit-exact identical) for "
                  f"{compressed_path.name} — lossy compression produced a "
                  f"lossless result. Recording NaN; investigate the encoder.")
            psnr_value = np.nan
        ssim_value = ssim(baseline, compressed, data_range=255, channel_axis=2, win_size=7)

        # RAW-based CR: raw_pixel_bytes / compressed_file_bytes.
        # Channel count is read from the SOURCE PNG (not the RGB-converted array)
        # so the value matches compress_images.get_compression_ratio.
        with Image.open(baseline_path) as src_img:
            source_channels = len(src_img.getbands())
            source_mode = src_img.mode
        # raw_size assumes 8-bit samples (1 byte/sample). ARCADE PNGs are
        # 8-bit, but assert it explicitly: a 16-bit source (mode I;16, I, F)
        # would silently make CR 2x too low. If 16-bit images ever appear,
        # this raises a readable error instead of producing wrong numbers.
        # (Must stay consistent with compress_images.get_compression_ratio.)
        if source_mode in ("I", "I;16", "I;16B", "I;16L", "F"):
            raise ValueError(
                f"{baseline_path.name}: non-8-bit source mode '{source_mode}'. "
                f"CR formula assumes 8-bit samples; add a bits/8 factor "
                f"(here and in compress_images.get_compression_ratio) and "
                f"recompress before measuring."
            )
        raw_size = baseline.shape[0] * baseline.shape[1] * source_channels
        compressed_size = compressed_path.stat().st_size
        compression_ratio = (
            raw_size / compressed_size if compressed_size > 0 else float("nan")
        )

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

    # Validate the baseline directory before globbing. Path.glob on a missing
    # directory returns an empty iterator silently, which would otherwise
    # write an empty (but valid-looking) quality CSV with no error.
    if not baseline_dir.is_dir():
        print(f"Error: baseline directory not found: {baseline_dir}. "
              f"Skipping {task}/{split}/{format}.")
        return

    baseline_images = sorted(list(baseline_dir.glob("*.png")))
    if not baseline_images:
        print(f"Error: no baseline .png images in {baseline_dir}. "
              f"Skipping {task}/{split}/{format}.")
        return

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
    parser.add_argument("--task", choices=["syntax", "all"], default="all")
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
