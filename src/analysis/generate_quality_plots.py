"""Generate publication-quality PDF plots for compression metrics (PSNR, SSIM, Compression Ratio).

Reads quality_*.csv files from results/metrics/ and generates PDF plots.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# Use PDF backend
matplotlib.use('Agg')

# Set publication-quality parameters
rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def load_quality_metrics():
    """Load all quality metrics CSV files and aggregate them."""
    metrics_dir = config.RESULTS_ROOT / "metrics"
    all_data = []

    # Load all quality_<task>_*.csv files. The task prefixes come from
    # config.TASKS so this stays in sync if the task list changes.
    csv_files = []
    for task in config.TASKS:
        csv_files.extend(sorted(metrics_dir.glob(f"quality_{task}_*.csv")))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            print(f"[WARNING] Skipping unreadable CSV {csv_file.name}: {exc}")
            continue
        if df.empty:
            print(f"[WARNING] Skipping empty CSV {csv_file.name}")
            continue
        all_data.append(df)

    if not all_data:
        print("[ERROR] No usable quality_*.csv files found in results/metrics/")
        return None, None

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    if combined.empty:
        print("[ERROR] All quality_*.csv files were empty")
        return None, None

    # Aggregate by format and quality (mean/std across train/val/test and all images)
    aggregated = combined.groupby(['format', 'quality']).agg(
        psnr=('psnr', 'mean'),
        psnr_std=('psnr', 'std'),
        ssim=('ssim', 'mean'),
        ssim_std=('ssim', 'std'),
        compression_ratio=('compression_ratio', 'mean'),
        compression_ratio_std=('compression_ratio', 'std'),
    ).reset_index()

    # Sort by quality descending
    aggregated = aggregated.sort_values('quality', ascending=False)

    return combined, aggregated


def plot_psnr(aggregated, output_dir):
    """Generate PSNR vs Quality plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    formats = ['jpeg', 'jpeg2000', 'avif']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for fmt, color, marker in zip(formats, colors, markers):
        data = aggregated[aggregated['format'] == fmt].sort_values('quality')
        if data.empty:
            # Skip formats with no data so they don't add a phantom legend entry.
            continue
        ax.errorbar(data['quality'], data['psnr'], yerr=data['psnr_std'],
                    marker=marker, linewidth=2, markersize=8, capsize=3,
                    label=fmt.upper(), color=color)

    ax.set_xlabel('Jakość kompresji (Q)')
    ax.set_ylabel('PSNR [dB]')
    ax.set_title('PSNR w funkcji poziomu jakości kompresji')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    output_path = output_dir / 'quality_psnr.pdf'
    plt.savefig(output_path, format='pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_ssim(aggregated, output_dir):
    """Generate SSIM vs Quality plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    formats = ['jpeg', 'jpeg2000', 'avif']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for fmt, color, marker in zip(formats, colors, markers):
        data = aggregated[aggregated['format'] == fmt].sort_values('quality')
        if data.empty:
            # Skip formats with no data so they don't add a phantom legend entry.
            continue
        ax.errorbar(data['quality'], data['ssim'], yerr=data['ssim_std'],
                    marker=marker, linewidth=2, markersize=8, capsize=3,
                    label=fmt.upper(), color=color)

    ax.set_xlabel('Jakość kompresji (Q)')
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM w funkcji poziomu jakości kompresji')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    output_path = output_dir / 'quality_ssim.pdf'
    plt.savefig(output_path, format='pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_compression_ratio(aggregated, output_dir):
    """Generate Compression Ratio vs Quality plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    formats = ['jpeg', 'jpeg2000', 'avif']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for fmt, color, marker in zip(formats, colors, markers):
        data = aggregated[aggregated['format'] == fmt].sort_values('quality')
        if data.empty:
            # Skip formats with no data so they don't add a phantom legend entry.
            continue
        ratio = data['compression_ratio']
        std = data['compression_ratio_std'].fillna(0.0)
        # On a log y-axis a symmetric error bar could reach <= 0; clip the lower
        # whisker so it stays strictly positive.
        lower = (ratio - std).clip(lower=ratio * 1e-3)
        yerr = [ratio - lower, std]
        ax.errorbar(data['quality'], ratio, yerr=yerr, marker=marker,
                    linewidth=2, markersize=8, capsize=3,
                    label=fmt.upper(), color=color)

    ax.set_xlabel('Jakość kompresji (Q)')
    ax.set_ylabel('Współczynnik kompresji')
    ax.set_title('Współczynnik kompresji w funkcji poziomu jakości (skala logarytmiczna)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    output_path = output_dir / 'compression_ratio.pdf'
    plt.savefig(output_path, format='pdf')
    print(f"Saved: {output_path}")
    plt.close()


def load_baseline_result(model_name, task="syntax"):
    """Load the baseline (uncompressed PNG) result for a model.

    The baseline file is a single-row CSV produced by run_baseline.py and is
    the upper-bound reference for Experiment A. It is optional: if the file is
    missing, empty or unreadable, this returns None silently.
    """
    baseline_path = (
        config.RESULTS_ROOT / "experiment_a"
        / f"{model_name}_arcade_{task}_baseline_results.csv"
    )
    if not baseline_path.exists():
        return None
    try:
        df = pd.read_csv(baseline_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"[WARNING] Skipping unreadable baseline CSV {baseline_path.name}: {exc}")
        return None
    if df.empty:
        return None
    return df


def plot_experiment_a_accuracy(output_dir, model_name="resnet50"):
    """Generate Experiment A metric plot (mAP leading, F1-Macro auxiliary).

    mAP (test_map) is threshold-free and the primary metric for strongly
    imbalanced multi-label classification; F1-Macro at a fixed 0.5 threshold is
    kept as an auxiliary series. A baseline (uncompressed PNG) reference line is
    drawn for each metric when the baseline result is available.
    """
    exp_a_path = config.RESULTS_ROOT / "experiment_a"

    # Load all experiment A results
    all_data = []
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        csv_file = exp_a_path / f"{model_name}_arcade_syntax_{fmt}_results.csv"
        if not csv_file.exists():
            continue
        try:
            df = pd.read_csv(csv_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            print(f"[WARNING] Skipping unreadable CSV {csv_file.name}: {exc}")
            continue
        if df.empty:
            print(f"[WARNING] Skipping empty CSV {csv_file.name}")
            continue
        all_data.append(df)

    if not all_data:
        print(f"[WARNING] No experiment A results found for {model_name}")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Baseline reference (uncompressed PNG) — optional, loaded separately so it
    # never mixes into the per-format data.
    baseline_df = load_baseline_result(model_name, task="syntax")
    b_map = b_f1 = None
    if baseline_df is not None:
        b_row = baseline_df.iloc[0]
        if 'test_map' in baseline_df.columns and pd.notna(b_row['test_map']):
            b_map = b_row['test_map'] * 100
        if 'test_f1_macro' in baseline_df.columns and pd.notna(b_row['test_f1_macro']):
            b_f1 = b_row['test_f1_macro'] * 100

    # Two panels: mAP (primary) on the left, F1-Macro (auxiliary) on the right.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    formats = ['jpeg', 'jpeg2000', 'avif']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    panels = [
        (axes[0], 'test_map', 'mAP', 'mAP [%]', b_map),
        (axes[1], 'test_f1_macro', 'F1-Macro', 'F1-Macro [%]', b_f1),
    ]

    for ax, metric_col, metric_label, ylabel, baseline_value in panels:
        for fmt, color, marker in zip(formats, colors, markers):
            data = combined[combined['format'] == fmt].sort_values('train_quality')
            if data.empty:
                # Skip formats with no data so they don't add a phantom legend entry.
                continue
            ax.plot(data['train_quality'], data[metric_col] * 100,
                    marker=marker, linewidth=2, markersize=8,
                    label=fmt.upper(), color=color)

        # Baseline reference line for this metric.
        if baseline_value is not None:
            ax.axhline(baseline_value, color='#444444', linestyle='--', linewidth=1.5,
                       label='Baseline (uncompressed)')

        ax.set_xlabel('Jakość kompresji (Q)')
        ax.set_ylabel(ylabel)
        ax.set_title(metric_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    fig.suptitle('Eksperyment A: Trening na skompresowanych, test na oryginałach',
                 fontsize=14)

    plt.tight_layout()
    output_path = output_dir / f'exp_a_accuracy_{model_name}.pdf'
    plt.savefig(output_path, format='pdf')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = config.PLOTS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING QUALITY PLOTS (PDF)")
    print("=" * 80)

    # Load quality metrics
    print("\nLoading quality metrics...")
    raw_data, aggregated = load_quality_metrics()

    if aggregated is None:
        return

    print(f"Loaded {len(raw_data)} quality measurements")
    print(f"Aggregated to {len(aggregated)} data points (format × quality)")

    # Generate plots
    print("\nGenerating PDF plots...")

    print("\n1. PSNR plot...")
    plot_psnr(aggregated, output_dir)

    print("2. SSIM plot...")
    plot_ssim(aggregated, output_dir)

    print("3. Compression Ratio plot...")
    plot_compression_ratio(aggregated, output_dir)

    # One accuracy plot per trained model (silently skips if CSV missing).
    for model in config.SUPPORTED_MODELS:
        print(f"\n4. Experiment A Accuracy plot — {model}...")
        plot_experiment_a_accuracy(output_dir, model_name=model)

    print("\n" + "=" * 80)
    print("ALL PDF PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
