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

    # Load all quality_*.csv files
    for csv_file in metrics_dir.glob("quality_syntax_*.csv"):
        df = pd.read_csv(csv_file)
        all_data.append(df)

    if not all_data:
        print("[ERROR] No quality_*.csv files found in results/metrics/")
        return None, None

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    # Aggregate by format and quality (mean across train/val/test and all images)
    aggregated = combined.groupby(['format', 'quality']).agg({
        'psnr': 'mean',
        'ssim': 'mean',
        'compression_ratio': 'mean'
    }).reset_index()

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
        ax.plot(data['quality'], data['psnr'], marker=marker, linewidth=2,
                markersize=8, label=fmt.upper(), color=color)

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
        ax.plot(data['quality'], data['ssim'], marker=marker, linewidth=2,
                markersize=8, label=fmt.upper(), color=color)

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
        ax.plot(data['quality'], data['compression_ratio'], marker=marker,
                linewidth=2, markersize=8, label=fmt.upper(), color=color)

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


def plot_experiment_a_accuracy(output_dir, model_name="resnet50"):
    """Generate Experiment A Accuracy plot (convert PNG to PDF)."""
    exp_a_path = config.RESULTS_ROOT / "experiment_a"

    # Load all experiment A results
    all_data = []
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        csv_file = exp_a_path / f"{model_name}_arcade_syntax_{fmt}_results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            all_data.append(df)

    if not all_data:
        print(f"[WARNING] No experiment A results found for {model_name}")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    formats = ['jpeg', 'jpeg2000', 'avif']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for fmt, color, marker in zip(formats, colors, markers):
        data = combined[combined['format'] == fmt].sort_values('train_quality')
        ax.plot(data['train_quality'], data['test_hamming_accuracy'] * 100,
                marker=marker, linewidth=2, markersize=8,
                label=fmt.upper(), color=color)

    ax.set_xlabel('Jakość kompresji (Q)')
    ax.set_ylabel('Dokładność testowa [%]')
    ax.set_title('Eksperyment A: Trening na skompresowanych, test na oryginałach')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

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
