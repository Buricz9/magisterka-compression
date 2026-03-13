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
        return None

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


def plot_experiment_a_accuracy(output_dir):
    """Generate Experiment A Accuracy plot (convert PNG to PDF)."""
    exp_a_path = config.RESULTS_ROOT / "experiment_a"

    # Load all experiment A results
    all_data = []
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        csv_file = exp_a_path / f"resnet50_arcade_syntax_{fmt}_results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            all_data.append(df)

    if not all_data:
        print("[WARNING] No experiment A results found")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    formats = ['jpeg', 'jpeg2000', 'avif']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for fmt, color, marker in zip(formats, colors, markers):
        data = combined[combined['format'] == fmt].sort_values('train_quality')
        ax.plot(data['train_quality'], data['test_accuracy'] * 100,
                marker=marker, linewidth=2, markersize=8,
                label=fmt.upper(), color=color)

    ax.set_xlabel('Jakość kompresji (Q)')
    ax.set_ylabel('Dokładność testowa [%]')
    ax.set_title('Eksperyment A: Trening na skompresowanych, test na oryginałach')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    output_path = output_dir / 'exp_a_accuracy.pdf'
    plt.savefig(output_path, format='pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_experiment_b_accuracy(output_dir):
    """Generate Experiment B Accuracy plot (from article table data)."""
    # Data from Table B in article (no CSV available)
    exp_b_data = {
        'jpeg': {
            100: 15.00, 85: 18.00, 70: 19.33, 50: 20.00, 30: 19.33, 10: 10.67
        },
        'jpeg2000': {
            100: 17.67, 85: 16.33, 70: 15.67, 50: 16.00, 30: 13.67, 10: 12.00
        },
        'avif': {
            100: 19.33, 85: 15.67, 70: 18.33, 50: 19.33, 30: 14.67, 10: 14.00
        }
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    formats = ['jpeg', 'jpeg2000', 'avif']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for fmt, color, marker in zip(formats, colors, markers):
        qualities = sorted(exp_b_data[fmt].keys(), reverse=True)
        accuracies = [exp_b_data[fmt][q] for q in qualities]
        ax.plot(qualities, accuracies, marker=marker, linewidth=2, markersize=8,
                label=fmt.upper(), color=color)

    ax.set_xlabel('Jakość kompresji (Q)')
    ax.set_ylabel('Dokładność testowa [%]')
    ax.set_title('Eksperyment B: Trening na oryginałach, test na skompresowanych')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    output_path = output_dir / 'exp_b_accuracy.pdf'
    plt.savefig(output_path, format='pdf')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_ab(output_dir):
    """Generate combined plot comparing Experiment A and B."""
    exp_a_path = config.RESULTS_ROOT / "experiment_a"

    # Load Experiment A data
    exp_a_data = {}
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        csv_file = exp_a_path / f"resnet50_arcade_syntax_{fmt}_results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            exp_a_data[fmt] = dict(zip(df['train_quality'], df['test_accuracy'] * 100))

    # Experiment B data (from article)
    exp_b_data = {
        'jpeg': {100: 15.00, 85: 18.00, 70: 19.33, 50: 20.00, 30: 19.33, 10: 10.67},
        'jpeg2000': {100: 17.67, 85: 16.33, 70: 15.67, 50: 16.00, 30: 13.67, 10: 12.00},
        'avif': {100: 19.33, 85: 15.67, 70: 18.33, 50: 19.33, 30: 14.67, 10: 14.00}
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    formats = ['jpeg', 'jpeg2000', 'avif']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # Experiment A
    ax = axes[0]
    for fmt, color, marker in zip(formats, colors, markers):
        if fmt in exp_a_data:
            qualities = sorted(exp_a_data[fmt].keys(), reverse=True)
            accuracies = [exp_a_data[fmt][q] for q in qualities]
            ax.plot(qualities, accuracies, marker=marker, linewidth=2, markersize=8,
                    label=fmt.upper(), color=color)

    ax.set_xlabel('Jakość kompresji (Q)')
    ax.set_ylabel('Dokładność testowa [%]')
    ax.set_title('Eksperyment A: Trening na skompresowanych')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Experiment B
    ax = axes[1]
    for fmt, color, marker in zip(formats, colors, markers):
        qualities = sorted(exp_b_data[fmt].keys(), reverse=True)
        accuracies = [exp_b_data[fmt][q] for q in qualities]
        ax.plot(qualities, accuracies, marker=marker, linewidth=2, markersize=8,
                label=fmt.upper(), color=color)

    ax.set_xlabel('Jakość kompresji (Q)')
    ax.set_ylabel('Dokładność testowa [%]')
    ax.set_title('Eksperyment B: Test na skompresowanych')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.suptitle('Porównanie Eksperymentu A i B', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'combined_ab.pdf'
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

    print("4. Experiment A Accuracy plot...")
    plot_experiment_a_accuracy(output_dir)

    print("5. Experiment B Accuracy plot...")
    plot_experiment_b_accuracy(output_dir)

    print("6. Combined A+B plot...")
    plot_combined_ab(output_dir)

    print("\n" + "=" * 80)
    print("ALL PDF PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
