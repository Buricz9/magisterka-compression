"""Generate plots for the thesis article."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})

COLORS = {'jpeg': '#2196F3', 'jpeg2000': '#FF9800', 'avif': '#4CAF50'}
MARKERS = {'jpeg': 'o', 'jpeg2000': 's', 'avif': '^'}
LABELS = {'jpeg': 'JPEG', 'jpeg2000': 'JPEG2000', 'avif': 'AVIF'}
QUALITY_LEVELS = [100, 85, 70, 50, 30, 10]


def load_experiment_a():
    frames = []
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        path = RESULTS_ROOT / "experiment_a" / f"resnet50_syntax_{fmt}_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_experiment_b():
    frames = []
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        path = RESULTS_ROOT / "experiment_b" / f"resnet50_syntax_{fmt}_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_quality_metrics():
    frames = []
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        path = RESULTS_ROOT / "metrics" / f"quality_syntax_train_{fmt}.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_experiment_a_accuracy():
    df = load_experiment_a()
    fig, ax = plt.subplots()
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df[df['format'] == fmt].sort_values('train_quality')
        ax.plot(subset['train_quality'], subset['test_accuracy'] * 100,
                color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                linewidth=2, markersize=8)
    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('Accuracy [%]')
    ax.set_title('Eksperyment A: Trening na skompresowanych, test na oryginałach')
    ax.set_xticks(QUALITY_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'exp_a_accuracy.png')
    fig.savefig(PLOTS_DIR / 'exp_a_accuracy.pdf')
    plt.close(fig)
    print("Saved: exp_a_accuracy.png/pdf")


def plot_experiment_b_accuracy():
    df = load_experiment_b()
    fig, ax = plt.subplots()
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df[df['format'] == fmt].sort_values('test_quality')
        ax.plot(subset['test_quality'], subset['test_accuracy'] * 100,
                color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                linewidth=2, markersize=8)
    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('Accuracy [%]')
    ax.set_title('Eksperyment B: Trening na oryginałach, test na skompresowanych')
    ax.set_xticks(QUALITY_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'exp_b_accuracy.png')
    fig.savefig(PLOTS_DIR / 'exp_b_accuracy.pdf')
    plt.close(fig)
    print("Saved: exp_b_accuracy.png/pdf")


def plot_experiment_a_f1():
    df = load_experiment_a()
    fig, ax = plt.subplots()
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df[df['format'] == fmt].sort_values('train_quality')
        ax.plot(subset['train_quality'], subset['test_f1_macro'],
                color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                linewidth=2, markersize=8)
    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('F1 macro')
    ax.set_title('Eksperyment A: F1 macro (trening na skompresowanych)')
    ax.set_xticks(QUALITY_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'exp_a_f1.png')
    fig.savefig(PLOTS_DIR / 'exp_a_f1.pdf')
    plt.close(fig)
    print("Saved: exp_a_f1.png/pdf")


def plot_experiment_b_f1():
    df = load_experiment_b()
    fig, ax = plt.subplots()
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df[df['format'] == fmt].sort_values('test_quality')
        ax.plot(subset['test_quality'], subset['test_f1_macro'],
                color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                linewidth=2, markersize=8)
    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('F1 macro')
    ax.set_title('Eksperyment B: F1 macro (test na skompresowanych)')
    ax.set_xticks(QUALITY_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'exp_b_f1.png')
    fig.savefig(PLOTS_DIR / 'exp_b_f1.pdf')
    plt.close(fig)
    print("Saved: exp_b_f1.png/pdf")


def plot_quality_psnr():
    df = load_quality_metrics()
    fig, ax = plt.subplots()
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df[df['format'] == fmt]
        means = subset.groupby('quality')['psnr'].mean().reset_index()
        means = means[means['psnr'] < 100]  # exclude lossless inf values
        means = means.sort_values('quality')
        ax.plot(means['quality'], means['psnr'],
                color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                linewidth=2, markersize=8)
    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('PSNR [dB]')
    ax.set_title('Jakość kompresji: PSNR vs poziom jakości')
    ax.set_xticks(QUALITY_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'quality_psnr.png')
    fig.savefig(PLOTS_DIR / 'quality_psnr.pdf')
    plt.close(fig)
    print("Saved: quality_psnr.png/pdf")


def plot_quality_ssim():
    df = load_quality_metrics()
    fig, ax = plt.subplots()
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df[df['format'] == fmt]
        means = subset.groupby('quality')['ssim'].mean().reset_index()
        means = means.sort_values('quality')
        ax.plot(means['quality'], means['ssim'],
                color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                linewidth=2, markersize=8)
    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('SSIM')
    ax.set_title('Jakość kompresji: SSIM vs poziom jakości')
    ax.set_xticks(QUALITY_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'quality_ssim.png')
    fig.savefig(PLOTS_DIR / 'quality_ssim.pdf')
    plt.close(fig)
    print("Saved: quality_ssim.png/pdf")


def plot_compression_ratio():
    df = load_quality_metrics()
    fig, ax = plt.subplots()
    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df[df['format'] == fmt]
        means = subset.groupby('quality')['compression_ratio'].mean().reset_index()
        means = means.sort_values('quality')
        ax.plot(means['quality'], means['compression_ratio'],
                color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                linewidth=2, markersize=8)
    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('Współczynnik kompresji')
    ax.set_title('Współczynnik kompresji vs poziom jakości')
    ax.set_xticks(QUALITY_LEVELS)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'compression_ratio.png')
    fig.savefig(PLOTS_DIR / 'compression_ratio.pdf')
    plt.close(fig)
    print("Saved: compression_ratio.png/pdf")


def plot_combined_ab():
    """Combined plot: Exp A and B side by side."""
    df_a = load_experiment_a()
    df_b = load_experiment_b()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df_a[df_a['format'] == fmt].sort_values('train_quality')
        ax1.plot(subset['train_quality'], subset['test_accuracy'] * 100,
                 color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                 linewidth=2, markersize=8)

    ax1.set_xlabel('Poziom jakości Q (trening)')
    ax1.set_ylabel('Accuracy [%]')
    ax1.set_title('Eksperyment A\n(trening na skompresowanych)')
    ax1.set_xticks(QUALITY_LEVELS)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        subset = df_b[df_b['format'] == fmt].sort_values('test_quality')
        ax2.plot(subset['test_quality'], subset['test_accuracy'] * 100,
                 color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                 linewidth=2, markersize=8)

    ax2.set_xlabel('Poziom jakości Q (test)')
    ax2.set_title('Eksperyment B\n(test na skompresowanych)')
    ax2.set_xticks(QUALITY_LEVELS)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'combined_ab.png')
    fig.savefig(PLOTS_DIR / 'combined_ab.pdf')
    plt.close(fig)
    print("Saved: combined_ab.png/pdf")


if __name__ == '__main__':
    print("Generating plots...")
    plot_experiment_a_accuracy()
    plot_experiment_b_accuracy()
    plot_experiment_a_f1()
    plot_experiment_b_f1()
    plot_quality_psnr()
    plot_quality_ssim()
    plot_compression_ratio()
    plot_combined_ab()
    print(f"\nAll plots saved to: {PLOTS_DIR}")
