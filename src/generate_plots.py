"""Generate plots for the thesis article."""
import sys
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

# Use config paths
RESULTS_ROOT = config.RESULTS_ROOT
PLOTS_DIR = config.PLOTS_ROOT
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


def load_experiment_a(model_name='resnet50', task='syntax'):
    """Load experiment A results for given model and task."""
    frames = []
    for fmt in config.COMPRESSION_FORMATS:
        path = RESULTS_ROOT / "experiment_a" / f"{model_name}_{task}_{fmt}_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No experiment A results found for {model_name}_{task}")
    return pd.concat(frames, ignore_index=True)


def load_experiment_b(model_name='resnet50', task='syntax'):
    """Load experiment B results for given model and task."""
    frames = []
    for fmt in config.COMPRESSION_FORMATS:
        path = RESULTS_ROOT / "experiment_b" / f"{model_name}_{task}_{fmt}_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No experiment B results found for {model_name}_{task}")
    return pd.concat(frames, ignore_index=True)


def load_quality_metrics(task='syntax'):
    """Load quality metrics for given task."""
    frames = []
    for fmt in config.COMPRESSION_FORMATS:
        path = RESULTS_ROOT / "metrics" / f"quality_{task}_train_{fmt}.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No quality metrics found for {task}")
    return pd.concat(frames, ignore_index=True)


def get_quality_levels(df):
    """Extract quality levels from dataframe."""
    if 'train_quality' in df.columns:
        return sorted(df['train_quality'].unique(), reverse=True)
    elif 'test_quality' in df.columns:
        return sorted(df['test_quality'].unique(), reverse=True)
    elif 'quality' in df.columns:
        return sorted(df['quality'].unique(), reverse=True)
    return config.QUALITY_LEVELS_MVP


def plot_experiment_a_accuracy(model_name='resnet50', task='syntax'):
    """Plot experiment A accuracy results."""
    df = load_experiment_a(model_name, task)
    quality_levels = get_quality_levels(df)

    fig, ax = plt.subplots()
    for fmt in config.COMPRESSION_FORMATS:
        subset = df[df['format'] == fmt].sort_values('train_quality')
        if len(subset) > 0:
            ax.plot(subset['train_quality'], subset['test_accuracy'] * 100,
                    color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                    linewidth=2, markersize=8)

    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('Accuracy [%]')
    ax.set_title(f'Eksperyment A: Trening na skompresowanych, test na oryginałach\n({model_name}, {task})')
    ax.set_xticks(quality_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()

    output_name = f'exp_a_accuracy_{model_name}_{task}'
    fig.savefig(PLOTS_DIR / f'{output_name}.png')
    fig.savefig(PLOTS_DIR / f'{output_name}.pdf')
    plt.close(fig)
    print(f"Saved: {output_name}.png/pdf")


def plot_experiment_b_accuracy(model_name='resnet50', task='syntax'):
    """Plot experiment B accuracy results."""
    df = load_experiment_b(model_name, task)
    quality_levels = get_quality_levels(df)

    fig, ax = plt.subplots()
    for fmt in config.COMPRESSION_FORMATS:
        subset = df[df['format'] == fmt].sort_values('test_quality')
        if len(subset) > 0:
            ax.plot(subset['test_quality'], subset['test_accuracy'] * 100,
                    color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                    linewidth=2, markersize=8)

    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('Accuracy [%]')
    ax.set_title(f'Eksperyment B: Trening na oryginałach, test na skompresowanych\n({model_name}, {task})')
    ax.set_xticks(quality_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()

    output_name = f'exp_b_accuracy_{model_name}_{task}'
    fig.savefig(PLOTS_DIR / f'{output_name}.png')
    fig.savefig(PLOTS_DIR / f'{output_name}.pdf')
    plt.close(fig)
    print(f"Saved: {output_name}.png/pdf")


def plot_experiment_a_f1(model_name='resnet50', task='syntax'):
    """Plot experiment A F1 results."""
    df = load_experiment_a(model_name, task)
    quality_levels = get_quality_levels(df)

    fig, ax = plt.subplots()
    for fmt in config.COMPRESSION_FORMATS:
        subset = df[df['format'] == fmt].sort_values('train_quality')
        if len(subset) > 0:
            ax.plot(subset['train_quality'], subset['test_f1_macro'],
                    color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                    linewidth=2, markersize=8)

    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('F1 macro')
    ax.set_title(f'Eksperyment A: F1 macro (trening na skompresowanych)\n({model_name}, {task})')
    ax.set_xticks(quality_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()

    output_name = f'exp_a_f1_{model_name}_{task}'
    fig.savefig(PLOTS_DIR / f'{output_name}.png')
    fig.savefig(PLOTS_DIR / f'{output_name}.pdf')
    plt.close(fig)
    print(f"Saved: {output_name}.png/pdf")


def plot_experiment_b_f1(model_name='resnet50', task='syntax'):
    """Plot experiment B F1 results."""
    df = load_experiment_b(model_name, task)
    quality_levels = get_quality_levels(df)

    fig, ax = plt.subplots()
    for fmt in config.COMPRESSION_FORMATS:
        subset = df[df['format'] == fmt].sort_values('test_quality')
        if len(subset) > 0:
            ax.plot(subset['test_quality'], subset['test_f1_macro'],
                    color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                    linewidth=2, markersize=8)

    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('F1 macro')
    ax.set_title(f'Eksperyment B: F1 macro (test na skompresowanych)\n({model_name}, {task})')
    ax.set_xticks(quality_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()

    output_name = f'exp_b_f1_{model_name}_{task}'
    fig.savefig(PLOTS_DIR / f'{output_name}.png')
    fig.savefig(PLOTS_DIR / f'{output_name}.pdf')
    plt.close(fig)
    print(f"Saved: {output_name}.png/pdf")


def plot_quality_psnr(task='syntax'):
    """Plot PSNR quality metrics."""
    df = load_quality_metrics(task)
    quality_levels = get_quality_levels(df)

    fig, ax = plt.subplots()
    for fmt in config.COMPRESSION_FORMATS:
        subset = df[df['format'] == fmt]
        if len(subset) > 0:
            means = subset.groupby('quality')['psnr'].mean().reset_index()
            means = means[means['psnr'] < 100]  # exclude lossless inf values
            means = means.sort_values('quality')
            ax.plot(means['quality'], means['psnr'],
                    color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                    linewidth=2, markersize=8)

    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('PSNR [dB]')
    ax.set_title(f'Jakość kompresji: PSNR vs poziom jakości ({task})')
    ax.set_xticks(quality_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()

    output_name = f'quality_psnr_{task}'
    fig.savefig(PLOTS_DIR / f'{output_name}.png')
    fig.savefig(PLOTS_DIR / f'{output_name}.pdf')
    plt.close(fig)
    print(f"Saved: {output_name}.png/pdf")


def plot_quality_ssim(task='syntax'):
    """Plot SSIM quality metrics."""
    df = load_quality_metrics(task)
    quality_levels = get_quality_levels(df)

    fig, ax = plt.subplots()
    for fmt in config.COMPRESSION_FORMATS:
        subset = df[df['format'] == fmt]
        if len(subset) > 0:
            means = subset.groupby('quality')['ssim'].mean().reset_index()
            means = means.sort_values('quality')
            ax.plot(means['quality'], means['ssim'],
                    color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                    linewidth=2, markersize=8)

    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('SSIM')
    ax.set_title(f'Jakość kompresji: SSIM vs poziom jakości ({task})')
    ax.set_xticks(quality_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()

    output_name = f'quality_ssim_{task}'
    fig.savefig(PLOTS_DIR / f'{output_name}.png')
    fig.savefig(PLOTS_DIR / f'{output_name}.pdf')
    plt.close(fig)
    print(f"Saved: {output_name}.png/pdf")


def plot_compression_ratio(task='syntax'):
    """Plot compression ratio."""
    df = load_quality_metrics(task)
    quality_levels = get_quality_levels(df)

    fig, ax = plt.subplots()
    for fmt in config.COMPRESSION_FORMATS:
        subset = df[df['format'] == fmt]
        if len(subset) > 0:
            means = subset.groupby('quality')['compression_ratio'].mean().reset_index()
            means = means.sort_values('quality')
            ax.plot(means['quality'], means['compression_ratio'],
                    color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                    linewidth=2, markersize=8)

    ax.set_xlabel('Poziom jakości kompresji (Q)')
    ax.set_ylabel('Współczynnik kompresji')
    ax.set_title(f'Współczynnik kompresji vs poziom jakości ({task})')
    ax.set_xticks(quality_levels)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()

    output_name = f'compression_ratio_{task}'
    fig.savefig(PLOTS_DIR / f'{output_name}.png')
    fig.savefig(PLOTS_DIR / f'{output_name}.pdf')
    plt.close(fig)
    print(f"Saved: {output_name}.png/pdf")


def plot_combined_ab(model_name='resnet50', task='syntax'):
    """Combined plot: Exp A and B side by side."""
    df_a = load_experiment_a(model_name, task)
    df_b = load_experiment_b(model_name, task)
    quality_levels = get_quality_levels(df_a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for fmt in config.COMPRESSION_FORMATS:
        subset = df_a[df_a['format'] == fmt].sort_values('train_quality')
        if len(subset) > 0:
            ax1.plot(subset['train_quality'], subset['test_accuracy'] * 100,
                     color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                     linewidth=2, markersize=8)

    ax1.set_xlabel('Poziom jakości Q (trening)')
    ax1.set_ylabel('Accuracy [%]')
    ax1.set_title(f'Eksperyment A\n(trening na skompresowanych)')
    ax1.set_xticks(quality_levels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    for fmt in config.COMPRESSION_FORMATS:
        subset = df_b[df_b['format'] == fmt].sort_values('test_quality')
        if len(subset) > 0:
            ax2.plot(subset['test_quality'], subset['test_accuracy'] * 100,
                     color=COLORS[fmt], marker=MARKERS[fmt], label=LABELS[fmt],
                     linewidth=2, markersize=8)

    ax2.set_xlabel('Poziom jakości Q (test)')
    ax2.set_title(f'Eksperyment B\n(test na skompresowanych)')
    ax2.set_xticks(quality_levels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    fig.suptitle(f'{model_name} - {task}', fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_name = f'combined_ab_{model_name}_{task}'
    fig.savefig(PLOTS_DIR / f'{output_name}.png')
    fig.savefig(PLOTS_DIR / f'{output_name}.pdf')
    plt.close(fig)
    print(f"Saved: {output_name}.png/pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for thesis")
    parser.add_argument("--model", type=str, default="resnet50", choices=config.SUPPORTED_MODELS,
                       help="Model name")
    parser.add_argument("--task", type=str, default="syntax", choices=config.TASKS,
                       help="Dataset task")
    parser.add_argument("--all-models", action="store_true",
                       help="Generate plots for all supported models")
    args = parser.parse_args()

    models = config.SUPPORTED_MODELS if args.all_models else [args.model]

    for model_name in models:
        print(f"\nGenerating plots for {model_name}...")
        try:
            plot_experiment_a_accuracy(model_name, args.task)
            plot_experiment_b_accuracy(model_name, args.task)
            plot_experiment_a_f1(model_name, args.task)
            plot_experiment_b_f1(model_name, args.task)
            plot_combined_ab(model_name, args.task)
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")

    # Quality metrics (task-specific, not model-specific)
    print(f"\nGenerating quality metric plots for {args.task}...")
    try:
        plot_quality_psnr(args.task)
        plot_quality_ssim(args.task)
        plot_compression_ratio(args.task)
    except FileNotFoundError as e:
        print(f"Skipping quality plots: {e}")

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    main()
