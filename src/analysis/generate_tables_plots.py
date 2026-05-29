"""
Generate publication-ready tables and plots for the master's thesis article.

Creates:
- LaTeX tables for Experiment A
- CSV summaries for data analysis
- Matplotlib plots showing compression impact
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def load_all_results(
    model_name: str = "resnet50",
    task: str = "syntax"
) -> Dict[str, pd.DataFrame]:
    """
    Load Experiment A results for the given model and task (ARCADE).

    Args:
        model_name: Model to load results for
        task: Task to load results for

    Returns:
        Dictionary with DataFrames for each experiment type
    """
    results = {}

    # Load ARCADE Experiment A
    exp_a_path = config.RESULTS_ROOT / "experiment_a" / f"{model_name}_arcade_{task}_jpeg_results.csv"
    if exp_a_path.exists():
        df_a = pd.read_csv(exp_a_path)
        # Load other formats
        for fmt in ['jpeg2000', 'avif']:
            fmt_path = config.RESULTS_ROOT / "experiment_a" / f"{model_name}_arcade_{task}_{fmt}_results.csv"
            if fmt_path.exists():
                df_a = pd.concat([df_a, pd.read_csv(fmt_path)], ignore_index=True)
        results['arcade_exp_a'] = df_a

    # Load baseline (uncompressed PNG reference) — optional. Kept separate from
    # the per-format DataFrame so it never enters the format groupby.
    baseline_df = load_baseline(model_name, task)
    if baseline_df is not None:
        results['arcade_baseline'] = baseline_df

    return results


def load_baseline(
    model_name: str = "resnet50",
    task: str = "syntax"
) -> Optional[pd.DataFrame]:
    """
    Load the baseline (uncompressed PNG) result for the given model and task.

    The baseline file is a single-row CSV produced by run_baseline.py and is
    the upper-bound reference for Experiment A. It is optional: if the file is
    missing or unreadable, this returns None silently.

    Args:
        model_name: Model to load the baseline for
        task: Task to load the baseline for

    Returns:
        Single-row DataFrame with baseline metrics, or None if unavailable.
    """
    baseline_path = (
        config.RESULTS_ROOT / "experiment_a"
        / f"{model_name}_arcade_{task}_baseline_results.csv"
    )
    if not baseline_path.exists():
        return None
    try:
        df = pd.read_csv(baseline_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if df.empty:
        return None
    return df


def generate_latex_table_experiment_a(
    df: pd.DataFrame,
    model_name: str,
    task: str,
    output_path: Optional[Path] = None,
    baseline_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate LaTeX table for Experiment A results.

    mAP (test_map) is the leading metric: it is threshold-free and the most
    reliable score for strongly imbalanced multi-label classification. F1-Macro
    and Hamming accuracy are kept as auxiliary columns.

    Args:
        df: DataFrame with Experiment A results
        model_name: Name of the model
        task: Task name
        output_path: Optional path to save the table
        baseline_df: Optional single-row DataFrame with the uncompressed
            (baseline) reference result. If provided, a "Baseline" row is added
            above the format rows as the upper-bound reference.

    Returns:
        LaTeX table as string
    """
    # Group by format and quality. Baseline is NOT part of this groupby — it is
    # loaded separately and rendered as a dedicated reference row.
    grouped = df.groupby(['format', 'train_quality']).agg({
        'test_map': ['mean', 'std'],
        'test_f1_macro': ['mean', 'std'],
        'test_hamming_accuracy': ['mean', 'std'],
    }).reset_index()

    grouped.columns = [
        'format', 'quality',
        'map_mean', 'map_std',
        'f1_mean', 'f1_std',
        'acc_mean', 'acc_std',
    ]

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Wyniki Eksperymentu A: {model_name.upper()} - {task.upper()}}}")
    latex.append(f"\\label{{tab:experiment_a_{model_name}}}")
    # 5 columns: Format, Q, mAP, F1-Macro, Test Acc.
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("Format & Q & mAP (\\%) & F1-Macro (\\%) & Test Acc (\\%) \\\\")
    latex.append("\\hline")

    def _fmt(mean, std):
        # Pandas std with n=1 produces NaN — render as just the mean.
        if pd.isna(std):
            return f"${mean:.2f}$"
        return f"${mean:.2f} \\pm {std:.2f}$"

    def _scalar(value):
        # Render a single baseline metric value (no std available).
        if value is None or pd.isna(value):
            return "$-$"
        return f"${value * 100:.2f}$"

    # Baseline reference row (uncompressed PNG) — separated by a midrule.
    if baseline_df is not None and not baseline_df.empty:
        b = baseline_df.iloc[0]
        b_map = b['test_map'] if 'test_map' in baseline_df.columns else None
        b_f1 = b['test_f1_macro'] if 'test_f1_macro' in baseline_df.columns else None
        b_acc = b['test_hamming_accuracy'] if 'test_hamming_accuracy' in baseline_df.columns else None
        latex.append(
            "Baseline (uncompressed) & -- & "
            f"{_scalar(b_map)} & {_scalar(b_f1)} & {_scalar(b_acc)} \\\\"
        )
        latex.append("\\hline")

    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        fmt_data = grouped[grouped['format'] == fmt].sort_values('quality')

        if not fmt_data.empty:
            for _, row in fmt_data.iterrows():
                quality = int(row['quality'])
                map_mean = row['map_mean'] * 100
                map_std = row['map_std'] * 100 if pd.notna(row['map_std']) else float('nan')
                f1_mean = row['f1_mean'] * 100
                f1_std = row['f1_std'] * 100 if pd.notna(row['f1_std']) else float('nan')
                acc_mean = row['acc_mean'] * 100
                acc_std = row['acc_std'] * 100 if pd.notna(row['acc_std']) else float('nan')

                latex.append(
                    f"{fmt.upper()} & {quality} & "
                    f"{_fmt(map_mean, map_std)} & {_fmt(f1_mean, f1_std)} & "
                    f"{_fmt(acc_mean, acc_std)} \\\\"
                )

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_path}")

    return latex_str


def generate_comparison_plot(
    results: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
    model_name: str = "model"
) -> None:
    """
    Generate comparison plots for all experiments.

    Args:
        results: Dictionary with experiment DataFrames
        output_dir: Directory to save plots
        model_name: Model name, embedded in the output filename so runs for
            different models do not overwrite each other.
    """
    if output_dir is None:
        output_dir = config.PLOTS_ROOT
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 'seaborn-v0_8-whitegrid' only exists on matplotlib >= 3.6; fall back gracefully
    # on older versions instead of crashing with OSError.
    for style in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid'):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    # Plot 1: Experiment A — mAP (leading metric) and F1-Macro (auxiliary) vs Quality.
    # mAP is threshold-free and the primary metric for imbalanced multi-label.
    if 'arcade_exp_a' in results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        df = results['arcade_exp_a']
        baseline_df = results.get('arcade_baseline')

        # Baseline reference values (uncompressed PNG), if available.
        b_map = b_f1 = None
        if baseline_df is not None and not baseline_df.empty:
            b_row = baseline_df.iloc[0]
            if 'test_map' in baseline_df.columns and pd.notna(b_row['test_map']):
                b_map = b_row['test_map'] * 100
            if 'test_f1_macro' in baseline_df.columns and pd.notna(b_row['test_f1_macro']):
                b_f1 = b_row['test_f1_macro'] * 100

        for idx, fmt in enumerate(['jpeg', 'jpeg2000', 'avif']):
            ax = axes[idx]
            fmt_data = df[df['format'] == fmt].sort_values('train_quality')

            if not fmt_data.empty:
                x = fmt_data['train_quality']
                # Primary series: mAP.
                ax.plot(x, fmt_data['test_map'] * 100, marker='o', linewidth=2,
                        markersize=8, color='#1f77b4', label='mAP')
                # Auxiliary series: F1-Macro.
                ax.plot(x, fmt_data['test_f1_macro'] * 100, marker='s', linewidth=2,
                        markersize=7, color='#ff7f0e', linestyle='--', label='F1-Macro')

                # Baseline reference lines (uncompressed).
                if b_map is not None:
                    ax.axhline(b_map, color='#1f77b4', linestyle=':', linewidth=1.5,
                               label='Baseline mAP (uncompressed)')
                if b_f1 is not None:
                    ax.axhline(b_f1, color='#ff7f0e', linestyle=':', linewidth=1.5,
                               label='Baseline F1-Macro (uncompressed)')

                ax.set_xlabel('Jakość kompresji (Q)', fontsize=12)
                ax.set_ylabel('Metryka (%)', fontsize=12)
                ax.set_title(f'{fmt.upper()}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                # Reverse x-axis so quality decreases left-to-right. Use only
                # invert_xaxis(); combining it with set_xlim(105, 5) would cancel
                # out (invert_xaxis swaps the current limits) and yield an
                # ascending axis instead.
                ax.invert_xaxis()

        plt.suptitle('Eksperyment A: Trening na skompresowanych, test na oryginałach (mAP wiodąca)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        out_png = output_dir / f'experiment_a_accuracy_{model_name}.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {out_png}")
        plt.close()


def generate_summary_tables(
    results: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
    model_name: str = "model"
) -> None:
    """
    Generate summary CSV tables for data analysis.

    Args:
        results: Dictionary with experiment DataFrames
        output_dir: Directory to save tables
        model_name: Model name, embedded in the output filenames so runs for
            different models do not overwrite each other.
    """
    if output_dir is None:
        output_dir = config.RESULTS_ROOT / "tables"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table for Experiment A — mAP first as the leading metric.
    if 'arcade_exp_a' in results:
        df = results['arcade_exp_a']
        summary_a = df.groupby(['format', 'train_quality']).agg({
            'test_map': ['mean', 'std', 'min', 'max'],
            'test_f1_macro': ['mean', 'std'],
            'test_hamming_accuracy': ['mean', 'std'],
        }).reset_index()
        summary_a.columns = [
            'format', 'quality',
            'map_mean', 'map_std', 'map_min', 'map_max',
            'f1_mean', 'f1_std',
            'acc_mean', 'acc_std',
        ]
        out_csv = output_dir / f'experiment_a_summary_{model_name}.csv'
        summary_a.to_csv(out_csv, index=False)
        print(f"Summary table saved: {out_csv}")

    # Baseline reference summary (uncompressed PNG) — written separately so it
    # never mixes into the per-format aggregation above.
    if 'arcade_baseline' in results:
        baseline_df = results['arcade_baseline']
        baseline_cols = [c for c in ['format', 'train_quality', 'test_map',
                                     'test_f1_macro', 'test_hamming_accuracy',
                                     'test_f1_micro', 'test_subset_accuracy']
                         if c in baseline_df.columns]
        out_bcsv = output_dir / f'experiment_a_baseline_summary_{model_name}.csv'
        baseline_df[baseline_cols].to_csv(out_bcsv, index=False)
        print(f"Baseline summary saved: {out_bcsv}")



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate tables and plots for article")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model name"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="syntax",
        help="Task name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots and tables"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GENERATING TABLES AND PLOTS FOR ARTICLE")
    print("=" * 80)

    # Load all results
    print("\nLoading experiment results...")
    results = load_all_results(model_name=args.model, task=args.task)

    if not results:
        print("[ERROR] No results found!")
        print("Please run the experiment first, e.g.:")
        print("  python -m src.experiments.experiment_a --model resnet50 --task syntax --format jpeg")
        return

    print(f"Loaded results for: {list(results.keys())}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Generate LaTeX tables. Default to results/tables/ when no --output-dir is
    # given (matching the summary-table fallback) so the .tex is always written;
    # filename carries the model name to avoid overwriting between models.
    print("\nGenerating LaTeX tables...")
    if 'arcade_exp_a' in results:
        tables_dir = (Path(output_dir) / 'tables') if output_dir else (config.RESULTS_ROOT / "tables")
        generate_latex_table_experiment_a(
            results['arcade_exp_a'],
            args.model,
            args.task,
            output_path=tables_dir / f'experiment_a_table_{args.model}.tex',
            baseline_df=results.get('arcade_baseline'),
        )

    # Generate plots
    print("\nGenerating plots...")
    generate_comparison_plot(results, output_dir=output_dir, model_name=args.model)

    # Generate summary tables
    print("\nGenerating summary tables...")
    generate_summary_tables(results, output_dir=output_dir, model_name=args.model)

    print("\n" + "=" * 80)
    print("ALL TABLES AND PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
