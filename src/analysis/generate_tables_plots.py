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

    return results


def generate_latex_table_experiment_a(
    df: pd.DataFrame,
    model_name: str,
    task: str,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate LaTeX table for Experiment A results.

    Args:
        df: DataFrame with Experiment A results
        model_name: Name of the model
        task: Task name
        output_path: Optional path to save the table

    Returns:
        LaTeX table as string
    """
    # Group by format and quality
    grouped = df.groupby(['format', 'train_quality']).agg({
        'test_hamming_accuracy': ['mean', 'std'],
        'test_f1_macro': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['format', 'quality', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std']

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Wyniki Eksperymentu A: {model_name.upper()} - {task.upper()}}}")
    latex.append("\\label{tab:experiment_a}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("Format & Q & Test Acc (\\%) & F1-Macro (\\%) \\\\")
    latex.append("\\hline")

    def _fmt(mean, std):
        # Pandas std with n=1 produces NaN — render as just the mean.
        if pd.isna(std):
            return f"${mean:.2f}$"
        return f"${mean:.2f} \\pm {std:.2f}$"

    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        fmt_data = grouped[grouped['format'] == fmt].sort_values('quality')

        if not fmt_data.empty:
            for _, row in fmt_data.iterrows():
                quality = row['quality']
                acc_mean = row['acc_mean'] * 100
                acc_std = row['acc_std'] * 100 if pd.notna(row['acc_std']) else float('nan')
                f1_mean = row['f1_mean'] * 100
                f1_std = row['f1_std'] * 100 if pd.notna(row['f1_std']) else float('nan')

                latex.append(
                    f"{fmt.upper()} & {quality} & "
                    f"{_fmt(acc_mean, acc_std)} & {_fmt(f1_mean, f1_std)} \\\\"
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
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate comparison plots for all experiments.

    Args:
        results: Dictionary with experiment DataFrames
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.PLOTS_ROOT
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Experiment A - Test Accuracy vs Quality
    if 'arcade_exp_a' in results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        df = results['arcade_exp_a']

        for idx, fmt in enumerate(['jpeg', 'jpeg2000', 'avif']):
            ax = axes[idx]
            fmt_data = df[df['format'] == fmt].sort_values('train_quality')

            if not fmt_data.empty:
                x = fmt_data['train_quality']
                y = fmt_data['test_hamming_accuracy'] * 100

                ax.plot(x, y, marker='o', linewidth=2, markersize=8, label=fmt.upper())
                ax.fill_between(x, y - 2, y + 2, alpha=0.2)  # Confidence interval approximation

                ax.set_xlabel('Jakość kompresji (Q)', fontsize=12)
                ax.set_ylabel('Dokładność testowa (%)', fontsize=12)
                ax.set_title(f'{fmt.upper()}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(105, 5)  # Reverse x-axis
                ax.invert_xaxis()

        plt.suptitle('Eksperyment A: Trening na skompresowanych, test na oryginałach',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'experiment_a_accuracy.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved: {output_dir / 'experiment_a_accuracy.png'}")
        plt.close()


def generate_summary_tables(
    results: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate summary CSV tables for data analysis.

    Args:
        results: Dictionary with experiment DataFrames
        output_dir: Directory to save tables
    """
    if output_dir is None:
        output_dir = config.RESULTS_ROOT / "tables"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table for Experiment A
    if 'arcade_exp_a' in results:
        df = results['arcade_exp_a']
        summary_a = df.groupby(['format', 'train_quality']).agg({
            'test_hamming_accuracy': ['mean', 'std', 'min', 'max'],
            'test_f1_macro': ['mean', 'std']
        }).reset_index()
        summary_a.columns = ['format', 'quality', 'acc_mean', 'acc_std', 'acc_min', 'acc_max', 'f1_mean', 'f1_std']
        summary_a.to_csv(output_dir / 'experiment_a_summary.csv', index=False)
        print(f"Summary table saved: {output_dir / 'experiment_a_summary.csv'}")



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

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    if 'arcade_exp_a' in results:
        generate_latex_table_experiment_a(
            results['arcade_exp_a'],
            args.model,
            args.task,
            output_path=(output_dir / 'tables' / 'experiment_a_table.tex') if output_dir else None
        )

    # Generate plots
    print("\nGenerating plots...")
    generate_comparison_plot(results, output_dir=output_dir)

    # Generate summary tables
    print("\nGenerating summary tables...")
    generate_summary_tables(results, output_dir=output_dir)

    print("\n" + "=" * 80)
    print("ALL TABLES AND PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
