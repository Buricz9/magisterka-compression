"""
Generate publication-ready tables and plots for the master's thesis article.

Creates:
- LaTeX tables for experiments A and B
- CSV summaries for data analysis
- Matplotlib plots showing compression impact
- Combined comparison plots

Author: Master's Thesis Project
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def load_all_results(
    model_name: str = "resnet50",
    task: str = "syntax"
) -> Dict[str, pd.DataFrame]:
    """
    Load all experiment results for both models and datasets.

    Args:
        model_name: Model to load results for
        task: Task to load results for

    Returns:
        Dictionary with DataFrames for each experiment type
    """
    results = {}

    # Load ARCADE Experiment A
    exp_a_path = config.RESULTS_ROOT / "experiment_a" / f"{model_name}_{task}_jpeg_results.csv"
    if exp_a_path.exists():
        df_a = pd.read_csv(exp_a_path)
        # Load other formats
        for fmt in ['jpeg2000', 'avif']:
            fmt_path = config.RESULTS_ROOT / "experiment_a" / f"{model_name}_{task}_{fmt}_results.csv"
            if fmt_path.exists():
                df_a = pd.concat([df_a, pd.read_csv(fmt_path)], ignore_index=True)
        results['arcade_exp_a'] = df_a

    # Load ARCADE Experiment B
    exp_b_path = config.RESULTS_ROOT / "experiment_b" / f"{model_name}_{task}_jpeg_results.csv"
    if exp_b_path.exists():
        df_b = pd.read_csv(exp_b_path)
        for fmt in ['jpeg2000', 'avif']:
            fmt_path = config.RESULTS_ROOT / "experiment_b" / f"{model_name}_{task}_{fmt}_results.csv"
            if fmt_path.exists():
                df_b = pd.concat([df_b, pd.read_csv(fmt_path)], ignore_index=True)
        results['arcade_exp_b'] = df_b

    # Load EfficientNet-B0 results if available
    eff_path = config.RESULTS_ROOT / "efficientnet_b0"
    if eff_path.exists():
        eff_files = list(eff_path.glob("efficientnet_b0_all_results_*.csv"))
        if eff_files:
            results['efficientnet'] = pd.read_csv(eff_files[-1])  # Latest

    # Load ISIC results if available
    for exp_type in ['isic_experiment_a', 'isic_experiment_b']:
        isic_path = config.RESULTS_ROOT / exp_type
        if isic_path.exists():
            isic_files = list(isic_path.glob(f"{model_name}_isic_results.csv"))
            if isic_files:
                results[exp_type] = pd.read_csv(isic_files[0])

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
        'test_accuracy': ['mean', 'std'],
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

    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        fmt_data = grouped[grouped['format'] == fmt].sort_values('quality')

        if not fmt_data.empty:
            for _, row in fmt_data.iterrows():
                quality = row['quality']
                acc_mean = row['acc_mean'] * 100
                acc_std = row['acc_std'] * 100
                f1_mean = row['f1_mean'] * 100
                f1_std = row['f1_std'] * 100

                latex.append(
                    f"{fmt.upper()} & {quality} & "
                    f"${acc_mean:.2f} \\pm {acc_std:.2f}$ & "
                    f"${f1_mean:.2f} \\pm {f1_std:.2f}$ \\\\"
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


def generate_latex_table_experiment_b(
    df: pd.DataFrame,
    model_name: str,
    task: str,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate LaTeX table for Experiment B results.

    Args:
        df: DataFrame with Experiment B results
        model_name: Name of the model
        task: Task name
        output_path: Optional path to save the table

    Returns:
        LaTeX table as string
    """
    # Group by format and quality
    grouped = df.groupby(['format', 'test_quality']).agg({
        'test_accuracy': ['mean', 'std'],
        'test_f1_macro': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['format', 'quality', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std']

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Wyniki Eksperymentu B: {model_name.upper()} - {task.upper()}}}")
    latex.append("\\label{tab:experiment_b}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("Format & Q & Test Acc (\\%) & F1-Macro (\\%) \\\\")
    latex.append("\\hline")

    for fmt in ['jpeg', 'jpeg2000', 'avif']:
        fmt_data = grouped[grouped['format'] == fmt].sort_values('quality')

        if not fmt_data.empty:
            for _, row in fmt_data.iterrows():
                quality = row['quality']
                acc_mean = row['acc_mean'] * 100
                acc_std = row['acc_std'] * 100
                f1_mean = row['f1_mean'] * 100
                f1_std = row['f1_std'] * 100

                latex.append(
                    f"{fmt.upper()} & {quality} & "
                    f"${acc_mean:.2f} \\pm {acc_std:.2f}$ & "
                    f"${f1_mean:.2f} \\pm {f1_std:.2f}$ \\\\"
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
                y = fmt_data['test_accuracy'] * 100

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

    # Plot 2: Experiment B - Test Accuracy vs Quality
    if 'arcade_exp_b' in results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        df = results['arcade_exp_b']

        for idx, fmt in enumerate(['jpeg', 'jpeg2000', 'avif']):
            ax = axes[idx]
            fmt_data = df[df['format'] == fmt].sort_values('test_quality')

            if not fmt_data.empty:
                x = fmt_data['test_quality']
                y = fmt_data['test_accuracy'] * 100

                ax.plot(x, y, marker='s', linewidth=2, markersize=8, label=fmt.upper(), color='red')
                ax.fill_between(x, y - 2, y + 2, alpha=0.2, color='red')

                ax.set_xlabel('Jakość kompresji (Q)', fontsize=12)
                ax.set_ylabel('Dokładność testowa (%)', fontsize=12)
                ax.set_title(f'{fmt.upper()}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(105, 5)
                ax.invert_xaxis()

        plt.suptitle('Eksperyment B: Trening na oryginałach, test na skompresowanych',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'experiment_b_accuracy.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved: {output_dir / 'experiment_b_accuracy.png'}")
        plt.close()

    # Plot 3: Format comparison at selected quality levels
    if 'arcade_exp_a' in results and 'arcade_exp_b' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        df_a = results['arcade_exp_a']
        df_b = results['arcade_exp_b']

        qualities = [100, 85, 70, 50, 30, 10]

        # Experiment A
        ax = axes[0]
        for fmt in ['jpeg', 'jpeg2000', 'avif']:
            fmt_data = df_a[df_a['format'] == fmt]
            if not fmt_data.empty:
                grouped = fmt_data.groupby('train_quality').agg({
                    'test_accuracy': 'mean'
                }).reindex(qualities)
                ax.plot(grouped.index, grouped['test_accuracy'] * 100,
                       marker='o', label=fmt.upper(), linewidth=2)

        ax.set_xlabel('Jakość kompresji (Q)', fontsize=12)
        ax.set_ylabel('Dokładność testowa (%)', fontsize=12)
        ax.set_title('Eksperyment A', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # Experiment B
        ax = axes[1]
        for fmt in ['jpeg', 'jpeg2000', 'avif']:
            fmt_data = df_b[df_b['format'] == fmt]
            if not fmt_data.empty:
                grouped = fmt_data.groupby('test_quality').agg({
                    'test_accuracy': 'mean'
                }).reindex(qualities)
                ax.plot(grouped.index, grouped['test_accuracy'] * 100,
                       marker='s', label=fmt.upper(), linewidth=2)

        ax.set_xlabel('Jakość kompresji (Q)', fontsize=12)
        ax.set_ylabel('Dokładność testowa (%)', fontsize=12)
        ax.set_title('Eksperyment B', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        plt.suptitle('Porównanie formatów kompresji',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'format_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved: {output_dir / 'format_comparison.png'}")
        plt.close()

    # Plot 4: Model comparison (ResNet-50 vs EfficientNet-B0)
    if 'efficientnet' in results and 'arcade_exp_a' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Get ResNet-50 results
        resnet_df = results['arcade_exp_a'][results['arcade_exp_a']['format'] == 'jpeg'].copy()
        efficientnet_df = results['efficientnet'][
            (results['efficientnet']['dataset'] == 'arcade') &
            (results['efficientnet']['format'] == 'jpeg')
        ].copy()

        if not resnet_df.empty and not efficientnet_df.empty:
            # Experiment A comparison
            ax = axes[0]

            resnet_grouped = resnet_df.groupby('train_quality').agg({'test_accuracy': 'mean'})
            efficientnet_grouped = efficientnet_df.groupby('train_quality').agg({'test_accuracy': 'mean'})

            ax.plot(resnet_grouped.index, resnet_grouped['test_accuracy'] * 100,
                   marker='o', label='ResNet-50', linewidth=2, markersize=8)
            ax.plot(efficientnet_grouped.index, efficientnet_grouped['test_accuracy'] * 100,
                   marker='s', label='EfficientNet-B0', linewidth=2, markersize=8)

            ax.set_xlabel('Jakość kompresji (Q)', fontsize=12)
            ax.set_ylabel('Dokładność testowa (%)', fontsize=12)
            ax.set_title('Porównanie modeli - Eksperyment A', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()

        # Placeholder for Experiment B if needed
        ax = axes[1]
        ax.text(0.5, 0.5, 'Eksperyment B\\n(dane niedostępne)',
               ha='center', va='center', fontsize=14,
               transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.suptitle('Porównanie architektur modeli',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved: {output_dir / 'model_comparison.png'}")
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
            'test_accuracy': ['mean', 'std', 'min', 'max'],
            'test_f1_macro': ['mean', 'std']
        }).reset_index()
        summary_a.columns = ['format', 'quality', 'acc_mean', 'acc_std', 'acc_min', 'acc_max', 'f1_mean', 'f1_std']
        summary_a.to_csv(output_dir / 'experiment_a_summary.csv', index=False)
        print(f"Summary table saved: {output_dir / 'experiment_a_summary.csv'}")

    # Summary table for Experiment B
    if 'arcade_exp_b' in results:
        df = results['arcade_exp_b']
        summary_b = df.groupby(['format', 'test_quality']).agg({
            'test_accuracy': ['mean', 'std', 'min', 'max'],
            'test_f1_macro': ['mean', 'std']
        }).reset_index()
        summary_b.columns = ['format', 'quality', 'acc_mean', 'acc_std', 'acc_min', 'acc_max', 'f1_mean', 'f1_std']
        summary_b.to_csv(output_dir / 'experiment_b_summary.csv', index=False)
        print(f"Summary table saved: {output_dir / 'experiment_b_summary.csv'}")

    # Combined format comparison
    if 'arcade_exp_a' in results and 'arcade_exp_b' in results:
        combined = []

        for fmt in ['jpeg', 'jpeg2000', 'avif']:
            df_a = results['arcade_exp_a'][results['arcade_exp_a']['format'] == fmt]
            df_b = results['arcade_exp_b'][results['arcade_exp_b']['format'] == fmt]

            for q in [100, 85, 70, 50, 30, 10]:
                row = {'format': fmt, 'quality': q}

                # Experiment A
                exp_a_q = df_a[df_a['train_quality'] == q]
                if not exp_a_q.empty:
                    row['exp_a_acc_mean'] = exp_a_q['test_accuracy'].mean()
                    row['exp_a_acc_std'] = exp_a_q['test_accuracy'].std()

                # Experiment B
                exp_b_q = df_b[df_b['test_quality'] == q]
                if not exp_b_q.empty:
                    row['exp_b_acc_mean'] = exp_b_q['test_accuracy'].mean()
                    row['exp_b_acc_std'] = exp_b_q['test_accuracy'].std()

                combined.append(row)

        combined_df = pd.DataFrame(combined)
        combined_df.to_csv(output_dir / 'combined_format_comparison.csv', index=False)
        print(f"Combined table saved: {output_dir / 'combined_format_comparison.csv'}")


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
        print("Please run experiments first:")
        print("  python src/experiment_a.py --model resnet50 --task syntax")
        print("  python src/experiment_b.py --model resnet50 --task syntax")
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

    if 'arcade_exp_b' in results:
        generate_latex_table_experiment_b(
            results['arcade_exp_b'],
            args.model,
            args.task,
            output_path=(output_dir / 'tables' / 'experiment_b_table.tex') if output_dir else None
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
