"""
Statistical analysis module for compression experiments - CORRECTED VERSION.

Provides:
- Paired t-tests between compression formats (corrected for experiment B)
- Independent t-tests (for experiment A)
- ANOVA for comparing all formats
- Effect sizes (Cohen's d)
- Statistical report generation

CORRECTIONS MADE:
- Added paired_t_test for experiment B (same model, different quality levels)
- Added wilcoxon_signed_rank_test for non-parametric paired analysis
- Corrected Mann-Whitney U to Wilcoxon signed-rank for paired data
- Improved documentation and experimental design guidance

Author: Master's Thesis Project
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


class StatisticalAnalyzer:
    """
    Statistical analyzer for compression experiment results.

    Provides methods for comparing performance metrics between different
    compression formats (JPEG, JPEG2000, AVIF) using various statistical tests.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical analyzer.

        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha
        self.formats = config.COMPRESSION_FORMATS  # ['jpeg', 'jpeg2000', 'avif']

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size between two groups.

        Cohen's d interpretation:
        - |d| < 0.2: negligible effect
        - 0.2 <= |d| < 0.5: small effect
        - 0.5 <= |d| < 0.8: medium effect
        - |d| >= 0.8: large effect

        Args:
            group1: First group of observations
            group2: Second group of observations

        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Use epsilon comparison instead of == 0 for floating point safety
        EPSILON = 1e-10
        if pooled_std < EPSILON:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def cohens_d_paired(self, differences: np.ndarray) -> float:
        """
        Calculate Cohen's d for paired samples.

        Uses the standard deviation of the differences as the denominator.

        Args:
            differences: Array of paired differences (group1 - group2)

        Returns:
            Cohen's d effect size for paired samples
        """
        std_diff = np.std(differences, ddof=1)
        mean_diff = np.mean(differences)

        EPSILON = 1e-10
        if std_diff < EPSILON:
            return 0.0

        return mean_diff / std_diff

    def interpret_cohens_d(self, d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def independent_t_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        format1: str = "Format1",
        format2: str = "Format2"
    ) -> Dict:
        """
        Perform independent samples t-test between two groups.

        Use this for EXPERIMENT A: Different models trained on different quality levels.

        Note: We use Welch's t-test (does not assume equal variances).

        Args:
            data1: First group of observations (e.g., accuracy scores for JPEG)
            data2: Second group of observations (e.g., accuracy scores for JPEG2000)
            format1: Name of first format
            format2: Name of second format

        Returns:
            Dictionary with test results
        """
        # Use Welch's t-test (does not assume equal variances)
        t_statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        effect_size = self.cohens_d(data1, data2)

        return {
            'test': 'independent_t_test',
            'comparison': f"{format1} vs {format2}",
            'format1': format1,
            'format2': format2,
            'mean_diff': float(np.mean(data1) - np.mean(data2)),
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'cohens_d': float(effect_size),
            'effect_interpretation': self.interpret_cohens_d(effect_size),
            'n_samples_1': len(data1),
            'n_samples_2': len(data2)
        }

    def paired_t_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        format1: str = "Format1",
        format2: str = "Format2"
    ) -> Dict:
        """
        Perform paired samples t-test between two groups.

        Use this for EXPERIMENT B: Same model tested on different quality levels.
        The data are PAIRED because they come from the same test images.

        Note: Assumes data1 and data2 are aligned (same order of test images).

        Args:
            data1: First group of observations (e.g., accuracy scores for Q=10)
            data2: Second group of observations (e.g., accuracy scores for Q=20)
            format1: Name of first condition
            format2: Name of second condition

        Returns:
            Dictionary with test results
        """
        # Calculate paired differences
        differences = np.array(data1) - np.array(data2)

        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(data1, data2)

        # Effect size for paired samples
        effect_size = self.cohens_d_paired(differences)

        return {
            'test': 'paired_t_test',
            'comparison': f"{format1} vs {format2}",
            'format1': format1,
            'format2': format2,
            'mean_diff': float(np.mean(differences)),
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'cohens_d': float(effect_size),
            'effect_interpretation': self.interpret_cohens_d(effect_size),
            'n_samples': len(data1),
            'mean_diff_std': float(np.std(differences, ddof=1))
        }

    def anova_test(
        self,
        groups: Dict[str, np.ndarray],
        metric_name: str = "metric"
    ) -> Dict:
        """
        Perform one-way ANOVA test across multiple groups.

        Args:
            groups: Dictionary mapping format names to observation arrays
            metric_name: Name of the metric being compared

        Returns:
            Dictionary with ANOVA results
        """
        group_arrays = list(groups.values())
        group_names = list(groups.keys())

        f_statistic, p_value = stats.f_oneway(*group_arrays)

        # Calculate effect size (eta-squared)
        all_data = np.concatenate(group_arrays)
        grand_mean = np.mean(all_data)

        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_arrays)
        ss_total = sum(np.sum((g - grand_mean) ** 2) for g in group_arrays)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        return {
            'test': 'one_way_anova',
            'metric': metric_name,
            'groups': group_names,
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'eta_squared': float(eta_squared),
            'eta_interpretation': self.interpret_eta_squared(eta_squared),
            'group_means': {name: float(np.mean(data)) for name, data in groups.items()},
            'group_stds': {name: float(np.std(data, ddof=1)) for name, data in groups.items()},
            'n_groups': len(groups)
        }

    def interpret_eta_squared(self, eta: float) -> str:
        """
        Interpret eta-squared effect size.

        Args:
            eta: Eta-squared value

        Returns:
            Interpretation string
        """
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"

    def wilcoxon_signed_rank_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        format1: str = "Format1",
        format2: str = "Format2"
    ) -> Dict:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Use this for EXPERIMENT B when data are not normally distributed.

        Args:
            data1: First group of observations
            data2: Second group of observations
            format1: Name of first format
            format2: Name of second format

        Returns:
            Dictionary with test results
        """
        statistic, p_value = stats.wilcoxon(data1, data2)

        # Rank-biserial correlation as effect size
        n = len(data1)
        z_score = (statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        r = abs(z_score) / np.sqrt(n)

        return {
            'test': 'wilcoxon_signed_rank',
            'comparison': f"{format1} vs {format2}",
            'format1': format1,
            'format2': format2,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'effect_size_r': float(r),
            'effect_interpretation': self.interpret_effect_size_r(r),
            'n_samples': n
        }

    def interpret_effect_size_r(self, r: float) -> str:
        """
        Interpret rank-biserial correlation effect size.

        Args:
            r: Effect size r value

        Returns:
            Interpretation string
        """
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"

    def mann_whitney_u_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        format1: str = "Format1",
        format2: str = "Format2"
    ) -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric alternative to independent t-test).

        Use this for EXPERIMENT A when data are not normally distributed.

        Args:
            data1: First group of observations
            data2: Second group of observations
            format1: Name of first format
            format2: Name of second format

        Returns:
            Dictionary with test results
        """
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

        # Rank-biserial correlation as effect size
        n1, n2 = len(data1), len(data2)
        u1 = statistic
        u2 = n1 * n2 - u1
        r = 1 - (2 * min(u1, u2)) / (n1 * n2)

        return {
            'test': 'mann_whitney_u',
            'comparison': f"{format1} vs {format2}",
            'format1': format1,
            'format2': format2,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'effect_size_r': float(r),
            'effect_interpretation': self.interpret_effect_size_r(r),
            'n_samples_1': n1,
            'n_samples_2': n2
        }

    def kruskal_wallis_test(
        self,
        groups: Dict[str, np.ndarray],
        metric_name: str = "metric"
    ) -> Dict:
        """
        Perform Kruskal-Wallis H-test (non-parametric alternative to ANOVA).

        Args:
            groups: Dictionary mapping format names to observation arrays
            metric_name: Name of the metric being compared

        Returns:
            Dictionary with test results
        """
        group_arrays = list(groups.values())
        group_names = list(groups.keys())

        h_statistic, p_value = stats.kruskal(*group_arrays)

        # Epsilon squared effect size for Kruskal-Wallis
        n_total = sum(len(g) for g in group_arrays)
        epsilon_squared = (h_statistic - len(group_arrays) + 1) / (n_total - len(group_arrays))

        return {
            'test': 'kruskal_wallis',
            'metric': metric_name,
            'groups': group_names,
            'h_statistic': float(h_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'epsilon_squared': float(max(0, epsilon_squared)),
            'group_medians': {name: float(np.median(data)) for name, data in groups.items()},
            'n_groups': len(groups)
        }

    def bonferroni_correction(
        self,
        p_values: List[float]
    ) -> Tuple[List[float], List[bool]]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values from multiple tests

        Returns:
            Tuple of (adjusted p-values, significance flags)
        """
        n_tests = len(p_values)
        adjusted_alpha = self.alpha / n_tests
        adjusted_p = [min(p * n_tests, 1.0) for p in p_values]
        significant = [p < self.alpha for p in adjusted_p]

        return adjusted_p, significant

    def holm_bonferroni_correction(
        self,
        p_values: List[float]
    ) -> Tuple[List[float], List[bool]]:
        """
        Apply Holm-Bonferroni correction (more powerful than Bonferroni).

        Args:
            p_values: List of p-values from multiple tests

        Returns:
            Tuple of (adjusted p-values, significance flags in original order)
        """
        n = len(p_values)
        indexed_p = [(p, i) for i, p in enumerate(p_values)]
        indexed_p.sort(key=lambda x: x[0])  # Sort by p-value

        adjusted = [0.0] * n
        significant = [False] * n

        for rank, (p, original_idx) in enumerate(indexed_p):
            adjusted_p = p * (n - rank)
            adjusted[original_idx] = min(adjusted_p, 1.0)
            significant[original_idx] = adjusted_p < self.alpha

        return adjusted, significant


def load_experiment_results(
    experiment_type: str,
    model_name: str = "resnet50",
    task: str = "syntax"
) -> pd.DataFrame:
    """
    Load experiment results from CSV files.

    Args:
        experiment_type: "experiment_a" or "experiment_b"
        model_name: Model architecture name
        task: Dataset task name

    Returns:
        Combined DataFrame with results from all formats
    """
    results_dir = config.RESULTS_ROOT / experiment_type
    frames = []

    for fmt in config.COMPRESSION_FORMATS:
        csv_path = results_dir / f"{model_name}_{task}_{fmt}_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No results found in {results_dir}")

    return pd.concat(frames, ignore_index=True)


def analyze_format_comparison(
    df: pd.DataFrame,
    metric_column: str = "test_accuracy",
    group_column: str = "format",
    experiment_type: str = "experiment_a"
) -> Dict:
    """
    Perform comprehensive statistical comparison between formats.

    IMPORTANT: Uses different tests for Experiment A vs B:
    - Experiment A: Independent t-test (different models, different data)
    - Experiment B: Paired t-test (same model, paired data)

    Args:
        df: DataFrame with experiment results
        metric_column: Column name for the metric to compare
        group_column: Column name for grouping (format)
        experiment_type: "experiment_a" or "experiment_b"

    Returns:
        Dictionary with all statistical test results
    """
    analyzer = StatisticalAnalyzer()

    # Group data by format
    groups = {
        fmt: df[df[group_column] == fmt][metric_column].values
        for fmt in config.COMPRESSION_FORMATS
        if fmt in df[group_column].values
    }

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}

    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for comparison")

    results = {
        'metric': metric_column,
        'experiment_type': experiment_type,
        'analysis_date': datetime.now().isoformat(),
        'alpha': 0.05,
        'descriptive_stats': {},
        'anova': None,
        'kruskal_wallis': None,
        'pairwise_tests': [],
        'pairwise_wilcoxon': []
    }

    # Descriptive statistics
    for fmt, data in groups.items():
        results['descriptive_stats'][fmt] = {
            'n': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data, ddof=1)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }

    # ANOVA test (parametric)
    if len(groups) >= 2:
        results['anova'] = analyzer.anova_test(groups, metric_column)

    # Kruskal-Wallis test (non-parametric)
    if len(groups) >= 2:
        results['kruskal_wallis'] = analyzer.kruskal_wallis_test(groups, metric_column)

    # Pairwise comparisons - USE CORRECT TEST BASED ON EXPERIMENT TYPE
    format_list = list(groups.keys())
    p_values_ttest = []
    p_values_wilcoxon = []

    for i, fmt1 in enumerate(format_list):
        for fmt2 in format_list[i + 1:]:
            # Choose test based on experiment type
            if experiment_type == "experiment_a":
                # Experiment A: Different models -> Independent t-test
                ttest_result = analyzer.independent_t_test(
                    groups[fmt1], groups[fmt2], fmt1, fmt2
                )
                # Mann-Whitney U as non-parametric alternative
                wilcoxon_result = analyzer.mann_whitney_u_test(
                    groups[fmt1], groups[fmt2], fmt1, fmt2
                )
            else:
                # Experiment B: Same model, paired data -> Paired t-test
                ttest_result = analyzer.paired_t_test(
                    groups[fmt1], groups[fmt2], fmt1, fmt2
                )
                # Wilcoxon signed-rank as non-parametric alternative
                wilcoxon_result = analyzer.wilcoxon_signed_rank_test(
                    groups[fmt1], groups[fmt2], fmt1, fmt2
                )

            results['pairwise_tests'].append(ttest_result)
            p_values_ttest.append(ttest_result['p_value'])
            results['pairwise_wilcoxon'].append(wilcoxon_result)
            p_values_wilcoxon.append(wilcoxon_result['p_value'])

    # Apply corrections for multiple comparisons
    if p_values_ttest:
        adjusted_p, significant = analyzer.bonferroni_correction(p_values_ttest)
        results['bonferroni_correction_ttest'] = {
            'original_p_values': p_values_ttest,
            'adjusted_p_values': adjusted_p,
            'significant': significant
        }

    if p_values_wilcoxon:
        adjusted_p, significant = analyzer.holm_bonferroni_correction(p_values_wilcoxon)
        results['holm_bonferroni_correction_wilcoxon'] = {
            'original_p_values': p_values_wilcoxon,
            'adjusted_p_values': adjusted_p,
            'significant': significant
        }

    return results


def generate_statistical_report(
    results: Dict,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a human-readable statistical report.

    Args:
        results: Dictionary with statistical analysis results
        output_path: Optional path to save the report

    Returns:
        Report as string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"\nAnalysis Date: {results.get('analysis_date', 'N/A')}")
    lines.append(f"Experiment Type: {results.get('experiment_type', 'N/A').upper()}")
    lines.append(f"Metric Analyzed: {results.get('metric', 'N/A')}")
    lines.append(f"Significance Level (alpha): {results.get('alpha', 0.05)}")

    # Add note about test selection
    exp_type = results.get('experiment_type', 'experiment_a')
    if exp_type == 'experiment_a':
        lines.append("\nNote: Using INDEPENDENT samples tests (different models trained separately)")
    else:
        lines.append("\nNote: Using PAIRED samples tests (same model, different quality levels)")

    # Descriptive Statistics
    lines.append("\n" + "-" * 40)
    lines.append("DESCRIPTIVE STATISTICS")
    lines.append("-" * 40)

    for fmt, stats_data in results.get('descriptive_stats', {}).items():
        lines.append(f"\n{fmt.upper()}:")
        lines.append(f"  N = {stats_data['n']}")
        lines.append(f"  Mean = {stats_data['mean']:.4f}")
        lines.append(f"  Std = {stats_data['std']:.4f}")
        lines.append(f"  Median = {stats_data['median']:.4f}")
        lines.append(f"  Range = [{stats_data['min']:.4f}, {stats_data['max']:.4f}]")

    # ANOVA Results
    if results.get('anova'):
        anova = results['anova']
        lines.append("\n" + "-" * 40)
        lines.append("ONE-WAY ANOVA")
        lines.append("-" * 40)
        lines.append(f"F-statistic: {anova['f_statistic']:.4f}")
        lines.append(f"p-value: {anova['p_value']:.6f}")
        lines.append(f"Significant: {'YES' if anova['significant'] else 'NO'}")
        lines.append(f"Effect size (eta-squared): {anova['eta_squared']:.4f}")
        lines.append(f"Effect interpretation: {anova['eta_interpretation']}")

        lines.append("\nGroup Means:")
        for fmt, mean in anova['group_means'].items():
            lines.append(f"  {fmt}: {mean:.4f}")

    # Kruskal-Wallis Results
    if results.get('kruskal_wallis'):
        kw = results['kruskal_wallis']
        lines.append("\n" + "-" * 40)
        lines.append("KRUSKAL-WALLIS TEST (Non-parametric)")
        lines.append("-" * 40)
        lines.append(f"H-statistic: {kw['h_statistic']:.4f}")
        lines.append(f"p-value: {kw['p_value']:.6f}")
        lines.append(f"Significant: {'YES' if kw['significant'] else 'NO'}")
        lines.append(f"Effect size (epsilon-squared): {kw['epsilon_squared']:.4f}")

        lines.append("\nGroup Medians:")
        for fmt, median in kw['group_medians'].items():
            lines.append(f"  {fmt}: {median:.4f}")

    # Pairwise Comparisons (t-test)
    if results.get('pairwise_tests'):
        test_name = "PAIRED" if exp_type == 'experiment_b' else "INDEPENDENT"
        lines.append("\n" + "-" * 40)
        lines.append(f"PAIRWISE COMPARISONS ({test_name} t-test)")
        lines.append("-" * 40)

        for test in results['pairwise_tests']:
            lines.append(f"\n{test['comparison']}:")
            lines.append(f"  Mean difference: {test['mean_diff']:.4f}")
            lines.append(f"  t-statistic: {test['t_statistic']:.4f}")
            lines.append(f"  p-value: {test['p_value']:.6f}")
            lines.append(f"  Significant: {'YES' if test['significant'] else 'NO'}")
            lines.append(f"  Cohen's d: {test['cohens_d']:.4f} ({test['effect_interpretation']})")

    # Pairwise Comparisons (Non-parametric)
    if results.get('pairwise_wilcoxon'):
        test_name = "Wilcoxon Signed-Rank" if exp_type == 'experiment_b' else "Mann-Whitney U"
        lines.append("\n" + "-" * 40)
        lines.append(f"PAIRWISE COMPARISONS ({test_name}, Non-parametric)")
        lines.append("-" * 40)

        for test in results['pairwise_wilcoxon']:
            lines.append(f"\n{test['comparison']}:")
            lines.append(f"  Statistic: {test['statistic']:.4f}")
            lines.append(f"  p-value: {test['p_value']:.6f}")
            lines.append(f"  Significant: {'YES' if test['significant'] else 'NO'}")
            if 'effect_size_r' in test:
                lines.append(f"  Effect size r: {test['effect_size_r']:.4f} ({test['effect_interpretation']})")

    # Multiple Comparison Corrections
    if results.get('bonferroni_correction_ttest'):
        bonf = results['bonferroni_correction_ttest']
        lines.append("\n" + "-" * 40)
        lines.append("BONFERRONI CORRECTION (t-test)")
        lines.append("-" * 40)
        lines.append(f"Original p-values: {[f'{p:.6f}' for p in bonf['original_p_values']]}")
        lines.append(f"Adjusted p-values: {[f'{p:.6f}' for p in bonf['adjusted_p_values']]}")
        lines.append(f"Significant after correction: {bonf['significant']}")

    # Summary
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)

    anova_sig = results.get('anova', {}).get('significant', False) if results.get('anova') else False
    kw_sig = results.get('kruskal_wallis', {}).get('significant', False) if results.get('kruskal_wallis') else False

    if anova_sig:
        lines.append("\nANOVA detected significant differences between formats (p < 0.05)")
    else:
        lines.append("\nANOVA did not detect significant differences between formats")

    if kw_sig:
        lines.append("Kruskal-Wallis test confirms significant differences (non-parametric)")
    else:
        lines.append("Kruskal-Wallis test did not detect significant differences")

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def run_full_statistical_analysis(
    experiment_type: str = "experiment_a",
    model_name: str = "resnet50",
    task: str = "syntax",
    metrics: List[str] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run complete statistical analysis on experiment results.

    Args:
        experiment_type: "experiment_a" or "experiment_b"
        model_name: Model architecture name
        task: Dataset task name
        metrics: List of metrics to analyze
        output_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    if metrics is None:
        metrics = ['test_accuracy', 'test_f1_macro']

    if output_dir is None:
        output_dir = config.RESULTS_ROOT / "statistical_analysis"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {experiment_type} results...")
    df = load_experiment_results(experiment_type, model_name, task)
    print(f"Loaded {len(df)} records")

    all_results = {
        'experiment_type': experiment_type,
        'model': model_name,
        'task': task,
        'analysis_date': datetime.now().isoformat(),
        'metrics': {}
    }

    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in data, skipping...")
            continue

        print(f"\nAnalyzing metric: {metric}")
        results = analyze_format_comparison(
            df,
            metric_column=metric,
            experiment_type=experiment_type
        )

        all_results['metrics'][metric] = results

        # Generate report
        report = generate_statistical_report(
            results,
            output_path=output_dir / f"report_{experiment_type}_{metric}.txt"
        )

        # Save JSON results
        json_path = output_dir / f"results_{experiment_type}_{metric}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {json_path}")

    # Save combined results
    combined_path = output_dir / f"combined_analysis_{experiment_type}.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to: {combined_path}")

    return all_results


def main():
    """Command-line interface for statistical analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Statistical analysis for compression experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment_a",
        choices=["experiment_a", "experiment_b"],
        help="Experiment type to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model architecture"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="syntax",
        choices=["syntax", "stenosis"],
        help="Dataset task"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["test_accuracy", "test_f1_macro"],
        help="Metrics to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_full_statistical_analysis(
        experiment_type=args.experiment,
        model_name=args.model,
        task=args.task,
        metrics=args.metrics,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
