"""
Statistical analysis module for compression experiments (Experiment A).

Provides:
- Paired/blocked statistical comparison of compression formats
- Friedman omnibus test (formats = treatments, quality levels = blocks)
- Kendall's W effect size for the Friedman test
- Pairwise paired t-test and Wilcoxon signed-rank tests
- Effect sizes (paired Cohen's d)
- Holm-Bonferroni multiple-comparison correction
- Baseline (uncompressed reference) loading and reporting
- Statistical report generation

METHODOLOGICAL NOTE (paired/blocked design):
In Experiment A every CSV row corresponds to one shared quality level Q
(config.QUALITY_LEVELS). The 13 Q levels are NOT exchangeable IID observations
of a "format population" - they form a systematic quality gradient and the SAME
Q grid is used for JPEG / JPEG2000 / AVIF. Quality is therefore a BLOCKING
factor and the observations are PAIRED across formats. The analysis below
follows the statistically correct paired/blocked design:
  * Omnibus comparison: Friedman test (formats = treatments, Q levels = blocks)
    with Kendall's W as the effect size.
  * Pairwise comparisons: paired t-test and Wilcoxon signed-rank test, computed
    on vectors paired by Q level, with paired Cohen's d as the effect size.
  * Multiple-comparison control: Holm-Bonferroni over the 3 format pairs.
The unpaired tests previously used (independent t-test, one-way ANOVA,
Mann-Whitney U, Kruskal-Wallis) did NOT match the study design and have been
removed.

Remaining important caveats (to keep in mind when interpreting results):
  * n = 1 observation per (format, Q) cell. Each cell is a single trained model
    evaluated once; there is no within-cell replication, so the analysis treats
    each (format, Q) pair as a single point and the only source of variability
    is the quality gradient itself.
  * "JP2 floor": for JPEG2000 the highest quality levels (Q = 100 / 95 / 90)
    often produce (near-)identical results because the codec hits a
    near-lossless floor. Such ties create zero or near-zero paired differences
    that REDUCE the statistical power of the Wilcoxon signed-rank test (and can
    make it undefined when ALL differences are zero).
  * Baseline (uncompressed PNG) is a single reference value; it is reported for
    context but is NOT entered as a treatment group into any test.
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

# mAP (test_map) is a THRESHOLD-FREE ranking metric and the standard choice for
# multi-label classification under strong class imbalance -> it is the LEADING
# metric. F1 at the fixed 0.5 threshold is auxiliary / supporting only.
DEFAULT_METRICS = ['test_map', 'test_f1_macro', 'test_hamming_accuracy']

# Metrics treated as PRIMARY (threshold-free, leading) vs SECONDARY (computed at
# the fixed 0.5 threshold and therefore systematically sub-optimal under
# pos_weight up to ~999 - interpret with caution; see KWESTIE pkt 2).
PRIMARY_METRICS = {'test_map'}

# Equivalence margins (Delta) for the TOST equivalence test, on the metric scale.
# Primary margin: Delta = 0.02 mAP (~2x the single-seed training-noise floor,
# sigma ~ 0.007-0.012; a <=2 pp mAP difference is below what the n=1 setup can
# attribute to format rather than seed, and is practically negligible for
# decision-support). 0.01 and 0.015 are reported as a sensitivity ladder.
TOST_DELTAS = [0.01, 0.015, 0.02]
TOST_PRIMARY_DELTA = 0.02


class StatisticalAnalyzer:
    """
    Statistical analyzer for compression experiment results.

    Provides methods for comparing performance metrics between compression
    formats (JPEG, JPEG2000, AVIF) using a PAIRED / BLOCKED design: quality
    levels are treated as blocks shared by every format.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical analyzer.

        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha

    # ------------------------------------------------------------------ #
    # Effect-size helpers
    # ------------------------------------------------------------------ #
    def paired_cohens_d(self, diffs: np.ndarray) -> float:
        """
        Calculate the paired Cohen's d effect size from paired differences.

        Paired Cohen's d = mean(diffs) / std(diffs, ddof=1).

        Interpretation:
        - |d| < 0.2: negligible effect
        - 0.2 <= |d| < 0.5: small effect
        - 0.5 <= |d| < 0.8: medium effect
        - |d| >= 0.8: large effect

        Args:
            diffs: Array of paired differences (format1 - format2 per block)

        Returns:
            Paired Cohen's d effect size (0.0 if it is undefined)
        """
        diffs = np.asarray(diffs, dtype=float)
        # Need at least 2 paired observations for an unbiased std (ddof=1).
        if len(diffs) < 2:
            return 0.0

        sd = np.std(diffs, ddof=1)

        # Guard against division by zero: when every paired difference is
        # identical (sd == 0) the effect size is undefined. Use an epsilon
        # comparison for floating-point safety.
        EPSILON = 1e-10
        if sd < EPSILON:
            return 0.0

        return float(np.mean(diffs) / sd)

    def interpret_cohens_d(self, d: float) -> str:
        """
        Interpret a (paired) Cohen's d effect size.

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if np.isnan(abs_d):
            return "undefined"
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def interpret_kendalls_w(self, w: float) -> str:
        """
        Interpret Kendall's W (coefficient of concordance / Friedman effect size).

        W ranges in [0, 1]: 0 = no agreement among blocks on the format ranking,
        1 = complete agreement.

        Args:
            w: Kendall's W value

        Returns:
            Interpretation string
        """
        if np.isnan(w):
            return "undefined"
        if w < 0.1:
            return "negligible"
        elif w < 0.3:
            return "small"
        elif w < 0.5:
            return "medium"
        else:
            return "large"

    # ------------------------------------------------------------------ #
    # Omnibus test: Friedman (paired / blocked design)
    # ------------------------------------------------------------------ #
    def friedman_test(
        self,
        paired_matrix: pd.DataFrame,
        metric_name: str = "metric"
    ) -> Dict:
        """
        Perform the Friedman test on a paired [block x treatment] matrix.

        The Friedman test is the omnibus (non-parametric, repeated-measures)
        test for the paired/blocked design: the compression FORMATS are the
        treatments and the shared quality levels Q are the blocks.

        Effect size: Kendall's W = friedman_chi2 / (n_blocks * (k - 1)), where
        n_blocks is the number of paired Q levels and k the number of formats.

        Args:
            paired_matrix: DataFrame indexed by quality level (blocks), one
                column per format (treatments). Must be fully paired (no NaN).
            metric_name: Name of the metric being compared

        Returns:
            Dictionary with the Friedman test results. If the test cannot be
            run (fewer than 3 formats, or fewer than 3 paired blocks) the
            returned dict carries a 'warning' and NaN statistics.
        """
        formats = list(paired_matrix.columns)
        n_blocks = int(paired_matrix.shape[0])
        k = len(formats)

        base = {
            'test': 'friedman',
            'metric': metric_name,
            'groups': formats,
            'n_formats': k,
            'n_blocks': n_blocks,
            'chi2_statistic': float('nan'),
            'p_value': float('nan'),
            'significant': False,
            'kendalls_w': float('nan'),
            'w_interpretation': 'undefined',
        }

        # Friedman requires at least 3 treatments (formats).
        if k < 3:
            base['warning'] = (
                f"Friedman omnibus test skipped: needs >= 3 formats, got {k}."
            )
            return base

        # Friedman requires at least 3 blocks (paired quality levels).
        if n_blocks < 3:
            base['warning'] = (
                f"Friedman omnibus test skipped: needs >= 3 paired quality "
                f"levels (blocks), got {n_blocks}."
            )
            return base

        columns = [paired_matrix[fmt].to_numpy(dtype=float) for fmt in formats]
        chi2, p_value = stats.friedmanchisquare(*columns)

        # Kendall's W (coefficient of concordance) as the effect size.
        denom = n_blocks * (k - 1)
        kendalls_w = float(chi2) / denom if denom > 0 else float('nan')

        base.update({
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'kendalls_w': float(kendalls_w),
            'w_interpretation': self.interpret_kendalls_w(kendalls_w),
        })
        return base

    # ------------------------------------------------------------------ #
    # Pairwise tests: paired t-test and Wilcoxon signed-rank
    # ------------------------------------------------------------------ #
    def paired_t_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        format1: str = "Format1",
        format2: str = "Format2"
    ) -> Dict:
        """
        Perform a paired-samples t-test between two formats.

        The two input vectors must be aligned by block (quality level): the
        i-th element of each vector is the metric for the same Q level.

        Args:
            data1: Metric values for format1, ordered by shared Q level
            data2: Metric values for format2, ordered by shared Q level
            format1: Name of first format
            format2: Name of second format

        Returns:
            Dictionary with the paired t-test results
        """
        data1 = np.asarray(data1, dtype=float)
        data2 = np.asarray(data2, dtype=float)
        n = len(data1)

        # Need >= 3 paired observations for a meaningful paired t-test.
        if n < 3 or len(data2) != n:
            return {
                'test': 'paired_t_test',
                'comparison': f"{format1} vs {format2}",
                'format1': format1,
                'format2': format2,
                'mean_diff': (float(np.mean(data1 - data2))
                              if n and len(data2) == n else float('nan')),
                't_statistic': float('nan'),
                'p_value': float('nan'),
                'significant': False,
                'cohens_d': float('nan'),
                'effect_interpretation': 'undefined',
                'n_pairs': n,
                'warning': 'insufficient paired sample size (need >= 3 paired Q levels)'
            }

        diffs = data1 - data2
        t_statistic, p_value = stats.ttest_rel(data1, data2)
        effect_size = self.paired_cohens_d(diffs)

        result = {
            'test': 'paired_t_test',
            'comparison': f"{format1} vs {format2}",
            'format1': format1,
            'format2': format2,
            'mean_diff': float(np.mean(diffs)),
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'cohens_d': float(effect_size),
            'effect_interpretation': self.interpret_cohens_d(effect_size),
            'n_pairs': n,
        }

        # ttest_rel returns NaN when every paired difference is identical
        # (zero variance, e.g. the "JP2 floor"). Flag it explicitly.
        if np.isnan(p_value):
            result['warning'] = (
                'paired t-test undefined (zero variance of paired '
                'differences, e.g. identical results across quality levels)'
            )
        return result

    def wilcoxon_signed_rank_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        format1: str = "Format1",
        format2: str = "Format2"
    ) -> Dict:
        """
        Perform a Wilcoxon signed-rank test between two formats.

        Non-parametric counterpart of the paired t-test. The two input vectors
        must be aligned by block (quality level).

        Edge case handled: when ALL paired differences are zero (possible via
        the "JP2 floor" - Q = 100/95/90 giving identical JPEG2000 results)
        scipy.stats.wilcoxon raises a ValueError. It is caught here and a
        result with p_value = NaN plus a warning is returned.

        Args:
            data1: Metric values for format1, ordered by shared Q level
            data2: Metric values for format2, ordered by shared Q level
            format1: Name of first format
            format2: Name of second format

        Returns:
            Dictionary with the Wilcoxon signed-rank test results
        """
        data1 = np.asarray(data1, dtype=float)
        data2 = np.asarray(data2, dtype=float)
        n = len(data1)

        base = {
            'test': 'wilcoxon_signed_rank',
            'comparison': f"{format1} vs {format2}",
            'format1': format1,
            'format2': format2,
            'statistic': float('nan'),
            'p_value': float('nan'),
            'significant': False,
            'n_pairs': n,
        }

        # Need >= 3 paired observations for the test to be meaningful.
        if n < 3 or len(data2) != n:
            base['warning'] = (
                'insufficient paired sample size (need >= 3 paired Q levels)'
            )
            return base

        try:
            statistic, p_value = stats.wilcoxon(data1, data2)
        except ValueError as exc:
            # Raised when all paired differences are zero (the "JP2 floor"
            # ties) - the signed-rank test is undefined in that case.
            base['warning'] = (
                f'Wilcoxon signed-rank test undefined: {exc} '
                '(all paired differences are zero, e.g. JP2 floor ties)'
            )
            return base

        base.update({
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
        })
        return base

    def paired_tost(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        delta: float,
        format1: str = "Format1",
        format2: str = "Format2"
    ) -> Dict:
        """
        Two One-Sided Tests (TOST) for paired-samples EQUIVALENCE.

        A non-significant difference test (e.g. Friedman p>0.05) only means
        "no evidence of a difference" - it cannot establish equivalence. TOST
        tests the positive claim that the true mean difference lies within an
        equivalence margin (-delta, +delta): it runs two one-sided paired
        t-tests and takes the LARGER p-value. p < alpha => formats are
        statistically equivalent within +/-delta.

        Args:
            data1, data2: metric values aligned by block (quality level).
            delta: equivalence margin on the metric scale (e.g. 0.01 = 1 pp mAP).
            format1, format2: format names.

        Returns:
            Dict with both one-sided p-values, the TOST p-value (their max),
            the equivalence verdict, and the 90% CI of the mean difference.
        """
        data1 = np.asarray(data1, dtype=float)
        data2 = np.asarray(data2, dtype=float)
        n = len(data1)
        diffs = data1 - data2

        base = {
            'test': 'paired_tost',
            'comparison': f"{format1} vs {format2}",
            'delta': float(delta),
            'mean_diff': float(np.mean(diffs)) if n else float('nan'),
            'p_lower': float('nan'),
            'p_upper': float('nan'),
            'p_tost': float('nan'),
            'equivalent': False,
            'ci90': [float('nan'), float('nan')],
            'n_pairs': n,
        }

        if n < 3 or len(data2) != n:
            base['warning'] = 'insufficient paired sample size (need >= 3 paired Q levels)'
            return base

        sd = np.std(diffs, ddof=1)
        if sd < 1e-10:
            # Zero variance of differences: equivalence holds iff the (constant)
            # difference is strictly inside the margin.
            inside = abs(np.mean(diffs)) < delta
            base.update({
                'p_lower': 0.0 if inside else 1.0,
                'p_upper': 0.0 if inside else 1.0,
                'p_tost': 0.0 if inside else 1.0,
                'equivalent': bool(inside),
                'ci90': [float(np.mean(diffs)), float(np.mean(diffs))],
                'warning': 'zero variance of paired differences',
            })
            return base

        se = sd / np.sqrt(n)
        df = n - 1
        mean_diff = np.mean(diffs)
        # H0_lower: diff <= -delta  (one-sided, expect diff > -delta)
        t_lower = (mean_diff + delta) / se
        p_lower = stats.t.sf(t_lower, df)   # P(T > t_lower)
        # H0_upper: diff >= +delta  (one-sided, expect diff < +delta)
        t_upper = (mean_diff - delta) / se
        p_upper = stats.t.cdf(t_upper, df)  # P(T < t_upper)
        p_tost = max(p_lower, p_upper)

        # 90% CI of the mean difference (matches the alpha=0.05 TOST procedure).
        tcrit = stats.t.ppf(1 - self.alpha, df)
        ci90 = [float(mean_diff - tcrit * se), float(mean_diff + tcrit * se)]

        base.update({
            'p_lower': float(p_lower),
            'p_upper': float(p_upper),
            'p_tost': float(p_tost),
            'equivalent': bool(p_tost < self.alpha),
            'ci90': ci90,
        })
        return base

    # ------------------------------------------------------------------ #
    # Multiple-comparison corrections
    # ------------------------------------------------------------------ #
    def bonferroni_correction(
        self,
        p_values: List[float]
    ) -> Tuple[List[float], List[bool]]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Kept as a utility; the main analysis path uses Holm-Bonferroni.

        Args:
            p_values: List of p-values from multiple tests

        Returns:
            Tuple of (adjusted p-values, significance flags)
        """
        n_tests = len(p_values)
        if n_tests == 0:
            return [], []

        # Standard "adjust p, keep alpha" formulation: p_adj = min(p * n, 1).
        adjusted_p = [min(p * n_tests, 1.0) for p in p_values]
        significant = [p < self.alpha for p in adjusted_p]

        return adjusted_p, significant

    def holm_bonferroni_correction(
        self,
        p_values: List[float]
    ) -> Tuple[List[float], List[bool]]:
        """
        Apply Holm-Bonferroni step-down correction (more powerful than Bonferroni).

        Correct algorithm:
        1. Sort p-values ascending.
        2. Adjusted p for rank k (0-based): p_adj[k] = p[k] * (n - k).
        3. Enforce monotonicity: p_adj is made non-decreasing along the sorted
           order via a running maximum (otherwise a small later p could appear
           "more significant" than an earlier larger one).
        4. Clip to [0, 1].
        5. Reject H_(k) only while all earlier hypotheses were rejected
           (step-down: stop at the first non-rejection). This is what makes the
           procedure control the family-wise error rate.

        Args:
            p_values: List of p-values from multiple tests

        Returns:
            Tuple of (adjusted p-values, significance flags in original order)
        """
        n = len(p_values)
        if n == 0:
            return [], []

        # Sort by p-value, remembering original positions.
        order = sorted(range(n), key=lambda i: p_values[i])

        adjusted_sorted = [0.0] * n
        running_max = 0.0
        for rank, idx in enumerate(order):
            raw_adj = p_values[idx] * (n - rank)
            # Step 3: enforce monotone non-decreasing adjusted p-values.
            running_max = max(running_max, raw_adj)
            # Step 4: clip to [0, 1].
            adjusted_sorted[rank] = min(running_max, 1.0)

        # Step 5: step-down rejection - stop at the first non-rejection.
        significant_sorted = [False] * n
        for rank in range(n):
            if adjusted_sorted[rank] < self.alpha:
                significant_sorted[rank] = True
            else:
                # All remaining (larger) hypotheses are not rejected either.
                break

        # Map results back to the original order.
        adjusted = [0.0] * n
        significant = [False] * n
        for rank, idx in enumerate(order):
            adjusted[idx] = adjusted_sorted[rank]
            significant[idx] = significant_sorted[rank]

        return adjusted, significant


def load_experiment_results(
    experiment_type: str,
    model_name: str = "resnet50",
    task: str = "syntax"
) -> pd.DataFrame:
    """
    Load experiment results from CSV files (one per compression format).

    Args:
        experiment_type: "experiment_a"
        model_name: Model architecture name
        task: Dataset task name

    Returns:
        Combined DataFrame with results from all compression formats
    """
    results_dir = config.RESULTS_ROOT / experiment_type
    frames = []

    for fmt in config.COMPRESSION_FORMATS:
        csv_path = results_dir / f"{model_name}_arcade_{task}_{fmt}_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No results found in {results_dir}")

    return pd.concat(frames, ignore_index=True)


def load_baseline_results(
    experiment_type: str,
    model_name: str = "resnet50",
    task: str = "syntax"
) -> Optional[pd.DataFrame]:
    """
    Load the baseline (uncompressed reference) results, if present.

    The baseline CSV holds a SINGLE row (format == 'baseline') measured on the
    uncompressed PNG images. It is a reference point, NOT a treatment group, so
    it is never fed into the statistical tests.

    Args:
        experiment_type: "experiment_a"
        model_name: Model architecture name
        task: Dataset task name

    Returns:
        DataFrame with the baseline row(s), or None if the file does not exist.
    """
    results_dir = config.RESULTS_ROOT / experiment_type
    csv_path = results_dir / f"{model_name}_arcade_{task}_baseline_results.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def get_baseline_reference(
    baseline_df: Optional[pd.DataFrame],
    metrics: List[str]
) -> Dict[str, float]:
    """
    Extract baseline reference metric values from a baseline DataFrame.

    Args:
        baseline_df: DataFrame returned by load_baseline_results (or None)
        metrics: Metric column names to extract

    Returns:
        Mapping metric -> baseline value. Empty if no baseline is available.
    """
    reference: Dict[str, float] = {}
    if baseline_df is None or baseline_df.empty:
        return reference

    for metric in metrics:
        if metric in baseline_df.columns:
            series = baseline_df[metric].dropna()
            if len(series) > 0:
                # Baseline CSV is expected to have a single row; take the first.
                reference[metric] = float(series.iloc[0])
    return reference


def build_paired_matrix(
    df: pd.DataFrame,
    metric_column: str,
    group_column: str = "format",
    block_column: str = "train_quality"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a paired [block x format] matrix for a metric.

    Quality levels (block_column) act as blocks shared by every format. Only
    quality levels present for EVERY compared format are kept (inner join over
    blocks) - this is what guarantees the data are fully paired.

    Args:
        df: Combined DataFrame with results from all formats
        metric_column: Metric column to extract
        group_column: Column identifying the format
        block_column: Column identifying the blocking factor (quality level)

    Returns:
        Tuple of (paired_matrix, warnings):
          - paired_matrix: DataFrame indexed by quality level, one column per
            format, no NaN. May be empty if no shared quality levels exist.
          - warnings: list of human-readable warning strings.
    """
    warnings: List[str] = []

    work = df[[group_column, block_column, metric_column]].copy()
    # Drop rows with a missing metric value so failed/partial runs cannot
    # silently poison the paired matrix.
    work = work.dropna(subset=[metric_column, block_column])

    formats_present = [
        fmt for fmt in config.COMPRESSION_FORMATS
        if fmt in work[group_column].values
    ]

    per_format = {}
    for fmt in formats_present:
        sub = work.loc[work[group_column] == fmt, [block_column, metric_column]]
        # Guard against duplicate quality rows for one format (keep the last).
        sub = sub.drop_duplicates(subset=[block_column], keep='last')
        series = sub.set_index(block_column)[metric_column]
        if len(series) > 0:
            per_format[fmt] = series

    if len(per_format) < 2:
        warnings.append(
            f"Paired matrix for '{metric_column}': fewer than 2 formats have "
            f"usable data ({len(per_format)} found)."
        )
        return pd.DataFrame(), warnings

    # Inner join over blocks: keep only quality levels present for ALL formats.
    matrix = pd.DataFrame(per_format).dropna(how='any')
    matrix = matrix.sort_index()

    if matrix.shape[0] < len(per_format[next(iter(per_format))]):
        warnings.append(
            f"Paired matrix for '{metric_column}': only "
            f"{matrix.shape[0]} quality level(s) are shared by all "
            f"{len(per_format)} formats and are used for paired testing."
        )

    return matrix, warnings


def analyze_format_comparison(
    df: pd.DataFrame,
    metric_column: str = "test_map",
    group_column: str = "format",
    experiment_type: str = "experiment_a",
    baseline_reference: Optional[float] = None
) -> Dict:
    """
    Perform a paired/blocked statistical comparison between compression formats.

    DESIGN: quality levels (train_quality) are a BLOCKING factor shared by every
    format, so observations are PAIRED across formats. The analysis therefore
    uses:
      - Friedman omnibus test (formats = treatments, Q levels = blocks) with
        Kendall's W as the effect size;
      - pairwise paired t-test and Wilcoxon signed-rank tests on vectors paired
        by Q level, with paired Cohen's d as the effect size;
      - Holm-Bonferroni correction over the format pairs.

    Args:
        df: DataFrame with experiment results (all formats combined)
        metric_column: Column name for the metric to compare
        group_column: Column name for the format
        experiment_type: "experiment_a"
        baseline_reference: Optional baseline metric value (uncompressed PNG).
            Stored for reporting only; never used as a treatment group.

    Returns:
        Dictionary with all statistical test results.
    """
    analyzer = StatisticalAnalyzer()
    BLOCK_COLUMN = "train_quality"

    results = {
        'metric': metric_column,
        'experiment_type': experiment_type,
        'analysis_date': datetime.now().isoformat(),
        'alpha': analyzer.alpha,
        'design': 'paired_blocked (blocks = quality levels, treatments = formats)',
        'descriptive_stats': {},
        'paired_quality_levels': [],
        'n_paired_blocks': 0,
        'friedman': None,
        'paired_t_tests': [],
        'wilcoxon_tests': [],
        'holm_bonferroni_paired_t': None,
        'holm_bonferroni_wilcoxon': None,
        'baseline_reference': baseline_reference,
        'warnings': []
    }

    # Build the paired [block x format] matrix (inner join over quality levels).
    paired_matrix, build_warnings = build_paired_matrix(
        df, metric_column, group_column=group_column, block_column=BLOCK_COLUMN
    )
    results['warnings'].extend(build_warnings)

    # Descriptive statistics per format - computed on the PAIRED data so they
    # describe exactly the values entering the tests.
    if not paired_matrix.empty:
        for fmt in paired_matrix.columns:
            data = paired_matrix[fmt].to_numpy(dtype=float)
            results['descriptive_stats'][fmt] = {
                'n': int(len(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data, ddof=1)) if len(data) >= 2 else float('nan'),
                'median': float(np.median(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
            }

    n_blocks = int(paired_matrix.shape[0]) if not paired_matrix.empty else 0
    results['n_paired_blocks'] = n_blocks
    results['paired_quality_levels'] = (
        [int(q) if float(q).is_integer() else float(q)
         for q in paired_matrix.index.tolist()]
        if not paired_matrix.empty else []
    )

    if paired_matrix.empty or len(paired_matrix.columns) < 2:
        results['warnings'].append(
            "Statistical tests skipped: need at least 2 formats with shared "
            "quality levels for a paired comparison."
        )
        return results

    # Need >= 3 paired blocks (quality levels) for the paired tests to be
    # meaningful - skip everything with a warning otherwise.
    if n_blocks < 3:
        results['warnings'].append(
            f"Statistical tests skipped: only {n_blocks} paired quality "
            f"level(s) available (need >= 3 for paired/blocked testing)."
        )
        return results

    formats = list(paired_matrix.columns)

    # ------------------------------------------------------------------ #
    # Omnibus test: Friedman (+ Kendall's W). Needs >= 3 formats.
    # ------------------------------------------------------------------ #
    friedman_result = analyzer.friedman_test(paired_matrix, metric_column)
    results['friedman'] = friedman_result
    if 'warning' in friedman_result:
        results['warnings'].append(friedman_result['warning'])

    # ------------------------------------------------------------------ #
    # Pairwise comparisons: paired t-test + Wilcoxon signed-rank.
    # ------------------------------------------------------------------ #
    p_values_ttest: List[float] = []
    p_values_wilcoxon: List[float] = []

    for i, fmt1 in enumerate(formats):
        for fmt2 in formats[i + 1:]:
            vec1 = paired_matrix[fmt1].to_numpy(dtype=float)
            vec2 = paired_matrix[fmt2].to_numpy(dtype=float)

            ttest_result = analyzer.paired_t_test(vec1, vec2, fmt1, fmt2)
            wilcoxon_result = analyzer.wilcoxon_signed_rank_test(
                vec1, vec2, fmt1, fmt2
            )

            results['paired_t_tests'].append(ttest_result)
            results['wilcoxon_tests'].append(wilcoxon_result)

            # Only feed valid (non-NaN) p-values into the corrections so a NaN
            # cannot corrupt the adjustment.
            if not np.isnan(ttest_result['p_value']):
                p_values_ttest.append(ttest_result['p_value'])
            if not np.isnan(wilcoxon_result['p_value']):
                p_values_wilcoxon.append(wilcoxon_result['p_value'])

    # ------------------------------------------------------------------ #
    # Holm-Bonferroni correction over the format pairs.
    # ------------------------------------------------------------------ #
    if p_values_ttest:
        adjusted_p, significant = analyzer.holm_bonferroni_correction(p_values_ttest)
        results['holm_bonferroni_paired_t'] = {
            'original_p_values': p_values_ttest,
            'adjusted_p_values': adjusted_p,
            'significant': significant
        }

    if p_values_wilcoxon:
        adjusted_p, significant = analyzer.holm_bonferroni_correction(p_values_wilcoxon)
        results['holm_bonferroni_wilcoxon'] = {
            'original_p_values': p_values_wilcoxon,
            'adjusted_p_values': adjusted_p,
            'significant': significant
        }

    # ------------------------------------------------------------------ #
    # Equivalence tests (TOST). A non-significant Friedman/pairwise result only
    # means "no evidence of a difference"; TOST establishes positive practical
    # equivalence within a margin. Margin Delta is a methodological choice (see
    # KWESTIE-DO-KONSULTACJI.md pkt 7) - reported at two values as a sensitivity
    # analysis. Marked 'significant' / authoritative results are the corrected
    # ones; TOST is auxiliary supporting evidence.
    # ------------------------------------------------------------------ #
    results['tost'] = {}
    for delta in TOST_DELTAS:
        delta_key = f"{delta:.3f}"
        results['tost'][delta_key] = []
        for i, fmt1 in enumerate(formats):
            for fmt2 in formats[i + 1:]:
                vec1 = paired_matrix[fmt1].to_numpy(dtype=float)
                vec2 = paired_matrix[fmt2].to_numpy(dtype=float)
                results['tost'][delta_key].append(
                    analyzer.paired_tost(vec1, vec2, delta, fmt1, fmt2)
                )

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
    metric = results.get('metric', 'N/A')
    is_primary = metric in PRIMARY_METRICS
    lines.append(f"\nAnalysis Date: {results.get('analysis_date', 'N/A')}")
    lines.append(f"Experiment Type: {results.get('experiment_type', 'N/A').upper()}")
    lines.append(
        f"Metric Analyzed: {metric}  "
        f"[{'PRIMARY (threshold-free, leading)' if is_primary else 'SECONDARY (fixed-0.5 threshold, interpret with caution)'}]"
    )
    lines.append(f"Significance Level (alpha): {results.get('alpha', 0.05)}")
    lines.append(
        "NOTE: significance is judged on Holm-corrected p-values; the per-pair "
        "'Significant' flags below are UNCORRECTED. Q=100 is included but is "
        "outside the matched-CR regime (KWESTIE pkt 4)."
    )

    lines.append(
        "\nDesign: PAIRED / BLOCKED - quality levels are blocks shared by every "
        "format."
    )
    lines.append(
        "Tests: Friedman omnibus (+ Kendall's W), pairwise paired t-test and "
        "Wilcoxon signed-rank, Holm-Bonferroni correction."
    )

    n_blocks = results.get('n_paired_blocks', 0)
    paired_q = results.get('paired_quality_levels', [])
    lines.append(f"Paired quality levels (blocks): {n_blocks}  {paired_q}")

    # Baseline reference (uncompressed PNG) - reported, not tested.
    baseline = results.get('baseline_reference', None)
    if baseline is not None:
        lines.append(
            f"Baseline (uncompressed reference): {results.get('metric', 'metric')} "
            f"= {baseline:.4f}"
        )

    for warning in results.get('warnings', []):
        lines.append(f"WARNING: {warning}")

    # Descriptive Statistics
    lines.append("\n" + "-" * 40)
    lines.append("DESCRIPTIVE STATISTICS (paired data)")
    lines.append("-" * 40)

    if results.get('descriptive_stats'):
        for fmt, stats_data in results['descriptive_stats'].items():
            lines.append(f"\n{fmt.upper()}:")
            lines.append(f"  N = {stats_data['n']}")
            lines.append(f"  Mean = {stats_data['mean']:.4f}")
            lines.append(f"  Std = {stats_data['std']:.4f}")
            lines.append(f"  Median = {stats_data['median']:.4f}")
            lines.append(f"  Range = [{stats_data['min']:.4f}, {stats_data['max']:.4f}]")
    else:
        lines.append("\n(No paired data available.)")

    # Friedman omnibus test
    if results.get('friedman'):
        fr = results['friedman']
        lines.append("\n" + "-" * 40)
        lines.append("FRIEDMAN TEST (omnibus, paired/blocked)")
        lines.append("-" * 40)
        lines.append(f"Formats (treatments): {fr.get('n_formats', 'N/A')}")
        lines.append(f"Quality levels (blocks): {fr.get('n_blocks', 'N/A')}")
        if 'warning' in fr:
            lines.append(f"WARNING: {fr['warning']}")
        else:
            lines.append(f"Chi-squared statistic: {fr['chi2_statistic']:.4f}")
            lines.append(f"p-value: {fr['p_value']:.6f}")
            lines.append(f"Significant: {'YES' if fr['significant'] else 'NO'}")
            lines.append(
                f"Effect size (Kendall's W): {fr['kendalls_w']:.4f} "
                f"({fr['w_interpretation']})"
            )

    # Pairwise paired t-tests
    if results.get('paired_t_tests'):
        lines.append("\n" + "-" * 40)
        lines.append("PAIRWISE COMPARISONS (paired t-test)")
        lines.append("-" * 40)
        for test in results['paired_t_tests']:
            lines.append(f"\n{test['comparison']}:")
            lines.append(f"  N pairs: {test['n_pairs']}")
            lines.append(f"  Mean difference: {test['mean_diff']:.4f}")
            lines.append(f"  t-statistic: {test['t_statistic']:.4f}")
            lines.append(f"  p-value: {test['p_value']:.6f}")
            lines.append(f"  Significant (uncorrected): {'YES' if test['significant'] else 'NO'}")
            lines.append(
                f"  Paired Cohen's d: {test['cohens_d']:.4f} "
                f"({test['effect_interpretation']})"
            )
            if 'warning' in test:
                lines.append(f"  WARNING: {test['warning']}")

    # Pairwise Wilcoxon signed-rank tests
    if results.get('wilcoxon_tests'):
        lines.append("\n" + "-" * 40)
        lines.append("PAIRWISE COMPARISONS (Wilcoxon signed-rank, non-parametric)")
        lines.append("-" * 40)
        for test in results['wilcoxon_tests']:
            lines.append(f"\n{test['comparison']}:")
            lines.append(f"  N pairs: {test['n_pairs']}")
            lines.append(f"  Statistic: {test['statistic']:.4f}")
            lines.append(f"  p-value: {test['p_value']:.6f}")
            lines.append(f"  Significant (uncorrected): {'YES' if test['significant'] else 'NO'}")
            if 'warning' in test:
                lines.append(f"  WARNING: {test['warning']}")

    # Holm-Bonferroni correction (paired t-test)
    if results.get('holm_bonferroni_paired_t'):
        holm = results['holm_bonferroni_paired_t']
        lines.append("\n" + "-" * 40)
        lines.append("HOLM-BONFERRONI CORRECTION (paired t-test)")
        lines.append("-" * 40)
        lines.append(f"Original p-values: {[f'{p:.6f}' for p in holm['original_p_values']]}")
        lines.append(f"Adjusted p-values: {[f'{p:.6f}' for p in holm['adjusted_p_values']]}")
        lines.append(f"Significant after correction: {holm['significant']}")

    # Holm-Bonferroni correction (Wilcoxon)
    if results.get('holm_bonferroni_wilcoxon'):
        holm = results['holm_bonferroni_wilcoxon']
        lines.append("\n" + "-" * 40)
        lines.append("HOLM-BONFERRONI CORRECTION (Wilcoxon signed-rank)")
        lines.append("-" * 40)
        lines.append(f"Original p-values: {[f'{p:.6f}' for p in holm['original_p_values']]}")
        lines.append(f"Adjusted p-values: {[f'{p:.6f}' for p in holm['adjusted_p_values']]}")
        lines.append(f"Significant after correction: {holm['significant']}")

    # TOST equivalence tests
    if results.get('tost'):
        lines.append("\n" + "-" * 40)
        lines.append("EQUIVALENCE TESTS (TOST, paired)")
        lines.append("-" * 40)
        lines.append(
            "Equivalent => true mean difference lies within +/-Delta. Primary "
            "margin Delta=0.02 (~2x single-seed noise floor); 0.01 / 0.015 shown "
            "as a sensitivity ladder."
        )
        for delta_key in sorted(results['tost'].keys()):
            lines.append(f"\nDelta = +/-{delta_key} ({float(delta_key) * 100:.1f} pp):")
            for t in results['tost'][delta_key]:
                verdict = 'EQUIVALENT' if t.get('equivalent') else 'not shown'
                ci = t.get('ci90', [float('nan'), float('nan')])
                lines.append(
                    f"  {t['comparison']}: TOST p={t['p_tost']:.4f} -> {verdict} "
                    f"(mean diff {t['mean_diff']:+.4f}, 90% CI "
                    f"[{ci[0]:+.4f}, {ci[1]:+.4f}])"
                )
                if 'warning' in t:
                    lines.append(f"    WARNING: {t['warning']}")

    # Summary
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)

    friedman = results.get('friedman') or {}
    friedman_sig = friedman.get('significant', False)
    friedman_ran = 'warning' not in friedman and results.get('friedman') is not None

    # Did any pairwise comparison survive Holm correction? (post-hoc truth)
    def _any_holm_sig(key):
        block = results.get(key) or {}
        return any(block.get('significant', []))
    holm_t_sig = _any_holm_sig('holm_bonferroni_paired_t')
    holm_w_sig = _any_holm_sig('holm_bonferroni_wilcoxon')

    if not friedman_ran:
        lines.append("\nFriedman omnibus test was not run (see warnings above).")
    elif friedman_sig:
        lines.append(
            "\nFriedman test detected significant differences between formats "
            f"(p < {results.get('alpha', 0.05)})."
        )
    else:
        lines.append(
            "\nFriedman test did not detect significant differences between "
            "formats."
        )

    # Reconcile omnibus with post-hoc so the summary never contradicts the
    # Holm-corrected pairwise results shown above.
    if holm_t_sig or holm_w_sig:
        which = []
        if holm_t_sig:
            which.append("paired t-test")
        if holm_w_sig:
            which.append("Wilcoxon")
        note = (
            f"NOTE: at least one pairwise comparison is significant after "
            f"Holm correction ({', '.join(which)})."
        )
        if not friedman_sig:
            note += (
                " The omnibus Friedman test is NOT significant, so this post-hoc "
                "result is not protected by the omnibus and should be treated as "
                "exploratory."
            )
        lines.append(note)
    else:
        lines.append(
            "No pairwise comparison is significant after Holm correction."
        )

    if not is_primary:
        lines.append(
            "CAUTION: this is a SECONDARY metric (fixed-0.5 threshold, sub-optimal "
            "under pos_weight) - the leading conclusion rests on mAP."
        )

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def load_quality_per_condition(task: str, split: str = "train") -> Optional[pd.DataFrame]:
    """
    Mean PSNR/SSIM/CR per (format, quality) from the image-quality CSVs.

    The model is trained on compressed train+val images; quality is essentially
    split-independent per (format, Q), so the train split (the bulk of the
    training signal) is used. Returns None if the quality CSVs are absent.
    """
    metrics_dir = config.RESULTS_ROOT / "metrics"
    frames = []
    for fmt in config.COMPRESSION_FORMATS:
        p = metrics_dir / f"quality_{task}_{split}_{fmt}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    q = pd.concat(frames, ignore_index=True)
    return q.groupby(['format', 'quality']).agg(
        psnr=('psnr', 'mean'),
        ssim=('ssim', 'mean'),
        cr=('compression_ratio', 'mean'),
    ).reset_index()


def analyze_quality_vs_performance(
    perf_df: pd.DataFrame,
    task: str,
    metric_column: str = "test_map",
) -> Dict:
    """
    Correlate a classification metric with perceptual image quality.

    Tests whether perceptual fidelity (PSNR / SSIM) or compression ratio (CR)
    predicts classification performance across the format x quality grid
    (KWESTIE pkt 8). Reports Spearman (primary, monotonic) and Pearson, both
    per-format (n=13) and pooled (n=39), on the full Q range and the matched-CR
    range Q=10-95 (pkt 4).

    IMPORTANT caveat (recorded in the output): the pooled n=39 points are NOT
    independent - the 13 Q levels are shared across formats and each cell is a
    single training seed (n=1). Pooled p-values are therefore optimistic and
    must be read as descriptive, not confirmatory.
    """
    result = {
        'metric': metric_column,
        'available': False,
        'note': (
            'Spearman is primary. Pooled n=39 is pseudoreplicated (shared Q grid, '
            'n=1 per cell) - p-values descriptive only.'
        ),
        'per_format': {},
        'pooled': {},
        'warnings': [],
    }

    quality = load_quality_per_condition(task)
    if quality is None:
        result['warnings'].append('No image-quality CSVs found - correlation skipped.')
        return result

    perf = perf_df[['format', 'train_quality', metric_column]].dropna()
    merged = perf.merge(
        quality, left_on=['format', 'train_quality'],
        right_on=['format', 'quality'], how='inner'
    )
    if merged.empty:
        result['warnings'].append('No overlap between performance and quality grids.')
        return result

    result['available'] = True

    def _corr(sub):
        out = {}
        for qmetric in ['psnr', 'ssim', 'cr']:
            if len(sub) >= 3 and sub[qmetric].std(ddof=1) > 1e-12 \
                    and sub[metric_column].std(ddof=1) > 1e-12:
                sr, sp = stats.spearmanr(sub[metric_column], sub[qmetric])
                pr, pp = stats.pearsonr(sub[metric_column], sub[qmetric])
                out[qmetric] = {
                    'n': int(len(sub)),
                    'spearman_rho': float(sr), 'spearman_p': float(sp),
                    'pearson_r': float(pr), 'pearson_p': float(pp),
                }
            else:
                out[qmetric] = {'n': int(len(sub)), 'spearman_rho': float('nan'),
                                'spearman_p': float('nan'), 'pearson_r': float('nan'),
                                'pearson_p': float('nan')}
        return out

    for rng_name, sub_all in [('Q10-100', merged), ('Q10-95', merged[merged['train_quality'] < 100])]:
        result['pooled'][rng_name] = _corr(sub_all)
        result['per_format'][rng_name] = {
            fmt: _corr(sub_all[sub_all['format'] == fmt])
            for fmt in config.COMPRESSION_FORMATS
            if (sub_all['format'] == fmt).any()
        }

    return result


def generate_quality_correlation_report(
    qcorr: Dict,
    model_name: str,
    output_path: Optional[Path] = None,
) -> str:
    """Human-readable report for the mAP <-> quality correlation analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("QUALITY-VS-PERFORMANCE CORRELATION (mAP <-> PSNR / SSIM / CR)")
    lines.append("=" * 80)
    lines.append(f"\nModel: {model_name}")
    lines.append(f"Performance metric: {qcorr.get('metric', 'N/A')}")
    lines.append(f"NOTE: {qcorr.get('note', '')}")

    if not qcorr.get('available'):
        for w in qcorr.get('warnings', []):
            lines.append(f"WARNING: {w}")
        report = "\n".join(lines)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Quality-correlation report saved to: {output_path}")
        return report

    def _fmt_block(block):
        out = []
        for qm in ['psnr', 'ssim', 'cr']:
            c = block[qm]
            out.append(
                f"    {qm.upper():5s} (n={c['n']}): Spearman rho={c['spearman_rho']:+.3f} "
                f"(p={c['spearman_p']:.3f}) | Pearson r={c['pearson_r']:+.3f} "
                f"(p={c['pearson_p']:.3f})"
            )
        return out

    for rng in ['Q10-100', 'Q10-95']:
        lines.append("\n" + "-" * 40)
        lines.append(f"RANGE {rng}")
        lines.append("-" * 40)
        lines.append("  POOLED (n=39, pseudoreplicated - descriptive):")
        lines.extend(_fmt_block(qcorr['pooled'][rng]))
        for fmt, block in qcorr['per_format'][rng].items():
            lines.append(f"  {fmt.upper()} (per-format, n=13):")
            lines.extend(_fmt_block(block))

    lines.append("\n" + "=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append(
        "\nIf perceptual fidelity predicted classification performance, the mAP-PSNR "
        "and mAP-SSIM correlations would be positive and consistent across formats "
        "and architectures. A correlation that changes sign/significance between "
        "architectures indicates the format x Q variation is dominated by single-seed "
        "training noise rather than by image quality (supports KWESTIE pkt 3 & 8)."
    )
    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Quality-correlation report saved to: {output_path}")
    return report


def run_full_statistical_analysis(
    experiment_type: str = "experiment_a",
    model_name: str = "resnet50",
    task: str = "syntax",
    metrics: List[str] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run the complete paired/blocked statistical analysis on experiment results.

    Args:
        experiment_type: "experiment_a"
        model_name: Model architecture name
        task: Dataset task name
        metrics: List of metrics to analyze. Defaults to DEFAULT_METRICS, which
            puts mAP (test_map, threshold-free, leading metric) first.
        output_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    # mAP (test_map) is the leading, threshold-free metric; F1 at threshold 0.5
    # is auxiliary.
    if metrics is None:
        metrics = list(DEFAULT_METRICS)

    if output_dir is None:
        output_dir = config.RESULTS_ROOT / "statistical_analysis"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load compressed-format data.
    print(f"Loading {experiment_type} results...")
    df = load_experiment_results(experiment_type, model_name, task)
    print(f"Loaded {len(df)} records")

    # Load the optional baseline (uncompressed reference). Missing -> skip.
    baseline_df = load_baseline_results(experiment_type, model_name, task)
    baseline_reference = get_baseline_reference(baseline_df, metrics)
    if baseline_reference:
        print(f"Loaded baseline reference for: {sorted(baseline_reference)}")
    else:
        print("No baseline reference found (optional) - continuing without it.")

    all_results = {
        'experiment_type': experiment_type,
        'model': model_name,
        'task': task,
        'analysis_date': datetime.now().isoformat(),
        'baseline_reference': baseline_reference,
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
            experiment_type=experiment_type,
            baseline_reference=baseline_reference.get(metric)
        )

        all_results['metrics'][metric] = results

        # Generate report. Filenames carry model_name so runs for different
        # models do not overwrite each other.
        report = generate_statistical_report(
            results,
            output_path=output_dir / f"report_{experiment_type}_{model_name}_{metric}.txt"
        )

        # Save JSON results
        json_path = output_dir / f"results_{experiment_type}_{model_name}_{metric}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {json_path}")

    # Quality-vs-performance correlation (mAP <-> PSNR/SSIM/CR), KWESTIE pkt 8.
    # Computed on the leading metric (mAP) when available.
    qmetric = 'test_map' if 'test_map' in df.columns else (metrics[0] if metrics else None)
    if qmetric:
        print(f"\nAnalyzing quality-vs-performance correlation ({qmetric})...")
        qcorr = analyze_quality_vs_performance(df, task, metric_column=qmetric)
        all_results['quality_correlation'] = qcorr
        generate_quality_correlation_report(
            qcorr, model_name,
            output_path=output_dir / f"quality_correlation_{experiment_type}_{model_name}.txt"
        )

    # Save combined results
    combined_path = output_dir / f"combined_analysis_{experiment_type}_{model_name}.json"
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
        choices=["experiment_a"],
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
        choices=["syntax"],
        help="Dataset task"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        # mAP first: threshold-free leading metric; F1@0.5 is auxiliary.
        default=list(DEFAULT_METRICS),
        help="Metrics to analyze (mAP is the leading threshold-free metric)"
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
