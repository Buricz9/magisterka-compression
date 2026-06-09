"""
Statistical analysis for Experiment B (train on PNG, test on COMPRESSED images).

Experiment B asks the MIRROR question to Experiment A: does compressing the TEST
images hurt a model that was trained on clean PNG? Here the model is FIXED (the
single PNG-trained baseline checkpoint), and only the test data varies. This has
an important methodological consequence:

    The drop seen in Experiment B CANNOT be single-seed training noise, because
    no training happens — it is the SAME model evaluated on different inputs.

So unlike Experiment A (where Q-to-Q variation was indistinguishable from seed
noise), any monotone degradation here is a genuine, deterministic effect of
test-time compression.

Tests:
  - Per format: Spearman + Pearson correlation of the metric with test-Q, plus
    the drop from Q=100 to Q=10 and the gap below the PNG baseline. A monotone
    rise with Q (lower Q -> lower metric) means compression hurts.
  - Across formats (Friedman, blocks = Q levels): are the three formats
    degraded differently? (Expectation: JPEG worst due to blocking artifacts.)

Run: python -m src.analysis.analyze_experiment_b --model resnet50 --task syntax
"""
import sys
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

FORMATS = list(config.COMPRESSION_FORMATS)          # jpeg, jpeg2000, avif
PRIMARY = "test_map"                                  # leading metric
METRICS = ["test_map", "test_f1_macro", "test_hamming_accuracy"]
ALPHA = 0.05


def _load_b(model_name, task):
    p = config.RESULTS_ROOT / "experiment_b" / f"{model_name}_arcade_{task}_results.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Experiment B results not found: {p}\n"
            f"Run `python -m src.experiments.experiment_b --model {model_name} "
            f"--format all` first."
        )
    d = pd.read_csv(p)
    d["test_quality"] = pd.to_numeric(d["test_quality"], errors="coerce")
    return d.dropna(subset=["test_quality"])


def _baseline(model_name, task, col):
    p = config.RESULTS_ROOT / "experiment_a" / f"{model_name}_arcade_{task}_baseline_results.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p)
    return float(b.iloc[0][col]) if col in b.columns and len(b) else None


def per_format_correlation(d, col):
    """Correlation of metric with test-Q, per format, on the matched-CR range.

    Q=100 is outside the matched-CR regime (see Experiment A discussion), so we
    report both the full range and Q=10..95. A POSITIVE correlation means the
    metric falls as Q falls (compression hurts).
    """
    out = {}
    for fmt in FORMATS:
        s = d[d["format"] == fmt].sort_values("test_quality")
        if s.empty:
            continue
        rec = {}
        for rng, sub in [("Q10-100", s), ("Q10-95", s[s["test_quality"] < 100])]:
            x = sub["test_quality"].to_numpy(float)
            y = sub[col].to_numpy(float)
            if len(x) >= 3 and np.std(x) > 1e-9 and np.std(y) > 1e-12:
                sr, sp = stats.spearmanr(x, y)
                pr, pp = stats.pearsonr(x, y)
                rec[rng] = {"n": int(len(x)),
                            "spearman_rho": float(sr), "spearman_p": float(sp),
                            "pearson_r": float(pr), "pearson_p": float(pp)}
            else:
                rec[rng] = {"n": int(len(x)), "spearman_rho": float("nan"),
                            "spearman_p": float("nan"), "pearson_r": float("nan"),
                            "pearson_p": float("nan")}
        # drop Q100 -> Q10 and gap vs PNG baseline
        hi = s[s["test_quality"] == 100][col]
        lo = s[s["test_quality"] == 10][col]
        rec["q100"] = float(hi.iloc[0]) if len(hi) else float("nan")
        rec["q10"] = float(lo.iloc[0]) if len(lo) else float("nan")
        rec["drop_100_10"] = rec["q100"] - rec["q10"]
        rec["mean"] = float(s[col].mean())
        rec["std"] = float(s[col].std(ddof=1))
        out[fmt] = rec
    return out


def friedman_across_formats(d, col):
    """Friedman test: are the 3 formats degraded differently across Q blocks?

    Blocks = the 13 Q levels shared by all formats; treatments = formats.
    """
    pivot = d.pivot_table(index="test_quality", columns="format", values=col)
    pivot = pivot.dropna(how="any")
    fmts = [f for f in FORMATS if f in pivot.columns]
    if len(fmts) < 3 or pivot.shape[0] < 3:
        return {"ran": False, "reason": "need 3 formats x >=3 shared Q levels"}
    cols = [pivot[f].to_numpy(float) for f in fmts]
    chi2, p = stats.friedmanchisquare(*cols)
    n_blocks, k = pivot.shape[0], len(fmts)
    w = float(chi2) / (n_blocks * (k - 1)) if n_blocks * (k - 1) > 0 else float("nan")
    return {"ran": True, "formats": fmts, "n_blocks": int(n_blocks),
            "chi2": float(chi2), "p_value": float(p), "kendalls_w": w,
            "significant": bool(p < ALPHA)}


def analyze(model_name, task):
    d = _load_b(model_name, task)
    res = {"model": model_name, "task": task,
           "n_rows": int(len(d)), "formats": FORMATS}
    res["baseline"] = {m: _baseline(model_name, task, m) for m in METRICS}
    res["correlation"] = {m: per_format_correlation(d, m) for m in METRICS}
    res["friedman"] = {m: friedman_across_formats(d, m) for m in METRICS}
    return res


def report(res):
    L = []
    L.append("=" * 80)
    L.append("EXPERIMENT B — STATISTICAL ANALYSIS (train on PNG, test on compressed)")
    L.append("=" * 80)
    L.append(f"\nModel: {res['model']}   Task: {res['task']}   rows: {res['n_rows']}")
    L.append("\nNOTE: the model is fixed (PNG-trained baseline); only the test set "
             "varies.\nTherefore any monotone degradation here is a deterministic "
             "effect of\ntest-time compression, NOT single-seed training noise.")
    bl = res["baseline"].get(PRIMARY)
    if bl is not None:
        L.append(f"Baseline (test on PNG) {PRIMARY} = {bl:.4f}")

    L.append("\n" + "-" * 40)
    L.append(f"PER-FORMAT CORRELATION OF {PRIMARY} WITH TEST-Q")
    L.append("(positive rho => metric falls as compression rises => compression hurts)")
    L.append("-" * 40)
    for fmt, rec in res["correlation"][PRIMARY].items():
        full = rec["Q10-100"]
        gap = (rec["mean"] - bl) if bl is not None else float("nan")
        L.append(f"\n{fmt.upper()}:")
        L.append(f"  Q=100 -> Q=10: {rec['q100']:.4f} -> {rec['q10']:.4f} "
                 f"(drop {rec['drop_100_10']:+.4f})")
        L.append(f"  Spearman rho={full['spearman_rho']:+.3f} (p={full['spearman_p']:.3f}) | "
                 f"Pearson r={full['pearson_r']:+.3f} (p={full['pearson_p']:.3f})")
        L.append(f"  mean={rec['mean']:.4f}  vs baseline gap={gap:+.4f}")

    L.append("\n" + "-" * 40)
    L.append(f"FRIEDMAN ACROSS FORMATS ({PRIMARY}; blocks = Q levels)")
    L.append("-" * 40)
    fr = res["friedman"][PRIMARY]
    if fr["ran"]:
        L.append(f"chi2={fr['chi2']:.3f}  p={fr['p_value']:.4f}  "
                 f"Kendall W={fr['kendalls_w']:.3f}  "
                 f"significant: {'YES' if fr['significant'] else 'NO'}")
    else:
        L.append(f"not run: {fr['reason']}")

    L.append("\n" + "=" * 80)
    L.append("SUMMARY")
    L.append("=" * 80)
    corr = res["correlation"][PRIMARY]
    hurt = [f for f, r in corr.items()
            if r["Q10-100"]["spearman_p"] < ALPHA and r["Q10-100"]["spearman_rho"] > 0]
    if hurt:
        L.append("Test-time compression significantly degrades " + str([f.upper() for f in hurt])
                 + f" (Spearman p<{ALPHA}). This is a deterministic effect (fixed model),")
        L.append("in contrast to the flat result of Experiment A.")
    else:
        L.append("No format shows a statistically significant monotone degradation.")
    L.append("=" * 80)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Statistical analysis of Experiment B")
    ap.add_argument("--model", default="resnet50", choices=config.SUPPORTED_MODELS)
    ap.add_argument("--task", default="syntax", choices=["syntax"])
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        config.RESULTS_ROOT / "statistical_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = analyze(args.model, args.task)
    txt = report(res)
    print(txt)

    (out_dir / f"report_experiment_b_{args.model}_test_map.txt").write_text(
        txt, encoding="utf-8")
    with open(out_dir / f"results_experiment_b_{args.model}.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
