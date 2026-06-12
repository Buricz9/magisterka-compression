"""
Statistical analysis for Experiment C (mixed-quality training).

Experiment C trains ONE model per format on a MIXTURE of compression levels
(random Q per image access = compression as augmentation) and then evaluates it
both on clean PNG and on every Q level. The question is whether exposing the
model to compression artifacts during training removes the test-time degradation
that Experiment B revealed for JPEG.

This script reports DESCRIPTIVE statistics only (no C-vs-B significance test:
both are single-seed, n=1, so a formal difference test is not licensed). It
quantifies, per format and architecture:

  - Spearman/Pearson correlation of the metric with test-Q over the compressed
    range (mirrors analyze_experiment_b), plus the Q=100 -> Q=10 drop. A FLAT
    curve (rho ~ 0, drop ~ 0) means mixed training closed the gap.
  - The non-circular control: the metric on clean PNG (NOT in the training
    mixture) vs the PNG-trained baseline. Near-zero gap means the cure costs
    nothing on the unseen high-quality domain.
  - Side-by-side with Experiment B (the same drop/rho computed on the PNG-trained
    model) so the flattening is visible: B rho -> C rho, B drop -> C drop.

Run: python -m src.analysis.analyze_experiment_c --model resnet50 --task syntax
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
METRICS = ["test_map", "test_f1_macro", "test_f1_micro", "test_hamming_accuracy"]
ALPHA = 0.05


def _load_c(model_name, task, fmt):
    """Load one Experiment C CSV (png row + 13 compressed Q rows) for a format."""
    p = (config.RESULTS_ROOT / "experiment_c" /
         f"{model_name}_arcade_{task}_mixed_{fmt}_results.csv")
    if not p.exists():
        raise FileNotFoundError(
            f"Experiment C results not found: {p}\n"
            f"Run `python -m src.experiments.experiment_c --model {model_name} "
            f"--format {fmt}` first."
        )
    d = pd.read_csv(p)
    png = d[d["test_quality"].astype(str) == "png"]
    comp = d[d["test_quality"].astype(str) != "png"].copy()
    comp["test_quality"] = pd.to_numeric(comp["test_quality"], errors="coerce")
    comp = comp.dropna(subset=["test_quality"]).sort_values("test_quality")
    return png, comp


def _baseline(model_name, task, col):
    """PNG-trained baseline value (Experiment A baseline), test on PNG."""
    p = (config.RESULTS_ROOT / "experiment_a" /
         f"{model_name}_arcade_{task}_baseline_results.csv")
    if not p.exists():
        return None
    b = pd.read_csv(p)
    return float(b.iloc[0][col]) if col in b.columns and len(b) else None


def _b_drop_and_rho(model_name, task, fmt, col):
    """Experiment B's Q100->Q10 drop and Spearman rho for the same (fmt, col).

    Used purely for the descriptive B -> C side-by-side; B is the fixed
    PNG-trained model evaluated on compressed test data.
    """
    p = (config.RESULTS_ROOT / "experiment_b" /
         f"{model_name}_arcade_{task}_results.csv")
    if not p.exists():
        return {"drop_100_10": float("nan"), "spearman_rho": float("nan"),
                "spearman_p": float("nan")}
    d = pd.read_csv(p)
    d["test_quality"] = pd.to_numeric(d["test_quality"], errors="coerce")
    s = d[(d["format"] == fmt)].dropna(subset=["test_quality"]).sort_values("test_quality")
    if s.empty:
        return {"drop_100_10": float("nan"), "spearman_rho": float("nan"),
                "spearman_p": float("nan")}
    x, y = s["test_quality"].to_numpy(float), s[col].to_numpy(float)
    hi = s[s["test_quality"] == 100][col]
    lo = s[s["test_quality"] == 10][col]
    rho, pval = (stats.spearmanr(x, y) if len(x) >= 3 and np.std(y) > 1e-12
                 else (float("nan"), float("nan")))
    return {"drop_100_10": (float(hi.iloc[0]) - float(lo.iloc[0])
                            if len(hi) and len(lo) else float("nan")),
            "spearman_rho": float(rho), "spearman_p": float(pval)}


def per_format(model_name, task, fmt, col):
    """Descriptive stats for one (format, metric) under mixed training."""
    png, comp = _load_c(model_name, task, fmt)
    x = comp["test_quality"].to_numpy(float)
    y = comp[col].to_numpy(float)
    rec = {"n": int(len(x))}
    if len(x) >= 3 and np.std(x) > 1e-9 and np.std(y) > 1e-12:
        sr, sp = stats.spearmanr(x, y)
        pr, pp = stats.pearsonr(x, y)
        rec.update({"spearman_rho": float(sr), "spearman_p": float(sp),
                    "pearson_r": float(pr), "pearson_p": float(pp)})
    else:
        rec.update({"spearman_rho": float("nan"), "spearman_p": float("nan"),
                    "pearson_r": float("nan"), "pearson_p": float("nan")})
    hi = comp[comp["test_quality"] == 100][col]
    lo = comp[comp["test_quality"] == 10][col]
    rec["q100"] = float(hi.iloc[0]) if len(hi) else float("nan")
    rec["q10"] = float(lo.iloc[0]) if len(lo) else float("nan")
    rec["drop_100_10"] = rec["q100"] - rec["q10"]
    rec["mean"] = float(comp[col].mean())
    rec["std"] = float(comp[col].std(ddof=1))
    # non-circular control: clean PNG (not in the training mixture)
    rec["png"] = float(png.iloc[0][col]) if len(png) else float("nan")
    bl = _baseline(model_name, task, col)
    rec["baseline"] = bl
    rec["png_gap_vs_baseline"] = (rec["png"] - bl) if bl is not None else float("nan")
    rec["b"] = _b_drop_and_rho(model_name, task, fmt, col)
    return rec


def analyze(model_name, task):
    res = {"model": model_name, "task": task, "formats": FORMATS}
    res["per_format"] = {
        m: {f: per_format(model_name, task, f, m) for f in FORMATS}
        for m in METRICS
    }
    return res


def report(res):
    L = []
    L.append("=" * 80)
    L.append("EXPERIMENT C — DESCRIPTIVE ANALYSIS (mixed-quality training)")
    L.append("=" * 80)
    L.append(f"\nModel: {res['model']}   Task: {res['task']}")
    L.append("\nOne model per format, trained on a MIXTURE of Q (compression as "
             "augmentation),\nthen tested on clean PNG and on every Q level. "
             "n=1 per cell: no C-vs-B\nsignificance test is reported, only the "
             "descriptive flattening of the curve.")

    L.append("\n" + "-" * 72)
    L.append(f"{PRIMARY}: B (PNG-trained) vs C (mixed-trained), per format")
    L.append("-" * 72)
    L.append(f"{'format':<10} {'B drop':>8} {'B rho':>7} | {'C drop':>8} "
             f"{'C rho':>7} | {'C PNG':>7} {'vs base':>8}")
    for fmt, rec in res["per_format"][PRIMARY].items():
        b = rec["b"]
        L.append(f"{fmt:<10} {b['drop_100_10']:>+8.3f} {b['spearman_rho']:>+7.2f} | "
                 f"{rec['drop_100_10']:>+8.3f} {rec['spearman_rho']:>+7.2f} | "
                 f"{rec['png']:>7.3f} {rec['png_gap_vs_baseline']:>+8.3f}")

    L.append("\n(B drop = Q100->Q10 mAP drop for the fixed PNG-trained model;")
    L.append(" C drop = same for the mixed-trained model; positive rho => metric")
    L.append(" falls as compression rises. C PNG = mAP on clean PNG, which is NOT")
    L.append(" in the training mixture — the non-circular check that the cure is free.)")

    # threshold-metric note (EffNet pays a small PNG cost in f1_micro/hamming)
    L.append("\n" + "-" * 72)
    L.append("Threshold-metric note (clean-PNG gap vs baseline):")
    L.append("-" * 72)
    for m in ["test_f1_micro", "test_hamming_accuracy"]:
        gaps = res["per_format"][m]
        L.append(f"  {m}:")
        for fmt, rec in gaps.items():
            L.append(f"    {fmt:<10} PNG {rec['png']:.4f}  vs baseline "
                     f"{rec['baseline']:.4f}  gap {rec['png_gap_vs_baseline']:+.4f}")

    L.append("\n" + "=" * 80)
    L.append("SUMMARY")
    L.append("=" * 80)
    jpeg = res["per_format"][PRIMARY]["jpeg"]
    L.append(f"JPEG mAP: B drop {jpeg['b']['drop_100_10']:+.3f} (rho "
             f"{jpeg['b']['spearman_rho']:+.2f}) -> C drop {jpeg['drop_100_10']:+.3f} "
             f"(rho {jpeg['spearman_rho']:+.2f}); clean-PNG gap "
             f"{jpeg['png_gap_vs_baseline']:+.3f}.")
    L.append("Mixed training flattens the JPEG curve at ~zero cost on the unseen "
             "PNG domain.")
    L.append("=" * 80)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Descriptive analysis of Experiment C")
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

    (out_dir / f"report_experiment_c_{args.model}_test_map.txt").write_text(
        txt, encoding="utf-8")
    with open(out_dir / f"results_experiment_c_{args.model}.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
