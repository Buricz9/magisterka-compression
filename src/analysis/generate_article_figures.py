"""
Article figures for Experiment A (both architectures, all 3 formats, 13 Q).

Produces publication-ready PDFs in plots/:
  - fig_exp_a_map.pdf   : mAP vs Q, ResNet-50 + EfficientNet-B0, baseline line
  - fig_exp_a_f1.pdf    : F1-macro vs Q, same layout
  - fig_quality.pdf     : PSNR / SSIM / CR vs Q for the 3 formats
  - fig_map_vs_psnr.pdf : mAP vs mean PSNR scatter (per format), per model

Design choices (so figures match the thesis' "no real effect / noise" message):
  - Both panels of a metric share the SAME Y axis (and X axis in the scatter),
    so the two architectures are visually comparable.
  - Q-vs-metric lines are thin and semi-transparent: the markers carry the data,
    the connecting line does not dramatise single-seed noise as a trend.
  - The scatter shows a regression line + Spearman rho, so the (lack of a)
    quality->performance relationship is visible, not just asserted.

Run: python -m src.analysis.generate_article_figures
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

MODELS = [("resnet50", "ResNet-50"), ("efficientnet_b0", "EfficientNet-B0")]
FORMATS = [
    ("jpeg", "JPEG", "#1f77b4", "o", "-"),
    ("jpeg2000", "JPEG2000", "#ff7f0e", "s", "--"),
    ("avif", "AVIF", "#2ca02c", "^", "-."),
]
PLOTS = config.PLOTS_ROOT
PLOTS.mkdir(parents=True, exist_ok=True)

for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
    try:
        plt.style.use(style)
        break
    except OSError:
        continue


def _load(model, fmt):
    p = config.RESULTS_ROOT / "experiment_a" / f"{model}_arcade_syntax_{fmt}_results.csv"
    return pd.read_csv(p).sort_values("train_quality") if p.exists() else None


def _baseline(model, col):
    p = config.RESULTS_ROOT / "experiment_a" / f"{model}_arcade_syntax_baseline_results.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p)
    return float(b.iloc[0][col]) if col in b.columns and len(b) else None


def _shared_ylim(col):
    """Common y-range (in %) across both models for a metric, incl. baselines."""
    vals = []
    for model, _ in MODELS:
        for fmt, *_ in FORMATS:
            d = _load(model, fmt)
            if d is not None and not d.empty:
                vals.extend((d[col] * 100).tolist())
        b = _baseline(model, col)
        if b is not None:
            vals.append(b * 100)
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    pad = max(0.4, (hi - lo) * 0.06)
    return lo - pad, hi + pad


def _metric_vs_q(col, ylabel, title, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ylim = _shared_ylim(col)
    for ax, (model, mlabel) in zip(axes, MODELS):
        for fmt, flabel, color, marker, ls in FORMATS:
            d = _load(model, fmt)
            if d is None or d.empty:
                continue
            # thin, semi-transparent connector; opaque markers carry the data
            ax.plot(d["train_quality"], d[col] * 100, linestyle=ls, color=color,
                    linewidth=1.0, alpha=0.45, zorder=1)
            ax.plot(d["train_quality"], d[col] * 100, linestyle="none", marker=marker,
                    color=color, markersize=6, label=flabel, zorder=3)
        b = _baseline(model, col)
        if b is not None:
            ax.axhline(b * 100, color="black", linestyle=":", linewidth=1.4,
                       label="Baseline (PNG)", zorder=2)
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_xlabel("Poziom jakości Q", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.invert_xaxis()  # Q decreases left->right => compression increases right
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {PLOTS / out_name}")


def quality_figure():
    """PSNR / SSIM / CR vs Q for the 3 formats (model-independent).

    Uses the TRAIN split only, to stay numerically consistent with the quality
    table in the article (tab:quality_metrics) and the cited CR values.
    """
    frames = []
    for fmt, *_ in FORMATS:
        p = config.RESULTS_ROOT / "metrics" / f"quality_syntax_train_{fmt}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        print("no quality CSVs - skipping fig_quality")
        return
    q = pd.concat(frames, ignore_index=True)
    agg = q.groupby(["format", "quality"]).agg(
        psnr=("psnr", "mean"), ssim=("ssim", "mean"),
        cr=("compression_ratio", "mean")).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panels = [("psnr", "PSNR [dB]"), ("ssim", "SSIM"), ("cr", "Współczynnik kompresji (CR)")]
    for ax, (col, ylabel) in zip(axes, panels):
        for fmt, flabel, color, marker, ls in FORMATS:
            s = agg[agg["format"] == fmt].sort_values("quality")
            if s.empty:
                continue
            ax.plot(s["quality"], s[col], marker=marker, color=color, linestyle=ls,
                    linewidth=1.8, markersize=6, label=flabel)
        ax.set_xlabel("Poziom jakości Q", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if col == "cr":
            ax.set_yscale("log")
            # The three CR curves overlap by design (matched-CR); say so.
            ax.text(0.5, 0.04, "krzywe pokrywają się\n(dopasowane CR)",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, style="italic", color="0.35")
    fig.suptitle("Jakość kompresji w funkcji poziomu Q (dopasowane CR)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig_quality.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {PLOTS / 'fig_quality.pdf'}")


def map_vs_psnr_figure():
    """Scatter: classification mAP vs mean PSNR of the training format/Q."""
    frames = []
    for fmt, *_ in FORMATS:
        p = config.RESULTS_ROOT / "metrics" / f"quality_syntax_train_{fmt}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        print("no train quality CSVs - skipping fig_map_vs_psnr")
        return
    q = pd.concat(frames, ignore_index=True)
    qa = q.groupby(["format", "quality"]).agg(psnr=("psnr", "mean")).reset_index()

    # Pre-compute merged points per model to derive shared axis ranges.
    merged_by_model = {}
    for model, _ in MODELS:
        rows = []
        for fmt, *_ in FORMATS:
            d = _load(model, fmt)
            if d is None or d.empty:
                continue
            m = d.merge(qa[qa["format"] == fmt], left_on="train_quality",
                        right_on="quality", how="inner")
            m["__fmt"] = fmt
            rows.append(m)
        merged_by_model[model] = pd.concat(rows, ignore_index=True) if rows else None

    # Shared X (PSNR) and Y (mAP %) across both panels, incl. baselines.
    xs, ys = [], []
    for model, _ in MODELS:
        mm = merged_by_model[model]
        if mm is not None and not mm.empty:
            xs.extend(mm["psnr"].tolist())
            ys.extend((mm["test_map"] * 100).tolist())
        b = _baseline(model, "test_map")
        if b is not None:
            ys.append(b * 100)
    xpad = max(0.5, (max(xs) - min(xs)) * 0.05)
    ypad = max(0.4, (max(ys) - min(ys)) * 0.06)
    xlim = (min(xs) - xpad, max(xs) + xpad)
    ylim = (min(ys) - ypad, max(ys) + ypad)

    fmt_meta = {f[0]: (f[1], f[2], f[3]) for f in FORMATS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, (model, mlabel) in zip(axes, MODELS):
        mm = merged_by_model[model]
        if mm is not None and not mm.empty:
            for fmt, (flabel, color, marker) in fmt_meta.items():
                sub = mm[mm["__fmt"] == fmt]
                if sub.empty:
                    continue
                ax.scatter(sub["psnr"], sub["test_map"] * 100, color=color,
                           marker=marker, s=45, label=flabel, alpha=0.7,
                           edgecolors="white", linewidths=0.5, zorder=3)
            # regression line + Spearman rho on the pooled points
            x = mm["psnr"].to_numpy(dtype=float)
            y = (mm["test_map"] * 100).to_numpy(dtype=float)
            if len(x) >= 3 and np.std(x) > 1e-9:
                a, b1 = np.polyfit(x, y, 1)
                xr = np.linspace(x.min(), x.max(), 50)
                ax.plot(xr, a * xr + b1, color="0.35", linestyle="-",
                        linewidth=1.4, zorder=2)
                rho, pval = stats.spearmanr(x, y)
                ax.text(0.04, 0.96, f"Spearman $\\rho={rho:+.2f}$ (p={pval:.2f})",
                        transform=ax.transAxes, ha="left", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                                  alpha=0.8))
        b = _baseline(model, "test_map")
        if b is not None:
            ax.axhline(b * 100, color="black", linestyle=":", linewidth=1.4,
                       label="Baseline (PNG)", zorder=1)
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_xlabel("Średni PSNR danych treningowych [dB]", fontsize=11)
        ax.set_ylabel("mAP testowe [%]", fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Skuteczność klasyfikacji a wierność percepcyjna danych treningowych",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig_map_vs_psnr.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {PLOTS / 'fig_map_vs_psnr.pdf'}")


def _load_b(model):
    """Experiment B results (train on PNG, test on compressed): all formats."""
    p = config.RESULTS_ROOT / "experiment_b" / f"{model}_arcade_syntax_results.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["test_quality"] = pd.to_numeric(d["test_quality"], errors="coerce")
    return d


def exp_b_figure(col="test_map", ylabel="mAP [%]", out_name="fig_exp_b_map.pdf"):
    """Experiment B: metric vs TEST compression level Q, per model.

    Mirror of fig_exp_a_*: same baseline (PNG test) reference line, but here the
    x-axis is the quality of the COMPRESSED TEST set; the model is the fixed
    PNG-trained baseline. A drop towards low Q means the model is hurt by
    test-time compression (the effect absent in Experiment A).
    """
    # shared y-range across both models (incl. baseline) for comparability
    vals = []
    for model, _ in MODELS:
        d = _load_b(model)
        if d is not None and not d.empty:
            vals.extend((d[col] * 100).tolist())
        b = _baseline(model, col)
        if b is not None:
            vals.append(b * 100)
    if not vals:
        print("no experiment_b CSVs - skipping fig_exp_b")
        return
    lo, hi = min(vals), max(vals)
    pad = max(0.4, (hi - lo) * 0.06)
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (model, mlabel) in zip(axes, MODELS):
        d = _load_b(model)
        if d is not None and not d.empty:
            for fmt, flabel, color, marker, ls in FORMATS:
                s = d[d["format"] == fmt].sort_values("test_quality")
                if s.empty:
                    continue
                ax.plot(s["test_quality"], s[col] * 100, linestyle=ls, color=color,
                        linewidth=1.0, alpha=0.45, zorder=1)
                ax.plot(s["test_quality"], s[col] * 100, linestyle="none",
                        marker=marker, color=color, markersize=6, label=flabel, zorder=3)
        b = _baseline(model, col)
        if b is not None:
            ax.axhline(b * 100, color="black", linestyle=":", linewidth=1.4,
                       label="Baseline (test PNG)", zorder=2)
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_xlabel("Poziom kompresji zbioru testowego Q", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.invert_xaxis()
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("Eksperyment B: mAP w funkcji kompresji zbioru testowego "
                 "(model trenowany na PNG)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {PLOTS / out_name}")


def _load_c(model):
    """Experiment C results (mixed-quality training), all formats concatenated.

    One CSV per format; each holds a clean-PNG row plus 13 compressed Q rows.
    Returns only the compressed rows (numeric test_quality), matching _load_b.
    """
    frames = []
    for fmt, *_ in FORMATS:
        p = (config.RESULTS_ROOT / "experiment_c" /
             f"{model}_arcade_syntax_mixed_{fmt}_results.csv")
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    comp = d[d["test_quality"].astype(str) != "png"].copy()
    comp["test_quality"] = pd.to_numeric(comp["test_quality"], errors="coerce")
    return comp.dropna(subset=["test_quality"]).sort_values("test_quality")


def exp_c_figure(col="test_map", ylabel="mAP [%]", out_name="fig_exp_c_map.pdf"):
    """Experiment C vs B for JPEG: does mixed-quality training cure the drop?

    Per model, overlays the JPEG curve from Experiment B (fixed PNG-trained model,
    falling towards low Q) and Experiment C (mixed-quality-trained model, flat),
    against the same PNG baseline. The collapse of the B->C gap is the cure: the
    model that saw compression in training is no longer hurt by it at test time.
    JPEG is shown because it is the only format Experiment B degraded.
    """
    color_b, color_c = "#d62728", "#1f77b4"
    vals = []
    for model, _ in MODELS:
        for loader in (_load_b, _load_c):
            d = loader(model)
            if d is not None and not d.empty:
                s = d[d["format"] == "jpeg"]
                vals.extend((s[col] * 100).tolist())
        b = _baseline(model, col)
        if b is not None:
            vals.append(b * 100)
    if not vals:
        print("no experiment_c CSVs - skipping fig_exp_c")
        return
    lo, hi = min(vals), max(vals)
    pad = max(0.4, (hi - lo) * 0.06)
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (model, mlabel) in zip(axes, MODELS):
        for loader, color, flabel, marker, ls in (
                (_load_b, color_b, "JPEG, trening PNG (Eksp. B)", "o", "--"),
                (_load_c, color_c, "JPEG, trening mieszany (Eksp. C)", "s", "-")):
            d = loader(model)
            if d is None or d.empty:
                continue
            s = d[d["format"] == "jpeg"].sort_values("test_quality")
            if s.empty:
                continue
            ax.plot(s["test_quality"], s[col] * 100, linestyle=ls, color=color,
                    linewidth=1.2, alpha=0.5, zorder=1)
            ax.plot(s["test_quality"], s[col] * 100, linestyle="none", marker=marker,
                    color=color, markersize=6, label=flabel, zorder=3)
        b = _baseline(model, col)
        if b is not None:
            ax.axhline(b * 100, color="black", linestyle=":", linewidth=1.4,
                       label="Baseline (test PNG)", zorder=2)
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_xlabel("Poziom kompresji zbioru testowego Q", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.invert_xaxis()
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("Eksperyment C: trening na mieszanej kompresji likwiduje spadek "
                 "JPEG z Eksperymentu B", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {PLOTS / out_name}")


def main():
    _metric_vs_q("test_map", "mAP [%]",
                 "Eksperyment A: mAP w funkcji poziomu kompresji Q", "fig_exp_a_map.pdf")
    _metric_vs_q("test_f1_macro", "F1-macro [%]",
                 "Eksperyment A: F1-macro w funkcji poziomu kompresji Q", "fig_exp_a_f1.pdf")
    quality_figure()
    map_vs_psnr_figure()
    exp_b_figure()
    exp_c_figure()


if __name__ == "__main__":
    main()
