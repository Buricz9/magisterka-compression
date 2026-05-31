"""
Article figures for Experiment A (both architectures, all 3 formats, 13 Q).

Produces publication-ready PDFs in plots/:
  - fig_exp_a_map.pdf   : mAP vs Q, ResNet-50 + EfficientNet-B0, baseline line
  - fig_exp_a_f1.pdf    : F1-macro vs Q, same layout
  - fig_quality.pdf     : PSNR / SSIM / CR vs Q for the 3 formats
  - fig_map_vs_psnr.pdf : mAP vs mean PSNR scatter (per format), per model

Run: python -m src.analysis.generate_article_figures
"""
import sys
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _metric_vs_q(col, ylabel, title, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (model, mlabel) in zip(axes, MODELS):
        for fmt, flabel, color, marker, ls in FORMATS:
            d = _load(model, fmt)
            if d is None or d.empty:
                continue
            ax.plot(d["train_quality"], d[col] * 100, marker=marker, color=color,
                    linestyle=ls, linewidth=1.8, markersize=6, label=flabel)
        b = _baseline(model, col)
        if b is not None:
            ax.axhline(b * 100, color="black", linestyle=":", linewidth=1.4,
                       label="Baseline (PNG)")
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_xlabel("Poziom jakości Q", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.invert_xaxis()  # Q decreases left->right => compression increases right
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
        for f in glob.glob(str(config.RESULTS_ROOT / "metrics" / f"quality_syntax_train_{fmt}.csv")):
            frames.append(pd.read_csv(f))
    if not frames:
        print("no train quality CSVs - skipping fig_map_vs_psnr")
        return
    q = pd.concat(frames, ignore_index=True)
    qa = q.groupby(["format", "quality"]).agg(psnr=("psnr", "mean")).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (model, mlabel) in zip(axes, MODELS):
        for fmt, flabel, color, marker, ls in FORMATS:
            d = _load(model, fmt)
            if d is None or d.empty:
                continue
            m = d.merge(qa[qa["format"] == fmt], left_on="train_quality",
                        right_on="quality", how="inner")
            ax.scatter(m["psnr"], m["test_map"] * 100, color=color, marker=marker,
                       s=45, label=flabel, alpha=0.85)
        b = _baseline(model, "test_map")
        if b is not None:
            ax.axhline(b * 100, color="black", linestyle=":", linewidth=1.4,
                       label="Baseline (PNG)")
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_xlabel("Średni PSNR danych treningowych [dB]", fontsize=11)
        ax.set_ylabel("mAP testowe [%]", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Skuteczność klasyfikacji a wierność percepcyjna danych treningowych",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig_map_vs_psnr.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {PLOTS / 'fig_map_vs_psnr.pdf'}")


def main():
    _metric_vs_q("test_map", "mAP [%]",
                 "Eksperyment A: mAP w funkcji poziomu kompresji Q", "fig_exp_a_map.pdf")
    _metric_vs_q("test_f1_macro", "F1-macro [%]",
                 "Eksperyment A: F1-macro w funkcji poziomu kompresji Q", "fig_exp_a_f1.pdf")
    quality_figure()
    map_vs_psnr_figure()


if __name__ == "__main__":
    main()
