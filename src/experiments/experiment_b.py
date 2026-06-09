"""Experiment B: train on raw PNG (baseline model), test on COMPRESSED images.

This is the mirror image of Experiment A. The model is the already-trained
baseline (trained and validated on uncompressed PNG); here it is only evaluated
on compressed test images, separately for every (format, quality) cell.

Scenario: a model deployed in a research/clean environment, then fed a
compression-degraded diagnostic stream (e.g. a PACS / telemedicine pipeline).
Research question: how much does test-time compression hurt a model that never
saw compression artifacts during training?

No training happens here — only inference on existing baseline checkpoints, so
the whole experiment is fast (minutes per model).
"""
import sys
import gc
from pathlib import Path
import argparse

import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.core.train import evaluate_model, create_model
from src.core.dataset import get_dataloader

_FORMAT_EXTENSIONS = {'jpeg': '.jpg', 'jpeg2000': '.jp2', 'avif': '.avif'}


def _find_baseline_checkpoint(model_name, task):
    """Locate the baseline model's checkpoint via its results CSV.

    The baseline run (run_baseline.py) writes a single-row CSV with the
    experiment_id; the checkpoint lives under models/checkpoints/<experiment_id>.
    Returns (experiment_id, checkpoint_path). Raises if either is missing.
    """
    csv_path = (config.RESULTS_ROOT / "experiment_a"
                / f"{model_name}_arcade_{task}_baseline_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Baseline results CSV not found: {csv_path}\n"
            f"Run `python -m src.experiments.run_baseline --model {model_name} "
            f"--task {task}` first."
        )
    experiment_id = str(pd.read_csv(csv_path).iloc[0]['experiment_id'])
    checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint missing: {checkpoint_path}\n"
            f"The CSV references experiment_id={experiment_id} but its checkpoint "
            f"is not on disk — re-run the baseline."
        )
    return experiment_id, checkpoint_path


def _preflight_check_compressed_test(task, quality_levels, formats):
    """Fail fast if any compressed TEST cell (format, quality) is missing."""
    expected_dir = config.get_data_path(task, 'test', quality=None)
    expected = len(list(expected_dir.glob('*.png'))) if expected_dir.exists() else None
    problems = []
    for fmt in formats:
        ext = _FORMAT_EXTENSIONS.get(fmt, '.jpg')
        for q in quality_levels:
            d = config.get_data_path(task, 'test', quality=q, format=fmt)
            if not d.exists():
                problems.append(f"{d}  (directory missing)")
                continue
            n = len(list(d.glob(f'*{ext}')))
            if n == 0:
                problems.append(f"{d}  (0 {ext} files)")
            elif expected is not None and n < expected:
                problems.append(f"{d}  ({n} {ext}, expected {expected} — incomplete)")
    if problems:
        raise FileNotFoundError(
            "Compressed TEST images missing/incomplete for Experiment B:\n  "
            + "\n  ".join(problems)
            + "\n\nRun `python -m src.processing.compress_images --format all "
            "--task syntax --split test` first."
        )


def _append_row_to_csv(row, model_name, task):
    """Persist one (format, quality) result immediately (crash-safe).

    Merges on (format, test_quality) so a partial rerun replaces only matching
    cells, preserving the rest. One CSV per model holds all 3 formats.
    """
    df = pd.DataFrame([row])
    output_dir = config.RESULTS_ROOT / "experiment_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_arcade_{task}_results.csv"

    if output_file.exists():
        existing = pd.read_csv(output_file)
        mask = ~((existing['format'] == row['format'])
                 & (existing['test_quality'] == row['test_quality']))
        existing = existing[mask]
        df = pd.concat([existing, df], ignore_index=True)
        sort_key = pd.to_numeric(df['test_quality'], errors='coerce')
        df = (df.assign(_fmt=df['format'], _sort=sort_key)
                .sort_values(['_fmt', '_sort'], ascending=[True, False],
                             na_position='last')
                .drop(columns=['_fmt', '_sort'])
                .reset_index(drop=True))

    df.to_csv(output_file, index=False)
    return output_file


def _done_cells(model_name, task):
    """Set of (format, quality) cells already present in the results CSV."""
    output_file = (config.RESULTS_ROOT / "experiment_b"
                   / f"{model_name}_arcade_{task}_results.csv")
    if not output_file.exists():
        return set()
    try:
        existing = pd.read_csv(output_file)
    except (pd.errors.EmptyDataError, OSError):
        return set()
    done = set()
    for _, r in existing.iterrows():
        try:
            done.add((str(r['format']), int(r['test_quality'])))
        except (ValueError, TypeError, KeyError):
            pass
    return done


def run_experiment_b(model_name, task, quality_levels, formats, batch_size,
                     device, force=False):
    """Evaluate the baseline (PNG-trained) model on compressed test images.

    For each (format, quality) the same baseline checkpoint is loaded once and
    evaluated on the compressed test split. CSV is updated after every cell.
    """
    config.set_seed()
    _preflight_check_compressed_test(task, quality_levels, formats)

    experiment_id, checkpoint_path = _find_baseline_checkpoint(model_name, task)
    print(f"Baseline checkpoint: {experiment_id}")

    # Load the model once; it is reused across every (format, quality) cell.
    # We need num_classes — read it from any test loader (all share the dataset).
    probe = get_dataloader(task, 'test', quality=None, format=None,
                           batch_size=batch_size, num_workers=config.NUM_WORKERS)
    num_classes = probe.dataset.num_classes
    del probe

    model = create_model(model_name, num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    baseline_val = float(checkpoint.get('val_score', float('nan')))

    done = set() if force else _done_cells(model_name, task)
    if done:
        print(f"Resume: {len(done)} cell(s) already done, will skip "
              f"(use --force to recompute).")

    n_run = 0
    for fmt in formats:
        for q in quality_levels:
            if not force and (fmt, q) in done:
                continue
            print(f"\n{'='*80}\nEval baseline on {fmt.upper()} test, Q={q}\n{'='*80}")

            test_loader = get_dataloader(
                task, 'test', quality=q, format=fmt,
                batch_size=batch_size, num_workers=config.NUM_WORKERS,
            )
            test_metrics = evaluate_model(model, test_loader, device)

            row = {
                'experiment_id': experiment_id,
                'format': fmt,
                'train_quality': 'baseline',   # trained on raw PNG
                'test_quality': q,             # tested on compressed Q
                'baseline_val_score': baseline_val,
            }
            row.update({f'test_{k}': v for k, v in test_metrics.items()})

            output_file = _append_row_to_csv(row, model_name, task)
            print(f"Row saved ({fmt} Q={q}) -> {output_file}")
            n_run += 1

            del test_loader
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f"\n{n_run} cell(s) evaluated this run for {model_name} "
          f"({len(done)} skipped as already done).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=config.SUPPORTED_MODELS)
    parser.add_argument("--task", type=str, default="syntax", choices=["syntax"])
    parser.add_argument("--format", type=str, default="all",
                        choices=["jpeg", "jpeg2000", "avif", "all"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--quality", type=int, default=None,
                        choices=config.QUALITY_LEVELS,
                        help="Evaluate a single quality level only.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true",
                        help="Recompute cells already present in the CSV.")
    args = parser.parse_args()

    formats = config.COMPRESSION_FORMATS if args.format == "all" else [args.format]
    quality_levels = [args.quality] if args.quality is not None else config.QUALITY_LEVELS

    run_experiment_b(
        model_name=args.model,
        task=args.task,
        quality_levels=quality_levels,
        formats=formats,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        force=args.force,
    )


if __name__ == "__main__":
    main()
