"""Experiment A: Train on different quality levels, test on baseline."""
import sys
from pathlib import Path
import torch
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.core.train import train_model, evaluate_model, create_model
from src.core.dataset import get_dataloader


def _append_row_to_csv(row, model_name, task, format):
    """Save one quality-level result to CSV immediately (crash-safe).

    Merges with existing rows so a partial rerun replaces only matching
    train_quality entries, preserving the rest.
    """
    df = pd.DataFrame([row])
    output_dir = config.RESULTS_ROOT / "experiment_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_arcade_{task}_{format}_results.csv"

    if output_file.exists():
        existing = pd.read_csv(output_file)
        existing = existing[existing['train_quality'] != row['train_quality']]
        df = pd.concat([existing, df], ignore_index=True)
        # Sort by a numeric key so the routine never raises TypeError if the
        # column is ever mixed int/str (e.g. a baseline CSV merged in by hand);
        # non-numeric train_quality values sort to the end.
        sort_key = pd.to_numeric(df['train_quality'], errors='coerce')
        df = (df.assign(_sort_key=sort_key)
                .sort_values('_sort_key', ascending=False, na_position='last')
                .drop(columns='_sort_key')
                .reset_index(drop=True))

    df.to_csv(output_file, index=False)
    return output_file


# File extension produced by the compressor for each format. Keep in sync
# with src/processing/compress_images.py::get_extension and dataset.py.
_FORMAT_EXTENSIONS = {'jpeg': '.jpg', 'jpeg2000': '.jp2', 'avif': '.avif'}


def _count_images_in_split(task, split):
    """Number of annotated images expected in a split (baseline PNG count).

    Mirrors how the dataset builds image_ids: only images that have at least
    one annotation are kept, so we count the PNG files actually present in
    the baseline split directory.
    """
    d = config.get_data_path(task, split, quality=None)
    if not d.exists():
        return None
    return len(list(d.glob('*.png')))


def _preflight_check_compressed(task, quality_levels, format):
    """Fail fast if compressed images for any (split, quality) are missing.

    Without this the trainer would crash mid-epoch from a worker process
    FileNotFoundError, which is hard to debug. The check is fast (just a
    file count) and runs before any GPU work begins.

    For each (split, quality) it counts files with the *correct* extension
    for the requested format (.jpg / .jp2 / .avif) — a directory full of
    leftover PNGs or partial output would otherwise pass a bare glob('*').
    It also flags counts that fall short of the baseline image count, and
    verifies the baseline PNG test directory exists (test stays uncompressed).
    """
    ext = _FORMAT_EXTENSIONS.get(format, '.jpg')
    problems = []

    # Train/val are compressed at every requested quality level.
    for split in ('train', 'val'):
        expected = _count_images_in_split(task, split)
        for quality in quality_levels:
            d = config.get_data_path(task, split, quality=quality, format=format)
            if not d.exists():
                problems.append(f"{d}  (directory missing)")
                continue
            n = len(list(d.glob(f'*{ext}')))
            if n == 0:
                problems.append(f"{d}  (0 {ext} files)")
            elif expected is not None and n < expected:
                problems.append(
                    f"{d}  ({n} {ext} files, expected {expected} — incomplete)"
                )

    # Test split is evaluated on baseline PNG (quality=None), not compressed.
    test_dir = config.get_data_path(task, 'test', quality=None)
    if not test_dir.exists() or not any(test_dir.glob('*.png')):
        problems.append(f"{test_dir}  (baseline PNG test split missing/empty)")

    if problems:
        raise FileNotFoundError(
            "Compressed/baseline images missing or incomplete for the "
            "requested run:\n  "
            + "\n  ".join(problems)
            + "\n\nRun `python -m src.processing.compress_images --format all "
            "--task syntax --split all` first."
        )


def _already_done_qualities(model_name, task, format):
    """Return the set of train_quality values already present in the CSV.

    Used to skip (format, quality) pairs that finished in a previous run so a
    crashed/interrupted run can be re-launched without wasting GPU time on
    work that is already persisted.
    """
    output_file = (config.RESULTS_ROOT / "experiment_a"
                   / f"{model_name}_arcade_{task}_{format}_results.csv")
    if not output_file.exists():
        return set()
    try:
        existing = pd.read_csv(output_file)
    except (pd.errors.EmptyDataError, OSError):
        return set()
    if 'train_quality' not in existing.columns:
        return set()
    # train_quality is written as int by this script; coerce defensively.
    done = set()
    for v in existing['train_quality'].tolist():
        try:
            done.add(int(v))
        except (ValueError, TypeError):
            done.add(v)
    return done


def run_experiment_a(model_name, task, quality_levels, num_epochs, batch_size,
                     device, format='jpeg', force=False):
    """Run Experiment A: train on compressed, test on baseline.

    CSV is updated AFTER EVERY quality level — if the run crashes at
    iteration 7/13, the first 6 rows are already persisted.

    Crash-safe resume: by default any quality level already present in the
    results CSV is skipped, so re-running after a crash only does the
    remaining work. Pass force=True to re-train and overwrite every level.
    """
    import gc
    config.set_seed()
    _preflight_check_compressed(task, quality_levels, format)

    done = set() if force else _already_done_qualities(model_name, task, format)
    if done:
        skipped = sorted(q for q in quality_levels if q in done)
        if skipped:
            print(f"Resume: skipping already-completed Q levels {skipped} "
                  f"(use --force to recompute)")

    results = []

    for quality in quality_levels:
        if not force and quality in done:
            continue

        print(f"\n{'='*80}")
        print(f"Training on {format.upper()} Q={quality}")
        print(f"{'='*80}")

        # Train AND validate on the same compressed domain so the best
        # checkpoint reflects model quality on the training domain rather
        # than cross-domain generalization. Test stays on baseline PNG.
        training_results = train_model(
            model_name=model_name,
            task=task,
            train_quality=quality,
            val_quality=quality,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=config.LEARNING_RATE,
            device=device,
            train_format=format,
            val_format=format,
        )

        # Load best model and evaluate on test
        experiment_id = training_results['experiment_id']
        checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"

        test_loader = get_dataloader(
            task, 'test', quality=None, format=None,
            batch_size=batch_size, num_workers=config.NUM_WORKERS,
        )
        num_classes = test_loader.dataset.num_classes
        model = create_model(model_name, num_classes).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate_model(model, test_loader, device)

        row = {
            'experiment_id': experiment_id,
            'format': format,
            'train_quality': quality,
            'test_quality': 'baseline',
            'best_val_score': training_results['best_val_score'],
            'primary_metric': training_results['primary_metric'],
        }
        row.update({f'test_{k}': v for k, v in test_metrics.items()})
        results.append(row)

        # Persist this row immediately so a later crash doesn't lose the work.
        output_file = _append_row_to_csv(row, model_name, task, format)
        print(f"Row saved (Q={quality}) -> {output_file}")

        # Release model + loaders before the next quality level to keep
        # GPU/CPU memory from creeping upward over the 13-iteration loop.
        del model, checkpoint, test_loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{len(results)} quality level(s) trained this run for "
          f"{model_name}/{format} ({len(done)} skipped as already done)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=config.SUPPORTED_MODELS)
    parser.add_argument("--task", type=str, default="syntax", choices=["syntax"],
                       help="ARCADE task")
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "jpeg2000", "avif"],
                       help="Compression format")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--mvp", action="store_true")
    parser.add_argument("--quality", type=int, default=None,
                       choices=config.QUALITY_LEVELS,
                       help="Train on a single quality level only (overrides --mvp)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true",
                       help="Re-train and overwrite quality levels already "
                            "present in the results CSV (default: skip them)")
    args = parser.parse_args()

    if args.quality is not None:
        quality_levels = [args.quality]
    elif args.mvp:
        quality_levels = config.QUALITY_LEVELS_MVP
    else:
        quality_levels = config.QUALITY_LEVELS

    run_experiment_a(
        model_name=args.model,
        task=args.task,
        quality_levels=quality_levels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        format=args.format,
        force=args.force,
    )


if __name__ == "__main__":
    main()
