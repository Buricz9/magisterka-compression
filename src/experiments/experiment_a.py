"""Experiment A: Train on different quality levels, test on baseline."""
import sys
from pathlib import Path
import torch
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config
from core.train import train_model, evaluate_model, create_model
from core.dataset import get_dataloader


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
        df = df.sort_values('train_quality', ascending=False).reset_index(drop=True)

    df.to_csv(output_file, index=False)
    return output_file


def _preflight_check_compressed(task, quality_levels, format):
    """Fail fast if compressed images for any (split, quality) are missing.

    Without this the trainer would crash mid-epoch from a worker process
    FileNotFoundError, which is hard to debug. The check is fast (just
    file count) and runs before any GPU work begins.
    """
    missing = []
    for quality in quality_levels:
        for split in ('train', 'val'):
            d = config.get_data_path(task, split, quality=quality, format=format)
            n = len(list(d.glob('*'))) if d.exists() else 0
            if n == 0:
                missing.append(f"{d}  (0 files)")
    if missing:
        raise FileNotFoundError(
            "Compressed images missing for the requested quality levels:\n  "
            + "\n  ".join(missing)
            + "\n\nRun `python -m src.processing.compress_images --format all "
            "--task syntax --split all` first."
        )


def run_experiment_a(model_name, task, quality_levels, num_epochs, batch_size,
                     device, format='jpeg'):
    """Run Experiment A: train on compressed, test on baseline.

    CSV is updated AFTER EVERY quality level — if the run crashes at
    iteration 7/13, the first 6 rows are already persisted.
    """
    import gc
    config.set_seed()
    _preflight_check_compressed(task, quality_levels, format)

    results = []

    for quality in quality_levels:
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

    print(f"\nAll {len(results)} quality levels finished for {model_name}/{format}")
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
                       help="Train on a single quality level only (overrides --mvp)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    )


if __name__ == "__main__":
    main()
