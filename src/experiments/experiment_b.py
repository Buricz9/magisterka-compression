"""Experiment B: Train on baseline, test on different quality levels."""
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
from core.isic_dataset import get_isic_dataloader


def run_experiment_b(model_name, task, quality_levels, num_epochs, batch_size, device, format='jpeg', dataset='arcade'):
    """Run Experiment B: train on baseline, test on compressed."""
    # Set seed for reproducibility
    config.set_seed()

    # Import appropriate dataset loader
    if dataset == 'isic':
        get_dataloader_func = get_isic_dataloader
        num_classes = config.ISIC_NUM_CLASSES
        # For ISIC, we need a dummy task since train_model requires it
        task_param = 'isic'  # Dummy task name for ISIC
    else:
        get_dataloader_func = get_dataloader
        num_classes = config.NUM_CLASSES.get(task, 26)
        task_param = task

    # Train on baseline
    print(f"\n{'='*80}")
    print(f"Training on baseline")
    print(f"{'='*80}")

    training_results = train_model(
        model_name=model_name,
        task=task_param,
        train_quality=None,  # Baseline
        val_quality=None,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=config.LEARNING_RATE,
        device=device,
        train_format=None,
        val_format=None
    )

    # Load best model
    experiment_id = training_results['experiment_id']
    checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"

    try:
        if dataset == 'isic':
            temp_loader = get_dataloader_func(split='test', quality=None, format=None,
                                              batch_size=batch_size, num_workers=config.NUM_WORKERS)
        else:
            temp_loader = get_dataloader_func(task, 'test', quality=None, format=None,
                                              batch_size=batch_size, num_workers=config.NUM_WORKERS)
        num_classes = temp_loader.dataset.num_classes
        model = create_model(model_name, num_classes).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError as e:
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        raise
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        raise

    # Test on different quality levels
    results = []
    for quality in quality_levels:
        print(f"\nTesting on {format.upper()} Q={quality}")
        if dataset == 'isic':
            test_loader = get_dataloader_func(split='test', quality=quality, format=format,
                                              batch_size=batch_size, num_workers=config.NUM_WORKERS)
        else:
            test_loader = get_dataloader_func(task, 'test', quality=quality, format=format,
                                              batch_size=batch_size, num_workers=config.NUM_WORKERS)
        test_metrics = evaluate_model(model, test_loader, device)

        results.append({
            'experiment_id': experiment_id,
            'format': format,
            'train_quality': 'baseline',
            'test_quality': quality,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_macro': test_metrics['f1_macro']
        })

    # Save results
    df = pd.DataFrame(results)
    output_dir = config.RESULTS_ROOT / "experiment_b"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use appropriate filename based on dataset
    if dataset == 'isic':
        output_file = output_dir / f"{model_name}_{dataset}_{format}_results.csv"
    else:
        output_file = output_dir / f"{model_name}_{dataset}_{task_param}_{format}_results.csv"

    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=config.SUPPORTED_MODELS)
    parser.add_argument("--task", type=str, default="syntax", choices=["syntax", "stenosis"],
                       help="ARCADE task (ignored for ISIC)")
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "jpeg2000", "avif"],
                       help="Compression format")
    parser.add_argument("--dataset", type=str, default="arcade", choices=["arcade", "isic"],
                       help="Dataset to use (arcade or isic)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--mvp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    quality_levels = config.QUALITY_LEVELS_MVP if args.mvp else config.QUALITY_LEVELS

    # For ISIC, ignore task parameter
    task = args.task if args.dataset == 'arcade' else None

    run_experiment_b(
        model_name=args.model,
        task=task,
        quality_levels=quality_levels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        format=args.format,
        dataset=args.dataset
    )


if __name__ == "__main__":
    main()
