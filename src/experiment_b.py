"""Experiment B: Train on baseline, test on different quality levels."""
import sys
from pathlib import Path
import torch
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from train import train_model, evaluate_model, create_model
from dataset import get_dataloader


def run_experiment_b(model_name, task, quality_levels, num_epochs, batch_size, device, format='jpeg'):
    """Run Experiment B: train on baseline, test on compressed."""
    # Train on baseline
    print(f"\n{'='*80}")
    print(f"Training on baseline")
    print(f"{'='*80}")

    training_results = train_model(
        model_name=model_name,
        task=task,
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

    num_classes = config.NUM_CLASSES[task]
    model = create_model(model_name, num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test on different quality levels
    results = []
    for quality in quality_levels:
        print(f"\nTesting on {format.upper()} Q={quality}")
        test_loader = get_dataloader(task, 'test', quality=quality, format=format, batch_size=batch_size, num_workers=0)
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
    output_file = output_dir / f"{model_name}_{task}_{format}_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--task", type=str, default="syntax", choices=["syntax", "stenosis"])
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "jpeg2000", "avif"],
                       help="Compression format")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mvp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    quality_levels = config.QUALITY_LEVELS_MVP if args.mvp else config.QUALITY_LEVELS

    run_experiment_b(
        model_name=args.model,
        task=args.task,
        quality_levels=quality_levels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        format=args.format
    )


if __name__ == "__main__":
    main()
