"""
Experiments A and B on ISIC 2019 benchmark dataset.

Runs the same compression impact experiments on skin lesion images.
"""
import sys
from pathlib import Path
import torch
import argparse
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from train import train_model, evaluate_model, create_model
from isic_dataset import get_isic_dataloader, check_isic_available, ISIC_CLASSES


def run_isic_experiment_a(
    model_name: str = 'resnet50',
    quality_levels: list = None,
    formats: list = None,
    num_epochs: int = 30,
    batch_size: int = 16,
    device: str = 'cuda'
):
    """
    Run Experiment A on ISIC 2019: Train on compressed, test on original.

    Args:
        model_name: Model architecture
        quality_levels: Quality levels to test
        formats: Compression formats
        num_epochs: Number of training epochs
        batch_size: Batch size
        device: Device (cuda/cpu)
    """
    config.set_seed()

    if quality_levels is None:
        quality_levels = config.QUALITY_LEVELS_MVP
    if formats is None:
        formats = config.COMPRESSION_FORMATS

    if not check_isic_available():
        print("[ERROR] ISIC 2019 dataset not found!")
        print("Please download and preprocess the dataset first:")
        print("  1. Download from https://challenge.isic-archive.com/landing/2019/")
        print("  2. Run: python src/preprocess_isic.py --input-root <download_dir>")
        return

    results = []

    for fmt in formats:
        for quality in quality_levels:
            print(f"\n{'='*60}")
            print(f"ISIC Experiment A: {model_name} | {fmt.upper()} | Q={quality}")
            print(f"{'='*60}")

            # Train on compressed
            experiment_id = f"{model_name}_isic_{fmt}_q{quality}_expA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            try:
                # Load compressed training data
                train_loader = get_isic_dataloader(
                    split='train',
                    quality=quality,
                    format=fmt,
                    batch_size=batch_size,
                    shuffle=True
                )

                val_loader = get_isic_dataloader(
                    split='val',
                    quality=None,  # Validate on originals
                    format=None,
                    batch_size=batch_size,
                    shuffle=False
                )

                # Create model - get num_classes from dataset
                num_classes = train_loader.dataset.num_classes
                model = create_model(model_name, num_classes).to(device)

                # Train
                training_results = train_model(
                    model_name=model_name,
                    task='isic',  # Special marker
                    train_quality=quality,
                    val_quality=None,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=config.LEARNING_RATE,
                    device=device,
                    train_format=fmt,
                    val_format=None,
                    experiment_id=experiment_id,
                    use_amp=True
                )

                # Test on originals
                test_loader = get_isic_dataloader(
                    split='test',
                    quality=None,
                    format=None,
                    batch_size=batch_size,
                    shuffle=False
                )

                # Load best model
                checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

                test_metrics = evaluate_model(model, test_loader, device)

                results.append({
                    'experiment_id': experiment_id,
                    'dataset': 'isic_2019',
                    'format': fmt,
                    'train_quality': quality,
                    'test_quality': 'original',
                    'best_val_acc': training_results['best_val_acc'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1_macro': test_metrics['f1_macro'],
                    'test_f1_weighted': test_metrics['f1_weighted']
                })

            except Exception as e:
                print(f"[ERROR] Failed: {e}")
                results.append({
                    'experiment_id': experiment_id,
                    'dataset': 'isic_2019',
                    'format': fmt,
                    'train_quality': quality,
                    'error': str(e)
                })

    # Save results
    df = pd.DataFrame(results)
    output_dir = config.RESULTS_ROOT / "isic_experiment_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_isic_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results


def run_isic_experiment_b(
    model_name: str = 'resnet50',
    quality_levels: list = None,
    formats: list = None,
    num_epochs: int = 30,
    batch_size: int = 16,
    device: str = 'cuda'
):
    """
    Run Experiment B on ISIC 2019: Train on original, test on compressed.
    """
    config.set_seed()

    if quality_levels is None:
        quality_levels = config.QUALITY_LEVELS_MVP
    if formats is None:
        formats = config.COMPRESSION_FORMATS

    if not check_isic_available():
        print("[ERROR] ISIC 2019 dataset not found!")
        return

    # Train on originals
    print(f"\n{'='*60}")
    print(f"ISIC Experiment B: Training on originals")
    print(f"{'='*60}")

    experiment_id = f"{model_name}_isic_original_expB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_results = train_model(
        model_name=model_name,
        task='isic',
        train_quality=None,
        val_quality=None,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=config.LEARNING_RATE,
        device=device,
        train_format=None,
        val_format=None,
        experiment_id=experiment_id,
        use_amp=True
    )

    # Load best model
    checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get num_classes from dataset
    temp_loader = get_isic_dataloader(split='test', quality=None, format=None, batch_size=1)
    num_classes = temp_loader.dataset.num_classes
    model = create_model(model_name, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test on different compression levels
    results = []

    for fmt in formats:
        for quality in quality_levels:
            print(f"\nTesting on {fmt.upper()} Q={quality}")

            try:
                test_loader = get_isic_dataloader(
                    split='test',
                    quality=quality,
                    format=fmt,
                    batch_size=batch_size,
                    shuffle=False
                )

                test_metrics = evaluate_model(model, test_loader, device)

                results.append({
                    'experiment_id': experiment_id,
                    'dataset': 'isic_2019',
                    'format': fmt,
                    'train_quality': 'original',
                    'test_quality': quality,
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1_macro': test_metrics['f1_macro'],
                    'test_f1_weighted': test_metrics['f1_weighted']
                })

            except Exception as e:
                print(f"[ERROR] Failed: {e}")

    # Save results
    df = pd.DataFrame(results)
    output_dir = config.RESULTS_ROOT / "isic_experiment_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_isic_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiments on ISIC 2019")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=['a', 'b', 'both'],
        default='both',
        help="Which experiment to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=config.SUPPORTED_MODELS
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--mvp",
        action="store_true",
        help="Use MVP quality levels"
    )

    args = parser.parse_args()

    quality_levels = config.QUALITY_LEVELS_MVP if args.mvp else config.QUALITY_LEVELS

    if args.experiment in ['a', 'both']:
        print("\n" + "="*60)
        print("RUNNING ISIC EXPERIMENT A")
        print("="*60)
        run_isic_experiment_a(
            model_name=args.model,
            quality_levels=quality_levels,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=torch.device(args.device)
        )

    if args.experiment in ['b', 'both']:
        print("\n" + "="*60)
        print("RUNNING ISIC EXPERIMENT B")
        print("="*60)
        run_isic_experiment_b(
            model_name=args.model,
            quality_levels=quality_levels,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=torch.device(args.device)
        )


if __name__ == "__main__":
    main()
