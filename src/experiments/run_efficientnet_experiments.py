"""
Run EfficientNet-B0 experiments for both ARCADE and ISIC 2019 datasets.

This script executes all required EfficientNet-B0 experiments as specified
by the promoter (16.02.2026 decision).

Experiments:
- Experiment A: Train on compressed, test on original
- Experiment B: Train on original, test on compressed
- Datasets: ARCADE (syntax), ISIC 2019 (benchmark)
- Formats: JPEG, JPEG2000, AVIF
- Quality levels: MVP [100, 85, 70, 50, 30, 10]
"""
import sys
from pathlib import Path
import torch
import argparse
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from train import train_model, evaluate_model, create_model
from dataset import get_dataloader
from isic_dataset import get_isic_dataloader, check_isic_available


def run_efficientnet_arcae_experiment_a(
    quality_levels=None,
    formats=None,
    num_epochs=30,
    batch_size=16,
    device='cuda'
):
    """Run Experiment A on ARCADE with EfficientNet-B0."""
    if quality_levels is None:
        quality_levels = config.QUALITY_LEVELS_MVP
    if formats is None:
        formats = config.COMPRESSION_FORMATS

    model_name = "efficientnet_b0"
    task = "syntax"
    all_results = []

    print(f"\n{'='*80}")
    print(f"EFFICIENTNET-B0 EXPERIMENT A: ARCADE {task.upper()}")
    print(f"{'='*80}")

    for fmt in formats:
        for quality in quality_levels:
            print(f"\n>>> Training: {model_name} | {fmt.upper()} | Q={quality}")

            try:
                # Train on compressed
                training_results = train_model(
                    model_name=model_name,
                    task=task,
                    train_quality=quality,
                    val_quality=None,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=config.LEARNING_RATE,
                    device=torch.device(device),
                    train_format=fmt,
                    val_format=None
                )

                # Load best model and test on original
                experiment_id = training_results['experiment_id']
                checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"

                test_loader = get_dataloader(
                    task, 'test', quality=None, format=None,
                    batch_size=batch_size, num_workers=config.NUM_WORKERS
                )
                num_classes = test_loader.dataset.num_classes
                model = create_model(model_name, num_classes).to(torch.device(device))
                checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
                model.load_state_dict(checkpoint['model_state_dict'])

                test_metrics = evaluate_model(model, test_loader, torch.device(device))

                all_results.append({
                    'experiment_id': experiment_id,
                    'dataset': 'arcade',
                    'task': task,
                    'model': model_name,
                    'format': fmt,
                    'train_quality': quality,
                    'test_quality': 'baseline',
                    'best_val_acc': training_results['best_val_acc'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1_macro': test_metrics['f1_macro'],
                    'test_f1_weighted': test_metrics['f1_weighted']
                })

                print(f"Test Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_macro']:.4f}")

            except Exception as e:
                print(f"[ERROR] Failed for {fmt} Q={quality}: {e}")
                all_results.append({
                    'dataset': 'arcade',
                    'task': task,
                    'model': model_name,
                    'format': fmt,
                    'train_quality': quality,
                    'error': str(e)
                })

    return all_results


def run_efficientnet_arcae_experiment_b(
    quality_levels=None,
    formats=None,
    num_epochs=30,
    batch_size=16,
    device='cuda'
):
    """Run Experiment B on ARCADE with EfficientNet-B0."""
    if quality_levels is None:
        quality_levels = config.QUALITY_LEVELS_MVP
    if formats is None:
        formats = config.COMPRESSION_FORMATS

    model_name = "efficientnet_b0"
    task = "syntax"
    all_results = []

    print(f"\n{'='*80}")
    print(f"EFFICIENTNET-B0 EXPERIMENT B: ARCADE {task.upper()}")
    print(f"{'='*80}")

    # Train on baseline once
    print(f"\n>>> Training: {model_name} | BASELINE")

    training_results = train_model(
        model_name=model_name,
        task=task,
        train_quality=None,
        val_quality=None,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=config.LEARNING_RATE,
        device=torch.device(device),
        train_format=None,
        val_format=None
    )

    # Load best model
    experiment_id = training_results['experiment_id']
    checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"

    temp_loader = get_dataloader(
        task, 'test', quality=None, format=None,
        batch_size=batch_size, num_workers=config.NUM_WORKERS
    )
    num_classes = temp_loader.dataset.num_classes
    model = create_model(model_name, num_classes).to(torch.device(device))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test on different quality levels
    for fmt in formats:
        for quality in quality_levels:
            print(f"\n>>> Testing: {fmt.upper()} | Q={quality}")

            try:
                test_loader = get_dataloader(
                    task, 'test', quality=quality, format=fmt,
                    batch_size=batch_size, num_workers=config.NUM_WORKERS
                )
                test_metrics = evaluate_model(model, test_loader, torch.device(device))

                all_results.append({
                    'experiment_id': experiment_id,
                    'dataset': 'arcade',
                    'task': task,
                    'model': model_name,
                    'format': fmt,
                    'train_quality': 'baseline',
                    'test_quality': quality,
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1_macro': test_metrics['f1_macro'],
                    'test_f1_weighted': test_metrics['f1_weighted']
                })

                print(f"Test Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_macro']:.4f}")

            except Exception as e:
                print(f"[ERROR] Failed for {fmt} Q={quality}: {e}")
                all_results.append({
                    'dataset': 'arcade',
                    'task': task,
                    'model': model_name,
                    'format': fmt,
                    'train_quality': 'baseline',
                    'test_quality': quality,
                    'error': str(e)
                })

    return all_results


def run_efficientnet_isic_experiment_a(
    quality_levels=None,
    formats=None,
    num_epochs=30,
    batch_size=16,
    device='cuda'
):
    """Run Experiment A on ISIC 2019 with EfficientNet-B0."""
    if quality_levels is None:
        quality_levels = config.QUALITY_LEVELS_MVP
    if formats is None:
        formats = config.COMPRESSION_FORMATS

    model_name = "efficientnet_b0"
    all_results = []

    if not check_isic_available():
        print("[WARNING] ISIC 2019 dataset not found. Skipping ISIC experiments.")
        print("To run ISIC experiments:")
        print("1. Download from https://challenge.isic-archive.com/landing/2019/")
        print("2. Run: python src/preprocess_isic.py --input-root <download_dir>")
        print("3. Run: python src/compress_isic.py --format all --mvp")
        return all_results

    print(f"\n{'='*80}")
    print(f"EFFICIENTNET-B0 EXPERIMENT A: ISIC 2019")
    print(f"{'='*80}")

    for fmt in formats:
        for quality in quality_levels:
            print(f"\n>>> Training: {model_name} | {fmt.upper()} | Q={quality}")

            try:
                # Train on compressed
                experiment_id = f"{model_name}_isic_{fmt}_q{quality}_expA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                train_loader = get_isic_dataloader(
                    split='train',
                    quality=quality,
                    format=fmt,
                    batch_size=batch_size,
                    shuffle=True
                )

                val_loader = get_isic_dataloader(
                    split='val',
                    quality=None,
                    format=None,
                    batch_size=batch_size,
                    shuffle=False
                )

                num_classes = train_loader.dataset.num_classes
                model = create_model(model_name, num_classes).to(torch.device(device))

                # Train
                training_results = train_model(
                    model_name=model_name,
                    task='isic',
                    train_quality=quality,
                    val_quality=None,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=config.LEARNING_RATE,
                    device=torch.device(device),
                    train_format=fmt,
                    val_format=None,
                    experiment_id=experiment_id
                )

                # Test on originals
                test_loader = get_isic_dataloader(
                    split='test',
                    quality=None,
                    format=None,
                    batch_size=batch_size,
                    shuffle=False
                )

                checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"
                checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
                model.load_state_dict(checkpoint['model_state_dict'])

                test_metrics = evaluate_model(model, test_loader, torch.device(device))

                all_results.append({
                    'experiment_id': experiment_id,
                    'dataset': 'isic_2019',
                    'model': model_name,
                    'format': fmt,
                    'train_quality': quality,
                    'test_quality': 'original',
                    'best_val_acc': training_results['best_val_acc'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1_macro': test_metrics['f1_macro'],
                    'test_f1_weighted': test_metrics['f1_weighted']
                })

                print(f"Test Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_macro']:.4f}")

            except Exception as e:
                print(f"[ERROR] Failed: {e}")
                all_results.append({
                    'dataset': 'isic_2019',
                    'model': model_name,
                    'format': fmt,
                    'train_quality': quality,
                    'error': str(e)
                })

    return all_results


def run_efficientnet_isic_experiment_b(
    quality_levels=None,
    formats=None,
    num_epochs=30,
    batch_size=16,
    device='cuda'
):
    """Run Experiment B on ISIC 2019 with EfficientNet-B0."""
    if quality_levels is None:
        quality_levels = config.QUALITY_LEVELS_MVP
    if formats is None:
        formats = config.COMPRESSION_FORMATS

    model_name = "efficientnet_b0"
    all_results = []

    if not check_isic_available():
        print("[WARNING] ISIC 2019 dataset not found. Skipping ISIC experiments.")
        return all_results

    print(f"\n{'='*80}")
    print(f"EFFICIENTNET-B0 EXPERIMENT B: ISIC 2019")
    print(f"{'='*80}")

    # Train on originals
    experiment_id = f"{model_name}_isic_original_expB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_results = train_model(
        model_name=model_name,
        task='isic',
        train_quality=None,
        val_quality=None,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=config.LEARNING_RATE,
        device=torch.device(device),
        train_format=None,
        val_format=None,
        experiment_id=experiment_id
    )

    # Load best model
    checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    temp_loader = get_isic_dataloader(split='test', quality=None, format=None, batch_size=1)
    num_classes = temp_loader.dataset.num_classes
    model = create_model(model_name, num_classes).to(torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test on different quality levels
    for fmt in formats:
        for quality in quality_levels:
            print(f"\n>>> Testing: {fmt.upper()} | Q={quality}")

            try:
                test_loader = get_isic_dataloader(
                    split='test',
                    quality=quality,
                    format=fmt,
                    batch_size=batch_size,
                    shuffle=False
                )

                test_metrics = evaluate_model(model, test_loader, torch.device(device))

                all_results.append({
                    'experiment_id': experiment_id,
                    'dataset': 'isic_2019',
                    'model': model_name,
                    'format': fmt,
                    'train_quality': 'original',
                    'test_quality': quality,
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1_macro': test_metrics['f1_macro'],
                    'test_f1_weighted': test_metrics['f1_weighted']
                })

                print(f"Test Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_macro']:.4f}")

            except Exception as e:
                print(f"[ERROR] Failed: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run EfficientNet-B0 experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=['a', 'b', 'both'],
        default='both',
        help="Which experiment to run"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['arcade', 'isic', 'both'],
        default='both',
        help="Which dataset to use"
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
    formats = config.COMPRESSION_FORMATS

    all_results = []

    # ARCADE experiments
    if args.dataset in ['arcade', 'both']:
        if args.experiment in ['a', 'both']:
            results = run_efficientnet_arcae_experiment_a(
                quality_levels=quality_levels,
                formats=formats,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )
            all_results.extend(results)

        if args.experiment in ['b', 'both']:
            results = run_efficientnet_arcae_experiment_b(
                quality_levels=quality_levels,
                formats=formats,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )
            all_results.extend(results)

    # ISIC experiments
    if args.dataset in ['isic', 'both']:
        if args.experiment in ['a', 'both']:
            results = run_efficientnet_isic_experiment_a(
                quality_levels=quality_levels,
                formats=formats,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )
            all_results.extend(results)

        if args.experiment in ['b', 'both']:
            results = run_efficientnet_isic_experiment_b(
                quality_levels=quality_levels,
                formats=formats,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )
            all_results.extend(results)

    # Save all results
    import pandas as pd

    df = pd.DataFrame(all_results)
    output_dir = config.RESULTS_ROOT / "efficientnet_b0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"efficientnet_b0_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"All EfficientNet-B0 experiments completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
