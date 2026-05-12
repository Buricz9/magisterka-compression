"""Baseline run: train on raw PNG, validate on raw PNG, test on raw PNG.

Provides the upper-bound reference point for Experiment A — how well the
model performs without any compression in the training pipeline. One run
per model (format is irrelevant since baseline is uncompressed PNG).
"""
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


def run_baseline(model_name, task, num_epochs, batch_size, device):
    config.set_seed()

    print(f"\n{'='*80}\nBaseline (raw PNG) — {model_name} / {task}\n{'='*80}")

    training_results = train_model(
        model_name=model_name,
        task=task,
        train_quality=None,
        val_quality=None,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=config.LEARNING_RATE,
        device=device,
        train_format=None,
        val_format=None,
    )

    experiment_id = training_results['experiment_id']
    checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"

    test_loader = get_dataloader(task, 'test', quality=None, format=None,
                                 batch_size=batch_size, num_workers=config.NUM_WORKERS)
    num_classes = test_loader.dataset.num_classes
    model = create_model(model_name, num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(model, test_loader, device)

    row = {
        'experiment_id': experiment_id,
        'format': 'baseline',
        'train_quality': 'baseline',
        'test_quality': 'baseline',
        'best_val_score': training_results['best_val_score'],
        'primary_metric': training_results['primary_metric'],
    }
    row.update({f'test_{k}': v for k, v in test_metrics.items()})

    output_dir = config.RESULTS_ROOT / "experiment_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_arcade_{task}_baseline_results.csv"
    pd.DataFrame([row]).to_csv(output_file, index=False)
    print(f"\nBaseline result saved: {output_file}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=config.SUPPORTED_MODELS)
    parser.add_argument("--task", type=str, default="syntax", choices=["syntax"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_baseline(
        model_name=args.model,
        task=args.task,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
