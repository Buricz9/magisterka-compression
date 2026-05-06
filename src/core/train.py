"""Training pipeline for ARCADE dataset."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import timm
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, hamming_loss
import numpy as np
import platform
from torch import __version__ as torch_version
import torchvision
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.core.dataset import get_dataloader


def get_system_metadata():
    """Collect system and library metadata for reproducibility."""
    return {
        'python_version': platform.python_version(),
        'pytorch_version': torch_version,
        'torchvision_version': torchvision.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cuda_available': torch.cuda.is_available(),
        'timm_version': timm.__version__,
        'numpy_version': np.__version__,
        'sklearn_version': sklearn.__version__,
    }


def create_model(model_name, num_classes):
    """Create model using timm library."""
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def _epoch_score(outputs_all, labels_all, multi_label):
    """Primary epoch-level score for early stopping / best checkpoint.

    Multi-label: macro F1 at threshold 0.5 (in %).
    Single-label: top-1 accuracy (in %).
    """
    if multi_label:
        preds = (outputs_all > 0).astype(np.int8)  # logits > 0 == sigmoid > 0.5
        return 100.0 * f1_score(labels_all, preds, average='macro', zero_division=0)
    return 100.0 * accuracy_score(labels_all, outputs_all.argmax(axis=1))


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=True, multi_label=False):
    """Train for one epoch with optional AMP."""
    model.train()
    running_loss, total = 0.0, 0
    out_chunks, lbl_chunks = [], []

    for images, labels, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if multi_label:
            labels = labels.float()

        if scaler is not None:
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        out_chunks.append(outputs.detach().float().cpu().numpy())
        lbl_chunks.append(labels.detach().cpu().numpy())

    outputs_all = np.concatenate(out_chunks)
    labels_all = np.concatenate(lbl_chunks)
    score = _epoch_score(outputs_all, labels_all, multi_label)
    return running_loss / total, score


def validate(model, dataloader, criterion, device, use_amp=True, multi_label=False):
    """Validate the model with optional AMP for consistency."""
    model.eval()
    running_loss, total = 0.0, 0
    out_chunks, lbl_chunks = [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if multi_label:
                labels = labels.float()

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            out_chunks.append(outputs.detach().float().cpu().numpy())
            lbl_chunks.append(labels.detach().cpu().numpy())

    outputs_all = np.concatenate(out_chunks)
    labels_all = np.concatenate(lbl_chunks)
    score = _epoch_score(outputs_all, labels_all, multi_label)
    return running_loss / total, score


def evaluate_model(model, dataloader, device, use_amp=True):
    """Evaluate model. Returns metrics appropriate for the task type.

    Multi-label: subset_accuracy, hamming_accuracy, f1_macro, f1_micro, mAP.
    Single-label: accuracy, f1_macro, f1_weighted.
    """
    multi_label = bool(getattr(dataloader.dataset, 'multi_label', False))
    model.eval()
    out_chunks, lbl_chunks = [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            out_chunks.append(outputs.detach().float().cpu().numpy())
            lbl_chunks.append(labels.numpy())

    outputs_all = np.concatenate(out_chunks)
    labels_all = np.concatenate(lbl_chunks)

    if multi_label:
        scores = 1.0 / (1.0 + np.exp(-outputs_all))  # sigmoid for mAP
        preds = (outputs_all > 0).astype(np.int8)
        # mAP only over classes with at least one positive sample
        present = labels_all.sum(axis=0) > 0
        if present.any():
            map_score = float(average_precision_score(
                labels_all[:, present], scores[:, present], average='macro'
            ))
        else:
            map_score = 0.0
        return {
            'subset_accuracy': float((preds == labels_all).all(axis=1).mean()),
            'hamming_accuracy': float(1.0 - hamming_loss(labels_all, preds)),
            'f1_macro': float(f1_score(labels_all, preds, average='macro', zero_division=0)),
            'f1_micro': float(f1_score(labels_all, preds, average='micro', zero_division=0)),
            'map': map_score,
        }

    preds = outputs_all.argmax(axis=1)
    return {
        'accuracy': float(accuracy_score(labels_all, preds)),
        'f1_macro': float(f1_score(labels_all, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(labels_all, preds, average='weighted', zero_division=0)),
    }


def train_model(model_name, task, train_quality, val_quality, num_epochs,
                batch_size, learning_rate, device, train_format=None, val_format=None,
                experiment_id=None, early_stopping_patience=None, use_amp=True):
    """Train a model on ARCADE dataset."""
    # Set seed for reproducibility
    config.set_seed()

    if early_stopping_patience is None:
        early_stopping_patience = config.EARLY_STOPPING_PATIENCE

    if experiment_id is None:
        quality_str = f"q{train_quality}" if train_quality else "baseline"
        format_str = f"_{train_format}" if train_format and train_format != 'jpeg' else ""
        experiment_id = f"{model_name}_{task}_{quality_str}{format_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nTraining: {experiment_id}")
    print(f"Model: {model_name}, Task: {task}, Epochs: {num_epochs}")
    if train_format:
        print(f"Format: {train_format}")
    print(f"AMP: {'Enabled' if use_amp else 'Disabled'}")

    # Create dataloaders with optimal num_workers
    train_loader = get_dataloader(task, 'train', train_quality, train_format, batch_size,
                                  num_workers=config.NUM_WORKERS, shuffle=True)
    val_loader = get_dataloader(task, 'val', val_quality, val_format, batch_size,
                                num_workers=config.NUM_WORKERS, shuffle=False)

    # Create model
    num_classes = train_loader.dataset.num_classes
    model = create_model(model_name, num_classes).to(device)

    multi_label = bool(getattr(train_loader.dataset, 'multi_label', False))
    if multi_label:
        # pos_weight handles ARCADE's 18x class imbalance in BCE
        if hasattr(train_loader.dataset, 'compute_pos_weight'):
            pos_weight = train_loader.dataset.compute_pos_weight().to(device)
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        primary_metric = 'f1_macro'
    else:
        criterion = nn.CrossEntropyLoss()
        primary_metric = 'accuracy'
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # AMP scaler - only enabled for CUDA devices
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')

    # Training loop. score = F1-macro (multi-label) or top-1 accuracy (single-label), in %.
    best_val_score, best_epoch, patience_counter = 0.0, 0, 0
    history = {'train_loss': [], 'train_score': [], 'val_loss': [], 'val_score': []}
    metric_label = 'F1-macro' if multi_label else 'Acc'

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_score = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, multi_label)
        val_loss, val_score = validate(model, val_loader, criterion, device, use_amp, multi_label)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_score'].append(train_score)
        history['val_loss'].append(val_loss)
        history['val_score'].append(val_score)

        print(f"Train - Loss: {train_loss:.4f}, {metric_label}: {train_score:.2f}%")
        print(f"Val - Loss: {val_loss:.4f}, {metric_label}: {val_score:.2f}%")

        if val_score > best_val_score:
            best_val_score, best_epoch, patience_counter = val_score, epoch, 0
            checkpoint_dir = config.get_checkpoint_path(experiment_id)
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score,
                'primary_metric': primary_metric,
            }, checkpoint_path)
            print(f"Saved best model: {val_score:.2f}% ({metric_label})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save results with full metadata for reproducibility
    results = {
        'experiment_id': experiment_id,
        'model_name': model_name,
        'task': task,
        'train_quality': train_quality,
        'val_quality': val_quality,
        'train_format': train_format,
        'val_format': val_format,
        'best_epoch': best_epoch,
        'best_val_score': best_val_score,
        'primary_metric': primary_metric,
        'multi_label': multi_label,
        'history': history,
        'use_amp': use_amp,

        # REPRODUCIBILITY METADATA
        'reproducibility': {
            'random_seed': config.RANDOM_SEED,
            'system': get_system_metadata(),
            'device': str(device),
        },

        # HYPERPARAMETERS
        'hyperparameters': {
            'learning_rate': learning_rate,
            'weight_decay': config.WEIGHT_DECAY,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'optimizer': 'Adam',
            'scheduler': 'CosineAnnealingLR',
            'early_stopping_patience': early_stopping_patience,
            'criterion': 'BCEWithLogitsLoss' if multi_label else 'CrossEntropyLoss',
            'pos_weight_used': bool(multi_label),
        },

        # EXPERIMENT DESIGN
        'experiment_design': {
            'data_augmentation': True,
            'target_size': (224, 224),
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
            'num_workers': config.NUM_WORKERS,
            'pin_memory': True,
            'persistent_workers': True
        }
    }

    results_dir = config.get_results_path(experiment_id)
    with open(results_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest Val {metric_label}: {best_val_score:.2f}% (epoch {best_epoch})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=config.SUPPORTED_MODELS)
    parser.add_argument("--task", type=str, default="syntax", choices=["syntax", "stenosis"])
    parser.add_argument("--train-quality", type=int, default=None)
    parser.add_argument("--val-quality", type=int, default=None)
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "jpeg2000", "avif"],
                       help="Compression format")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (mixed precision)")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        task=args.task,
        train_quality=args.train_quality,
        val_quality=args.val_quality,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=torch.device(args.device),
        train_format=args.format,
        val_format=args.format,
        use_amp=not args.no_amp
    )


if __name__ == "__main__":
    main()
