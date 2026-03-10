"""Training pipeline with K-fold cross-validation for ARCADE dataset."""
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import platform
from torch import __version__ as torch_version
import torchvision
import sklearn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.core.dataset import ArcadeClassificationDataset


def create_model(model_name, num_classes):
    """Create model using timm library."""
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


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


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=True):
    """Train for one epoch with optional AMP."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels, _ in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        if scaler is not None:
            # Mixed precision training
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def validate(model, dataloader, criterion, device, use_amp=True):
    """Validate the model with optional AMP for consistency."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def evaluate_model(model, dataloader, device):
    """Evaluate model with metrics."""
    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    return {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'f1_macro': f1_score(all_labels, all_predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    }


def create_k_fold_datasets(task, k=5, seed=42):
    """
    Create K-fold train/val splits for cross-validation.

    Args:
        task: "syntax" or "stenosis"
        k: Number of folds
        seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    # Load full training dataset to get all labels
    full_train_dataset = ArcadeClassificationDataset(task, 'train', quality=None, format=None)

    # Get all labels for stratified splitting
    all_labels = [full_train_dataset.image_labels[img_id] for img_id in full_train_dataset.image_ids]

    # Create stratified K-fold split
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(all_labels)), all_labels)):
        fold_splits.append({
            'fold': fold_idx,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(),
            'train_labels': [all_labels[i] for i in train_idx],
            'val_labels': [all_labels[i] for i in val_idx]
        })

    return fold_splits, full_train_dataset


def create_fold_subset_dataset(full_dataset, indices, transform=None):
    """
    Create a subset dataset from indices.

    Args:
        full_dataset: Full ArcadeClassificationDataset
        indices: List of indices to include
        transform: Optional transforms (uses dataset's default if None)

    Returns:
        Subset dataset with specified indices
    """
    from torch.utils.data import Subset

    if transform is None:
        subset = Subset(full_dataset, indices)
    else:
        # Create new dataset with custom transform
        class TransformSubset:
            def __init__(self, dataset, indices, transform):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                actual_idx = self.indices[idx]
                image, label, image_id = self.dataset[actual_idx]
                if self.transform:
                    image = self.transform(image)
                return image, label, image_id

        subset = TransformSubset(full_dataset, indices, transform)

    # Add num_classes attribute
    subset.num_classes = full_dataset.num_classes

    return subset


def train_fold(
    model_name,
    task,
    fold_idx,
    train_dataset,
    val_dataset,
    train_quality,
    val_quality,
    num_epochs,
    batch_size,
    learning_rate,
    device,
    train_format=None,
    val_format=None,
    use_amp=True
):
    """
    Train a single fold in K-fold cross-validation.

    Returns:
        Dictionary with fold results
    """
    from torch.utils.data import DataLoader

    # Set seed for reproducibility (different seed per fold)
    fold_seed = config.RANDOM_SEED + fold_idx
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    fold_id = f"{model_name}_{task}_fold{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*80}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*80}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None
    )

    # Create model
    num_classes = train_dataset.num_classes
    model = create_model(model_name, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # AMP scaler
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')

    # Training loop
    best_val_acc, best_epoch, patience_counter = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nFold {fold_idx + 1} | Epoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience_counter = val_acc, epoch, 0
            checkpoint_dir = config.get_checkpoint_path(fold_id)
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"Saved best model: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save fold results
    fold_results = {
        'fold_id': fold_id,
        'fold': fold_idx,
        'model_name': model_name,
        'task': task,
        'train_quality': train_quality,
        'val_quality': val_quality,
        'train_format': train_format,
        'val_format': val_format,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'history': history,
        'use_amp': use_amp,
        'random_seed': fold_seed
    }

    return fold_results


def train_model_cv(
    model_name,
    task,
    train_quality,
    val_quality,
    num_epochs,
    batch_size,
    learning_rate,
    device,
    train_format=None,
    val_format=None,
    k_folds=5,
    use_amp=True
):
    """
    Train model with K-fold cross-validation.

    Args:
        model_name: Model architecture name
        task: Dataset task name
        train_quality: Compression quality for training (None for baseline)
        val_quality: Compression quality for validation (None for baseline)
        num_epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: PyTorch device
        train_format: Compression format for training
        val_format: Compression format for validation
        k_folds: Number of folds for cross-validation
        use_amp: Use automatic mixed precision

    Returns:
        Dictionary with aggregated results across all folds
    """
    # Set global seed
    config.set_seed()

    experiment_id = f"{model_name}_{task}_cv{k_folds}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    quality_str = f"q{train_quality}" if train_quality else "baseline"
    format_str = f"_{train_format}" if train_format and train_format != 'jpeg' else ""

    print(f"\n{'='*80}")
    print(f"K-Fold Cross-Validation Training: {experiment_id}")
    print(f"Model: {model_name}, Task: {task}, K: {k_folds}")
    print(f"Training: {quality_str}{format_str}, Validation: baseline")
    print(f"Epochs: {num_epochs}, AMP: {'Enabled' if use_amp else 'Disabled'}")
    print(f"{'='*80}")

    # Create K-fold splits
    fold_splits, full_train_dataset = create_k_fold_datasets(task, k=k_folds, seed=config.RANDOM_SEED)

    # Train each fold
    all_fold_results = []

    for fold_data in fold_splits:
        fold_idx = fold_data['fold']

        # Create train/val datasets for this fold
        train_dataset = create_fold_subset_dataset(
            full_train_dataset,
            fold_data['train_indices'],
            transform=None  # Use dataset's default transform (with augmentation for training)
        )

        val_dataset = create_fold_subset_dataset(
            full_train_dataset,
            fold_data['val_indices'],
            transform=None  # Use dataset's default transform (no augmentation for validation)
        )

        # Override quality/format for this fold
        train_dataset.quality = train_quality
        train_dataset.format = train_format if train_format else config.DEFAULT_FORMAT
        val_dataset.quality = val_quality
        val_dataset.format = val_format if val_format else config.DEFAULT_FORMAT

        # Train this fold
        fold_results = train_fold(
            model_name=model_name,
            task=task,
            fold_idx=fold_idx,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_quality=train_quality,
            val_quality=val_quality,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            train_format=train_format,
            val_format=val_format,
            use_amp=use_amp
        )

        all_fold_results.append(fold_results)

    # Aggregate results
    val_accs = [fold['best_val_acc'] for fold in all_fold_results]

    aggregated_results = {
        'experiment_id': experiment_id,
        'model_name': model_name,
        'task': task,
        'train_quality': train_quality,
        'val_quality': val_quality,
        'train_format': train_format,
        'val_format': val_format,
        'k_folds': k_folds,
        'num_epochs': num_epochs,
        'use_amp': use_amp,

        # Aggregated metrics
        'val_acc_mean': float(np.mean(val_accs)),
        'val_acc_std': float(np.std(val_accs)),
        'val_acc_min': float(np.min(val_accs)),
        'val_acc_max': float(np.max(val_accs)),
        'val_acc_ci_lower': float(np.mean(val_accs) - 1.96 * np.std(val_accs) / np.sqrt(k_folds)),
        'val_acc_ci_upper': float(np.mean(val_accs) + 1.96 * np.std(val_accs) / np.sqrt(k_folds)),

        # Individual fold results
        'fold_results': all_fold_results,

        # Metadata for reproducibility
        'metadata': {
            'random_seed': config.RANDOM_SEED,
            'system': get_system_metadata(),
            'hyperparameters': {
                'learning_rate': learning_rate,
                'weight_decay': config.WEIGHT_DECAY,
                'batch_size': batch_size,
                'optimizer': 'Adam',
                'scheduler': 'CosineAnnealingLR',
                'early_stopping_patience': config.EARLY_STOPPING_PATIENCE
            }
        }
    }

    # Save results
    results_dir = config.get_results_path(experiment_id)
    results_path = results_dir / "cv_results.json"
    with open(results_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"K-Fold Cross-Validation Results Summary")
    print(f"{'='*80}")
    print(f"Validation Accuracy: {aggregated_results['val_acc_mean']:.2f} ± {aggregated_results['val_acc_std']:.2f}%")
    print(f"95% CI: [{aggregated_results['val_acc_ci_lower']:.2f}, {aggregated_results['val_acc_ci_upper']:.2f}]")
    print(f"Range: [{aggregated_results['val_acc_min']:.2f}, {aggregated_results['val_acc_max']:.2f}]")
    print(f"\nResults saved to: {results_path}")
    print(f"{'='*80}")

    return aggregated_results


def main():
    parser = argparse.ArgumentParser(description="Train model with K-fold cross-validation")
    parser.add_argument("--model", type=str, default="resnet50", choices=config.SUPPORTED_MODELS)
    parser.add_argument("--task", type=str, default="syntax", choices=["syntax", "stenosis"])
    parser.add_argument("--train-quality", type=int, default=None)
    parser.add_argument("--val-quality", type=int, default=None)
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "jpeg2000", "avif"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (mixed precision)")
    args = parser.parse_args()

    train_model_cv(
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
        k_folds=args.k_folds,
        use_amp=not args.no_amp
    )


if __name__ == "__main__":
    main()
