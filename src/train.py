"""Training pipeline for ARCADE dataset."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from dataset import get_dataloader


def create_model(model_name, num_classes):
    """Create model using timm library."""
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels, _ in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
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


def train_model(model_name, task, train_quality, val_quality, num_epochs,
                batch_size, learning_rate, device, train_format=None, val_format=None,
                experiment_id=None, early_stopping_patience=10):
    """Train a model on ARCADE dataset."""
    if experiment_id is None:
        quality_str = f"q{train_quality}" if train_quality else "baseline"
        format_str = f"_{train_format}" if train_format and train_format != 'jpeg' else ""
        experiment_id = f"{model_name}_{task}_{quality_str}{format_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nTraining: {experiment_id}")
    print(f"Model: {model_name}, Task: {task}, Epochs: {num_epochs}")
    if train_format:
        print(f"Format: {train_format}")

    # Create dataloaders
    train_loader = get_dataloader(task, 'train', train_quality, train_format, batch_size, num_workers=0, shuffle=True)
    val_loader = get_dataloader(task, 'val', val_quality, val_format, batch_size, num_workers=0, shuffle=False)

    # Create model
    num_classes = train_loader.dataset.num_classes
    model = create_model(model_name, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_val_acc, best_epoch, patience_counter = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience_counter = val_acc, epoch, 0
            checkpoint_dir = config.get_checkpoint_path(experiment_id)
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
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save results
    results = {
        'experiment_id': experiment_id,
        'model_name': model_name,
        'task': task,
        'train_quality': train_quality,
        'val_quality': val_quality,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'history': history
    }

    results_dir = config.get_results_path(experiment_id)
    with open(results_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--task", type=str, default="syntax", choices=["syntax", "stenosis"])
    parser.add_argument("--train-quality", type=int, default=None)
    parser.add_argument("--val-quality", type=int, default=None)
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "jpeg2000", "avif"],
                       help="Compression format")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        task=args.task,
        train_quality=args.train_quality,
        val_quality=args.val_quality,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        train_format=args.format,
        val_format=args.format
    )


if __name__ == "__main__":
    main()
