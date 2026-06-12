"""Experiment C: mixed-quality training (compression as augmentation).

Motivation: Experiment B showed that a model trained on clean PNG degrades when
tested on COMPRESSED images (especially JPEG at low Q). Experiment C asks whether
exposing the model to a MIXTURE of compression levels during training makes it
robust to that test-time compression.

Setup:
  - Train: each training image is loaded at a RANDOM quality level drawn from
    config.QUALITY_LEVELS (a given format). Compression thus acts as a data
    augmentation; the model sees the full range of artifacts.
  - Validation: deterministic mixed quality (a fixed Q per image, derived from
    image_id) so the early-stopping signal is stable yet representative.
  - Test: evaluated BOTH on clean PNG (the supervisor's request) AND on each
    compressed Q level (mirroring Experiment B), so we can check whether mixed
    training closes the gap that JPEG opened in Experiment B.

Reuses the validated train/eval primitives from src.core.train; only the dataset
(mixed quality) and the run loop are new.
"""
import sys
import gc
import hashlib
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.core.dataset import ArcadeClassificationDataset, get_transforms, _worker_init
from src.core.train import (create_model, train_epoch, validate, evaluate_model)

_EXT = {'jpeg': '.jpg', 'jpeg2000': '.jp2', 'avif': '.avif'}


class MixedQualityDataset(ArcadeClassificationDataset):
    """ARCADE dataset that draws a random compression quality per access.

    Subclasses the validated ArcadeClassificationDataset; only image LOADING is
    overridden to pick a quality level from `quality_pool` for each __getitem__.

    mode='random'      -> a fresh random Q every access (training augmentation).
    mode='deterministic' -> a fixed Q per image (image_id-hashed), for stable val.
    """

    def __init__(self, task, split, fmt, quality_pool, mode='random',
                 target_size=(224, 224)):
        # Initialise the base dataset in PNG mode; we override the image path
        # per-item, so the base `quality`/`images_dir` are not used for loading.
        super().__init__(task, split, quality=None, format=fmt, target_size=target_size)
        self.fmt = fmt
        self._split = split          # used by __getitem__ to build the compressed path
        self.quality_pool = list(quality_pool)
        self.mode = mode
        self.ext = _EXT.get(fmt, '.jpg')
        # Use the split-appropriate transforms (train augments, val/test do not).
        self.transform = get_transforms(split, target_size)

    def _pick_quality(self, image_id, idx):
        if self.mode == 'deterministic':
            # Stable Q per image. Uses md5 (NOT Python's built-in hash(), which is
            # randomised per process via PYTHONHASHSEED) so the assignment is
            # identical across workers, runs and machines -> reproducible val.
            h = int(hashlib.md5(f"{self.fmt}_{int(image_id)}".encode()).hexdigest(), 16)
            return self.quality_pool[h % len(self.quality_pool)]
        # Random per access (training): numpy RNG is seeded per worker.
        return self.quality_pool[np.random.randint(len(self.quality_pool))]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images_info[image_id]
        label = torch.from_numpy(self.image_labels[image_id])

        q = self._pick_quality(image_id, idx)
        # Build the compressed-image path directly (config.get_data_path layout).
        images_dir = config.COMPRESSED_ROOT / self.fmt / f"q{q}" / 'syntax' / \
            self._split / 'images'
        fname = Path(image_info['file_name']).stem + self.ext
        path = images_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, str(image_id)


def _make_mixed_dataset(task, split, fmt, quality_pool, mode):
    return MixedQualityDataset(task, split, fmt, quality_pool, mode=mode)


def _mixed_loader(task, split, fmt, quality_pool, mode, batch_size, shuffle):
    ds = _make_mixed_dataset(task, split, fmt, quality_pool, mode)
    gen = torch.Generator(); gen.manual_seed(config.RANDOM_SEED)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None,
        worker_init_fn=_worker_init if config.NUM_WORKERS > 0 else None,
        generator=gen if shuffle else None,
    )


def _train_mixed_model(model_name, task, fmt, quality_pool, num_epochs,
                       batch_size, device, use_amp=True):
    """Train one model on mixed-quality data; return (model, experiment_id, val)."""
    config.set_seed()
    experiment_id = f"{model_name}_{task}_mixed_{fmt}"

    train_loader = _mixed_loader(task, 'train', fmt, quality_pool, 'random',
                                 batch_size, shuffle=True)
    val_loader = _mixed_loader(task, 'val', fmt, quality_pool, 'deterministic',
                               batch_size, shuffle=False)

    num_classes = train_loader.dataset.num_classes
    model = create_model(model_name, num_classes).to(device)
    pos_weight = train_loader.dataset.compute_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')

    best_val, best_epoch, patience = float('-inf'), 0, 0
    ckpt_dir = config.get_checkpoint_path(experiment_id)
    ckpt_path = ckpt_dir / "best_model.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}  [{model_name} mixed-{fmt}]")
        tr_loss, tr_score = train_epoch(model, train_loader, criterion, optimizer,
                                        device, scaler, use_amp)
        val_loss, val_score = validate(model, val_loader, criterion, device, use_amp=False)
        scheduler.step()
        print(f"Train F1={tr_score:.2f}%  Val F1={val_score:.2f}%")

        if val_score > best_val or best_val <= 0:
            best_val, best_epoch, patience = val_score, epoch, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_score': val_score}, ckpt_path)
            print(f"  saved best ({val_score:.2f}%)")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # restore best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True)['model_state_dict'])
    return model, experiment_id, best_val


def run_experiment_c(model_name, task, formats, num_epochs, batch_size, device):
    quality_pool = config.QUALITY_LEVELS
    out_dir = config.RESULTS_ROOT / "experiment_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        print(f"\n{'='*80}\nMixed-quality training: {model_name} / {fmt}\n{'='*80}")
        model, experiment_id, best_val = _train_mixed_model(
            model_name, task, fmt, quality_pool, num_epochs, batch_size, device)

        rows = []
        # Test on clean PNG (supervisor's primary request)
        from src.core.dataset import get_dataloader
        png_loader = get_dataloader(task, 'test', quality=None, format=None,
                                    batch_size=batch_size, num_workers=config.NUM_WORKERS)
        m = evaluate_model(model, png_loader, device)
        rows.append({'experiment_id': experiment_id, 'format': fmt,
                     'train_quality': 'mixed', 'test_quality': 'png',
                     'best_val_score': best_val, **{f'test_{k}': v for k, v in m.items()}})
        print(f"  test PNG: mAP={m['map']:.4f}")
        del png_loader

        # Test on each compressed Q level (compare with Experiment B)
        for q in quality_pool:
            tl = get_dataloader(task, 'test', quality=q, format=fmt,
                                batch_size=batch_size, num_workers=config.NUM_WORKERS)
            m = evaluate_model(model, tl, device)
            rows.append({'experiment_id': experiment_id, 'format': fmt,
                         'train_quality': 'mixed', 'test_quality': q,
                         'best_val_score': best_val, **{f'test_{k}': v for k, v in m.items()}})
            print(f"  test {fmt} Q={q}: mAP={m['map']:.4f}")
            del tl
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        out_file = out_dir / f"{model_name}_arcade_{task}_mixed_{fmt}_results.csv"
        pd.DataFrame(rows).to_csv(out_file, index=False)
        print(f"\nSaved: {out_file}")

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description="Experiment C: mixed-quality training")
    p.add_argument("--model", default="resnet50", choices=config.SUPPORTED_MODELS)
    p.add_argument("--task", default="syntax", choices=["syntax"])
    p.add_argument("--format", default="jpeg", choices=["jpeg", "jpeg2000", "avif", "all"])
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    formats = config.COMPRESSION_FORMATS if args.format == "all" else [args.format]
    run_experiment_c(args.model, args.task, formats, args.epochs, args.batch_size,
                     torch.device(args.device))


if __name__ == "__main__":
    main()
