"""
PyTorch Dataset class for ARCADE dataset.

Supports loading images at different compression quality levels.
"""
import sys
from pathlib import Path
import json
from collections import defaultdict

import numpy as np
from PIL import Image
try:
    import pillow_avif  # noqa: F401  # registers AVIF plugin in Pillow
except ImportError:
    pass
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def get_transforms(split='train', target_size=(224, 224)):
    """
    Get appropriate transforms for train/val/test splits.

    Training includes data augmentation for better generalization,
    especially important for medical datasets with limited samples.

    Args:
        split: 'train', 'val', or 'test'
        target_size: Tuple (H, W) for resizing images

    Returns:
        Composed transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        # NOTE: no horizontal flip — angiography is anatomically asymmetric
        # (LCA vs RCA are on opposite sides of the chest, mirroring relabels
        # the segment). Vertical flip is also unsafe for the same reason.
        # Mild rotation (≤15°) and ColorJitter stay since they preserve laterality.
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # No augmentation for validation/test
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            normalize
        ])


class ArcadeClassificationDataset(Dataset):
    """
    ARCADE dataset for multi-label classification.

    Each angiogram contains multiple coronary artery segments from different
    classes (1..26). The label is a multi-hot vector marking every category
    that appears in the image's annotations.

    Args:
        task: "syntax"
        split: "train", "val", or "test"
        quality: None for baseline PNG, or int (10-100)
        format: "jpeg", "jpeg2000", or "avif" (only used if quality is not None)
        transform: Optional torchvision transforms
        target_size: Tuple (H, W) for resizing images
    """

    multi_label = True

    def __init__(self, task, split, quality=None, format=None, transform=None, target_size=(224, 224)):
        self.quality = quality
        self.format = format if format else config.DEFAULT_FORMAT

        annotations_path = config.get_annotations_path(task, split)
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)

        self.images_info = {img['id']: img for img in self.coco_data['images']}

        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        # Class count and category_id->idx mapping are pinned to the fixed
        # ARCADE/Syntax contract (category ids 1..26 -> indices 0..25) instead of
        # being derived from each split's `categories` block. Per-split derivation
        # (sorted(keys()) + enumerate) would silently desync the model head size
        # and the label columns between splits if any split's COCO file omitted a
        # 0-annotation category.
        self.num_classes = config.NUM_CLASSES
        self.category_to_idx = {cat_id: cat_id - 1 for cat_id in range(1, config.NUM_CLASSES + 1)}

        unexpected = sorted(c for c in self.categories if c not in self.category_to_idx)
        if unexpected:
            raise ValueError(
                f"{split}: COCO `categories` contains category_id(s) {unexpected} "
                f"outside the expected ARCADE range 1..{config.NUM_CLASSES}."
            )

        self.image_annotations = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.image_annotations[ann['image_id']].append(ann)

        # Multi-hot label per image: 1.0 for every category present in annotations.
        # Annotations with unknown category_id (outside the `categories` block)
        # are skipped with a warning instead of crashing — defensive guard for
        # potentially corrupt/extended COCO files.
        self.image_labels = {}
        unknown_cats = set()
        skipped_no_annotations = 0
        for image_id in self.images_info.keys():
            annotations = self.image_annotations[image_id]
            if not annotations:
                skipped_no_annotations += 1
                continue
            label = np.zeros(self.num_classes, dtype=np.float32)
            for ann in annotations:
                cat_id = ann['category_id']
                idx = self.category_to_idx.get(cat_id)
                if idx is None:
                    unknown_cats.add(cat_id)
                    continue
                label[idx] = 1.0
            self.image_labels[image_id] = label
        if unknown_cats:
            import warnings
            warnings.warn(
                f"{split}: skipped annotations with unknown category_id(s) "
                f"{sorted(unknown_cats)} — not present in categories block."
            )
        if skipped_no_annotations:
            import warnings
            warnings.warn(
                f"{split}: {skipped_no_annotations} image(s) with no annotations "
                f"dropped from the dataset (not treated as negative samples)."
            )

        self.image_ids = list(self.image_labels.keys())

        # Get image directory path
        self.images_dir = config.get_data_path(task, split, quality, format=self.format)

        # Set up transforms - use data augmentation for training
        if transform is None:
            self.transform = get_transforms(split, target_size)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def compute_pos_weight(self):
        """Per-class pos_weight = (#neg / #pos) for BCEWithLogitsLoss.

        Classes never positive get weight 1.0 (loss contribution stays zero anyway).
        """
        labels = np.stack([self.image_labels[i] for i in self.image_ids])  # (N, C)
        pos = labels.sum(axis=0)
        neg = labels.shape[0] - pos
        weight = np.where(pos > 0, neg / np.clip(pos, 1, None), 1.0).astype(np.float32)
        return torch.from_numpy(weight)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images_info[image_id]

        label = torch.from_numpy(self.image_labels[image_id])

        # Load image
        if self.quality is None:
            # Baseline PNG
            image_filename = image_info['file_name']
        else:
            # Compressed - get appropriate extension
            extensions = {'jpeg': '.jpg', 'jpeg2000': '.jp2', 'avif': '.avif'}
            ext = extensions.get(self.format, '.jpg')
            image_filename = Path(image_info['file_name']).stem + ext

        image_path = self.images_dir / image_filename

        # Validate image exists before loading
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label, str(image_id)


def _worker_init(worker_id):
    """Seed a DataLoader worker's RNGs for reproducible augmentation.

    Must be a module-level function (NOT a closure inside get_dataloader) so it
    stays picklable: Windows uses the 'spawn' start method, which pickles
    worker_init_fn to send it to each worker process — a nested function cannot
    be pickled. torchvision v1 transforms (RandomRotation, ColorJitter) sample
    via the torch RNG, so torch.manual_seed is seeded here alongside numpy/random.
    """
    import random
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_dataloader(task, split, quality=None, format=None, batch_size=16, num_workers=4, shuffle=None):
    """
    Create a DataLoader for ARCADE dataset.

    Args:
        task: "syntax"
        split: "train", "val", or "test"
        quality: None for baseline, or int (10-100)
        format: "jpeg", "jpeg2000", or "avif" (only used if quality is not None)
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle (default: True for train, False otherwise)

    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = ArcadeClassificationDataset(task, split, quality, format)

    generator = torch.Generator()
    generator.manual_seed(config.RANDOM_SEED)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        generator=generator if shuffle else None,
    )

    return dataloader
