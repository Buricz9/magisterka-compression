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
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
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
        # Data augmentation for training
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
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
    ARCADE dataset for classification task.

    For simplicity, we'll treat each image as having a single dominant class
    (the category that appears most frequently in that image).

    Args:
        task: "syntax" or "stenosis"
        split: "train", "val", or "test"
        quality: None for baseline PNG, or int (10-100)
        format: "jpeg", "jpeg2000", or "avif" (only used if quality is not None)
        transform: Optional torchvision transforms
        target_size: Tuple (H, W) for resizing images
    """

    def __init__(self, task, split, quality=None, format=None, transform=None, target_size=(224, 224)):
        self.task = task
        self.split = split
        self.quality = quality
        self.format = format if format else config.DEFAULT_FORMAT
        self.target_size = target_size

        # Load annotations
        self.annotations_path = config.get_annotations_path(task, split)
        with open(self.annotations_path, 'r') as f:
            self.coco_data = json.load(f)

        # Parse images and create image_id to info mapping
        self.images_info = {img['id']: img for img in self.coco_data['images']}

        # Parse categories and create proper label mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.num_classes = len(self.categories)

        # Create category_id to label_idx mapping (handles non-contiguous category IDs)
        sorted_category_ids = sorted(self.categories.keys())
        self.category_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_category_ids)}

        # Group annotations by image_id
        self.image_annotations = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.image_annotations[ann['image_id']].append(ann)

        # Create mapping: image_id -> dominant category
        self.image_labels = {}
        for image_id in self.images_info.keys():
            annotations = self.image_annotations[image_id]
            if len(annotations) > 0:
                # Find most frequent category in this image
                category_counts = defaultdict(int)
                for ann in annotations:
                    category_counts[ann['category_id']] += 1
                dominant_category = max(category_counts.items(), key=lambda x: x[1])[0]
                # Use proper mapping instead of hardcoded offset
                self.image_labels[image_id] = self.category_to_idx[dominant_category]
            else:
                # No annotations for this image, skip it
                continue

        # Filter to only images with labels
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

    def __getitem__(self, idx):
        # Get image info
        image_id = self.image_ids[idx]
        image_info = self.images_info[image_id]

        # Get label
        label = self.image_labels[image_id]

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

        return image, label, image_id


def get_dataloader(task, split, quality=None, format=None, batch_size=16, num_workers=4, shuffle=None):
    """
    Create a DataLoader for ARCADE dataset.

    Args:
        task: "syntax" or "stenosis"
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

    # Optimized DataLoader settings for performance
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )

    return dataloader
