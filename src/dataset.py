"""
PyTorch Dataset classes for ARCADE dataset.

Supports loading images at different compression quality levels.
"""
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


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

        # Parse categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.num_classes = len(self.categories)

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
                self.image_labels[image_id] = dominant_category - 1  # 0-indexed
            else:
                # No annotations for this image, skip it
                continue

        # Filter to only images with labels
        self.image_ids = list(self.image_labels.keys())

        # Get image directory path
        self.images_dir = config.get_data_path(task, split, quality, format=self.format)

        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
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

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load {image_path}: {e}")
            # Return a black image and label
            image = Image.new('RGB', (512, 512), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label, image_id

    def get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        class_counts = defaultdict(int)
        for label in self.image_labels.values():
            class_counts[label] += 1
        return dict(sorted(class_counts.items()))


class ArcadeSegmentationDataset(Dataset):
    """
    ARCADE dataset for segmentation task.

    Returns images and their corresponding segmentation masks.

    Args:
        task: "syntax" or "stenosis"
        split: "train", "val", or "test"
        quality: None for baseline PNG, or int (10-100)
        format: "jpeg", "jpeg2000", or "avif" (only used if quality is not None)
        transform: Optional transforms (applied to both image and mask)
        target_size: Tuple (H, W) for resizing
    """

    def __init__(self, task, split, quality=None, format=None, transform=None, target_size=(512, 512)):
        self.task = task
        self.split = split
        self.quality = quality
        self.format = format if format else config.DEFAULT_FORMAT
        self.target_size = target_size

        # Load annotations
        self.annotations_path = config.get_annotations_path(task, split)
        with open(self.annotations_path, 'r') as f:
            self.coco_data = json.load(f)

        # Parse images
        self.images_info = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.images_info.keys())

        # Parse categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.num_classes = len(self.categories)

        # Group annotations by image_id
        self.image_annotations = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.image_annotations[ann['image_id']].append(ann)

        # Get image directory path
        self.images_dir = config.get_data_path(task, split, quality, format=self.format)

        # Set up image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_ids)

    def _create_mask(self, image_id, height, width):
        """Create segmentation mask from COCO annotations."""
        from pycocotools import mask as coco_mask

        annotations = self.image_annotations[image_id]

        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations:
            category_id = ann['category_id']

            # Convert segmentation to mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                from pycocotools import mask as maskUtils
                rles = maskUtils.frPyObjects(ann['segmentation'], height, width)
                rle = maskUtils.merge(rles)
                m = maskUtils.decode(rle)
                mask[m > 0] = category_id

        return mask

    def __getitem__(self, idx):
        # Get image info
        image_id = self.image_ids[idx]
        image_info = self.images_info[image_id]

        # Load image
        if self.quality is None:
            image_filename = image_info['file_name']
        else:
            # Compressed - get appropriate extension
            extensions = {'jpeg': '.jpg', 'jpeg2000': '.jp2', 'avif': '.avif'}
            ext = extensions.get(self.format, '.jpg')
            image_filename = Path(image_info['file_name']).stem + ext

        image_path = self.images_dir / image_filename

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load {image_path}: {e}")
            image = Image.new('RGB', (512, 512), (0, 0, 0))

        # Create mask
        mask = self._create_mask(image_id, image_info['height'], image_info['width'])
        mask = Image.fromarray(mask)

        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0).long()  # Remove channel dim and convert to long

        return image, mask, image_id


def get_dataloader(task, split, quality=None, format=None, batch_size=16, num_workers=4,
                   shuffle=None, dataset_type='classification'):
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
        dataset_type: "classification" or "segmentation"

    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')

    if dataset_type == 'classification':
        dataset = ArcadeClassificationDataset(task, split, quality, format)
    elif dataset_type == 'segmentation':
        dataset = ArcadeSegmentationDataset(task, split, quality, format)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == "__main__":
    # Test the dataset
    print("Testing ArcadeClassificationDataset...")

    # Test baseline
    dataset = ArcadeClassificationDataset(task='syntax', split='val', quality=None)
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    # Test loading one sample
    image, label, image_id = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label}")
    print(f"  Image ID: {image_id}")

    # Test with compressed quality
    dataset_q50 = ArcadeClassificationDataset(task='syntax', split='val', quality=50)
    print(f"\nCompressed dataset (Q=50) size: {len(dataset_q50)}")

    image_q50, label_q50, _ = dataset_q50[0]
    print(f"Sample 0 (Q=50):")
    print(f"  Image shape: {image_q50.shape}")
    print(f"  Label: {label_q50}")

    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = get_dataloader(task='syntax', split='val', quality=None, batch_size=8)
    batch_images, batch_labels, batch_ids = next(iter(dataloader))
    print(f"Batch shape: {batch_images.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    print(f"Batch IDs: {batch_ids}")

    print("\n[OK] Dataset tests passed!")
