"""
ISIC 2019 Dataset loader for benchmark experiments.

ISIC 2019 - Skin lesion analysis dataset with 8 diagnostic categories.
https://challenge.isic-archive.com/landing/2019/

Classes:
- MEL: Melanoma
- NV: Melanocytic nevus
- BCC: Basal cell carcinoma
- AK: Actinic keratosis
- BKL: Benign keratosis
- DF: Dermatofibroma
- VASC: Vascular lesion
- SCC: Squamous cell carcinoma
"""
import sys
from pathlib import Path
from typing import Optional, Tuple
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


# ISIC 2019 class names
ISIC_CLASSES = [
    'MEL',   # Melanoma
    'NV',    # Melanocytic nevus
    'BCC',   # Basal cell carcinoma
    'AK',    # Actinic keratosis
    'BKL',   # Benign keratosis
    'DF',    # Dermatofibroma
    'VASC',  # Vascular lesion
    'SCC'    # Squamous cell carcinoma
]

ISIC_CLASS_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic nevus',
    'BCC': 'Basal cell carcinoma',
    'AK': 'Actinic keratosis',
    'BKL': 'Benign keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular lesion',
    'SCC': 'Squamous cell carcinoma'
}


class ISIC2019Dataset(Dataset):
    """
    ISIC 2019 dataset for skin lesion classification.

    Expected directory structure:
    dataset/
    └── isic_2019/
        ├── train/
        │   ├── ISIC_0000001.jpg
        │   ├── ISIC_0000002.jpg
        │   └── ...
        ├── val/
        │   └── ...
        ├── test/
        │   └── ...
        └── labels.json  # {"ISIC_0000001": 0, "ISIC_0000002": 3, ...}
    """

    def __init__(
        self,
        split: str = 'train',
        quality: Optional[int] = None,
        format: Optional[str] = None,
        transform=None,
        target_size: Tuple[int, int] = (224, 224),
        data_root: Optional[Path] = None
    ):
        """
        Initialize ISIC 2019 dataset.

        Args:
            split: 'train', 'val', or 'test'
            quality: Compression quality (None for original)
            format: Compression format ('jpeg', 'jpeg2000', 'avif')
            transform: Optional transforms
            target_size: Image size after resizing
            data_root: Root directory (default: config.DATASET_ROOT / 'isic_2019')
        """
        self.split = split
        self.quality = quality
        self.format = format if format else config.DEFAULT_FORMAT
        self.target_size = target_size

        # Set data root
        if data_root is None:
            self.data_root = config.DATASET_ROOT / 'isic_2019'
        else:
            self.data_root = Path(data_root)

        # Load labels
        labels_path = self.data_root / 'labels.json'
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                all_labels = json.load(f)
        else:
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}\n"
                "Please run ISIC 2019 preprocessing first."
            )

        # Load split indices
        split_path = self.data_root / f'{split}_images.txt'
        if split_path.exists():
            with open(split_path, 'r') as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: use all images with labels
            self.image_ids = list(all_labels.keys())

        # Filter labels to current split
        self.labels = {img_id: all_labels[img_id] for img_id in self.image_ids if img_id in all_labels}
        self.image_ids = list(self.labels.keys())

        # Get num_classes from config or use default (ISIC 2019 has 8 classes)
        self.num_classes = len(ISIC_CLASSES)

        # Get image directory
        if quality is None:
            self.images_dir = self.data_root / split
        else:
            # Compressed images
            self.images_dir = config.DATASET_ROOT / 'compressed_isic' / format / f'q{quality}' / split

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
        image_id = self.image_ids[idx]
        label = self.labels[image_id]

        # Get image path
        if self.quality is None:
            image_filename = f"{image_id}.jpg"
        else:
            extensions = {'jpeg': '.jpg', 'jpeg2000': '.jp2', 'avif': '.avif'}
            ext = extensions.get(self.format, '.jpg')
            image_filename = f"{image_id}{ext}"

        image_path = self.images_dir / image_filename

        # Validate image exists before loading
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label, image_id


def get_isic_dataloader(
    split: str = 'train',
    quality: Optional[int] = None,
    format: Optional[str] = None,
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    target_size: Tuple[int, int] = (224, 224)
) -> DataLoader:
    """
    Create DataLoader for ISIC 2019 dataset.

    Args:
        split: 'train', 'val', or 'test'
        quality: Compression quality (None for original)
        format: Compression format
        batch_size: Batch size
        num_workers: Number of workers (default: config.NUM_WORKERS)
        shuffle: Whether to shuffle (default: True for train)
        target_size: Image size

    Returns:
        DataLoader
    """
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = ISIC2019Dataset(
        split=split,
        quality=quality,
        format=format,
        target_size=target_size
    )

    # Optimized DataLoader settings for performance
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )

    return dataloader


def download_isic_instructions() -> str:
    """Return instructions for downloading ISIC 2019 dataset."""
    return """
ISIC 2019 Dataset Download Instructions
========================================

1. Visit: https://challenge.isic-archive.com/landing/2019/

2. Download files:
   - ISIC_2019_Training_Input.zip (Training images, ~10 GB)
   - ISIC_2019_Training_GroundTruth.csv (Labels)

3. Extract and organize:
   dataset/
   └── isic_2019/
       ├── train/
       ├── val/
       ├── test/
       └── labels.json

4. Run preprocessing to create splits and labels.json:
   python src/preprocess_isic.py --data-root dataset/isic_2019_raw --output-root dataset/isic_2019

5. Run compression:
   python src/compress_isic.py --format all --mvp

The dataset has 25,331 images across 8 diagnostic categories.
"""


def check_isic_available() -> bool:
    """Check if ISIC 2019 dataset is available."""
    isic_root = config.DATASET_ROOT / 'isic_2019'
    labels_path = isic_root / 'labels.json'
    return labels_path.exists()


if __name__ == "__main__":
    print(download_isic_instructions())

    if check_isic_available():
        print("\n[OK] ISIC 2019 dataset is available!")
        loader = get_isic_dataloader(split='train', batch_size=4)
        images, labels, ids = next(iter(loader))
        print(f"Sample batch: images shape = {images.shape}, labels = {labels}")
    else:
        print("\n[WARNING] ISIC 2019 dataset not found. Follow instructions above.")
