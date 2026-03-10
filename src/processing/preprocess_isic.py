"""
Preprocessing script for ISIC 2019 dataset.

Downloads, organizes, and creates train/val/test splits.
"""
import sys
from pathlib import Path
import argparse
import json
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def preprocess_isic_2019(
    input_root: Path,
    output_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_size: tuple = (224, 224),
    seed: int = 42
):
    """
    Preprocess ISIC 2019 dataset.

    Args:
        input_root: Directory with raw ISIC data (ISIC_2019_Training_Input/)
        output_root: Output directory for processed data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        target_size: Target image size
        seed: Random seed
    """
    np.random.seed(seed)

    input_root = Path(input_root)
    output_root = Path(output_root)

    # Find images directory
    images_dir = input_root / 'ISIC_2019_Training_Input'
    if not images_dir.exists():
        images_dir = input_root

    # Find ground truth file
    gt_path = input_root / 'ISIC_2019_Training_GroundTruth.csv'
    if not gt_path.exists():
        # Try to find it
        gt_files = list(input_root.glob('*GroundTruth*.csv'))
        if gt_files:
            gt_path = gt_files[0]
        else:
            raise FileNotFoundError(
                f"Ground truth file not found in {input_root}\n"
                "Download ISIC_2019_Training_GroundTruth.csv from "
                "https://challenge.isic-archive.com/landing/2019/"
            )

    print(f"Reading ground truth from: {gt_path}")
    df = pd.read_csv(gt_path)

    # Parse labels (one-hot encoded in CSV)
    class_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    image_labels = {}

    for _, row in df.iterrows():
        image_id = row['image']
        # Find which class is 1
        for i, col in enumerate(class_cols):
            if row[col] == 1:
                image_labels[image_id] = i
                break

    print(f"Found {len(image_labels)} labeled images")

    # Class distribution
    class_counts = defaultdict(int)
    for label in image_labels.values():
        class_counts[label] += 1

    print("\nClass distribution:")
    for i, count in sorted(class_counts.items()):
        print(f"  {class_cols[i]}: {count}")

    # Split into train/val/test with stratification
    image_ids = list(image_labels.keys())
    labels = [image_labels[img_id] for img_id in image_ids]

    # Group by class for stratified split
    class_images = defaultdict(list)
    for img_id, label in zip(image_ids, labels):
        class_images[label].append(img_id)

    train_ids, val_ids, test_ids = [], [], []

    for label, ids in class_images.items():
        np.random.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    np.random.shuffle(train_ids)
    np.random.shuffle(val_ids)
    np.random.shuffle(test_ids)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_ids)}")
    print(f"  Val: {len(val_ids)}")
    print(f"  Test: {len(test_ids)}")

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_root / split).mkdir(parents=True, exist_ok=True)

    # Copy and resize images
    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    for split_name, split_ids in splits.items():
        print(f"\nProcessing {split_name} set...")
        split_dir = output_root / split_name

        for img_id in tqdm(split_ids, desc=split_name):
            src_path = images_dir / f"{img_id}.jpg"
            if not src_path.exists():
                src_path = images_dir / f"{img_id}.png"
            if not src_path.exists():
                print(f"Warning: Image not found: {img_id}")
                continue

            dst_path = split_dir / f"{img_id}.jpg"

            # Resize and save
            try:
                img = Image.open(src_path).convert('RGB')
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                img.save(dst_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"Error processing {img_id}: {e}")

    # Save labels
    all_labels = {img_id: image_labels[img_id] for img_id in image_ids}
    with open(output_root / 'labels.json', 'w') as f:
        json.dump(all_labels, f, indent=2)

    # Save split lists
    for split_name, split_ids in splits.items():
        with open(output_root / f'{split_name}_images.txt', 'w') as f:
            for img_id in split_ids:
                f.write(f"{img_id}\n")

    # Save class names
    class_info = {
        'class_names': class_cols,
        'class_full_names': {
            'MEL': 'Melanoma',
            'NV': 'Melanocytic nevus',
            'BCC': 'Basal cell carcinoma',
            'AK': 'Actinic keratosis',
            'BKL': 'Benign keratosis',
            'DF': 'Dermatofibroma',
            'VASC': 'Vascular lesion',
            'SCC': 'Squamous cell carcinoma'
        },
        'num_classes': 8
    }
    with open(output_root / 'class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)

    print(f"\n[OK] Preprocessing complete!")
    print(f"Output saved to: {output_root}")
    print(f"\nDirectory structure:")
    print(f"  {output_root}/")
    print(f"  ├── train/       ({len(train_ids)} images)")
    print(f"  ├── val/         ({len(val_ids)} images)")
    print(f"  ├── test/        ({len(test_ids)} images)")
    print(f"  ├── labels.json")
    print(f"  ├── class_info.json")
    print(f"  ├── train_images.txt")
    print(f"  ├── val_images.txt")
    print(f"  └── test_images.txt")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ISIC 2019 dataset")
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Directory with raw ISIC data (ISIC_2019_Training_Input/ and GroundTruth.csv)"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="dataset/isic_2019",
        help="Output directory (default: dataset/isic_2019)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (default: 224 224)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    preprocess_isic_2019(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_size=tuple(args.target_size),
        seed=args.seed
    )


if __name__ == "__main__":
    main()
