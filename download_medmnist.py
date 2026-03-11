"""Download and explore MedMNIST datasets."""
import medmnist
from medmnist import INFO
from pathlib import Path
import numpy as np

# Create dataset directory
dataset_dir = Path("dataset/medmnist")
dataset_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MedMNIST v2 - Available Datasets")
print("="*80)

# Get all available datasets
all_datasets = list(INFO.keys())

# Filter 2D datasets only (exclude 3D datasets for compression study)
datasets_2d = []
for key in all_datasets:
    info = INFO[key]
    if info['task'] in ['multi-class', 'binary', 'multi-label']:  # Classification tasks
        datasets_2d.append((key, info))

print(f"\nFound {len(datasets_2d)} 2D classification datasets:\n")

# Print dataset info in a table format
print(f"{'Dataset':<20} {'Classes':<10} {'Task':<20} {'Images':<15} {'Size':<10}")
print("-"*80)

for key, info in datasets_2d:
    n_samples = sum(info['n_samples'].values())
    n_classes = len(info['label'])
    task = info['task']
    shape = info['n_channels']
    size = f"{shape}x28x28"  # MedMNIST images are 28x28 by default

    print(f"{key:<20} {n_classes:<10} {task:<20} {n_samples:<15} {size:<10}")

print("\n" + "="*80)
print("Recommended for compression study:")
print("="*80)

# Recommended datasets for compression study (similar to ARCADE)
recommended = ['dermamnist', 'pathmnist', 'chestmnist', 'retinamnist']

for key in recommended:
    info = INFO[key]
    n_samples = sum(info['n_samples'].values())
    n_classes = len(info['label'])

    print(f"\n{key.upper()}:")
    print(f"  - Classes: {n_classes}")
    print(f"  - Total images: {n_samples}")
    print(f"  - Task: {info['task']}")
    print(f"  - Description: {info['description']}")

print("\n" + "="*80)
print("Downloading recommended datasets...")
print("="*80)

# Download recommended datasets using the dataset classes directly
data_flag = 'pathmnist'
download = True

for key in recommended:
    print(f"\nDownloading {key}...")
    try:
        # Map key to correct class name (handle special cases)
        class_map = {
            'dermamnist': 'DermaMNIST',
            'pathmnist': 'PathMNIST',
            'chestmnist': 'ChestMNIST',
            'retinamnist': 'RetinaMNIST'
        }
        class_name = class_map.get(key, key.capitalize() + 'MNIST')

        # Import the specific dataset class
        DatasetClass = getattr(medmnist, class_name)

        # Download the dataset (train split only to trigger download)
        dataset = DatasetClass(split='train', download=True, root=str(dataset_dir))
        print(f"[OK] {key} downloaded successfully")
        print(f"  Train samples: {len(dataset)}")
        print(f"  Image shape: {dataset[0][0].shape}")
        print(f"  Number of classes: {len(dataset.labels)}")

    except Exception as e:
        print(f"[ERROR] Error downloading {key}: {e}")

print("\n" + "="*80)
print("Download complete!")
print("="*80)
print(f"\nDataset location: {dataset_dir.absolute()}")
print("\nDataset structure:")
print(f"  {dataset_dir}/")
print(f"    ├── dermamnist.npz")
print(f"    ├── pathmnist.npz")
print(f"    ├── chestmnist.npz")
print(f"    └── retinamnist.npz")
print("\nNext steps:")
print("1. Review the downloaded datasets")
print("2. Choose which ones to use for compression experiments")
print("3. Implement dataloader (similar to isic_dataset.py)")
print(f"\nDataset location: {dataset_dir.absolute()}")
print("\nDataset structure:")
print(f"  {dataset_dir}/")
print(f"    ├── {key}/")
print(f"    │   ├── pathmnist.npz (or similar)")
print(f"\nNext steps:")
print("1. Review the downloaded datasets")
print("2. Choose which ones to use for compression experiments")
print("3. Implement dataloader (similar to isic_dataset.py)")
