"""Configuration for compression experiments on ARCADE dataset."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATA_ROOT = DATASET_ROOT / "arcade"
COMPRESSED_ROOT = DATASET_ROOT / "compressed"
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
LOGS_ROOT = PROJECT_ROOT / "logs"

# Dataset
TASKS = ["syntax", "stenosis"]
IMAGE_SIZE = (512, 512)
NUM_CLASSES = {"syntax": 26, "stenosis": 2}

# Compression formats
COMPRESSION_FORMATS = ["jpeg", "jpeg2000", "avif"]
DEFAULT_FORMAT = "jpeg"

# Compression levels
QUALITY_LEVELS = [100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]
QUALITY_LEVELS_MVP = [100, 85, 70, 50, 30, 10]

# Model settings
ARCHITECTURES = {"resnet50": "resnet50", "efficientnet_b0": "efficientnet_b0"}

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def get_data_path(task, split, quality=None, format=None):
    """
    Get path to data for specific task, split, quality level, and format.

    Args:
        task: "syntax" or "stenosis"
        split: "train", "val", or "test"
        quality: None (baseline PNG) or int (10-100)
        format: "jpeg", "jpeg2000", or "avif" (only used if quality is not None)

    Returns:
        Path to images directory
    """
    if quality is None:
        # Baseline PNG
        return DATA_ROOT / task / split / "images"
    else:
        # Compressed (use default format if not specified)
        if format is None:
            format = DEFAULT_FORMAT
        return COMPRESSED_ROOT / format / f"q{quality}" / task / split / "images"


def get_annotations_path(task, split):
    """Get path to annotations file."""
    return DATA_ROOT / task / split / "annotations" / f"{split}.json"


def get_checkpoint_path(experiment_id):
    """Get path for saving model checkpoint."""
    checkpoint_dir = MODELS_ROOT / "checkpoints" / experiment_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_results_path(experiment_id):
    """Get path for saving results."""
    results_dir = RESULTS_ROOT / "metrics" / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
