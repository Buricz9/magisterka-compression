"""
Feature analysis module for compression experiments.

Provides:
- Feature map extraction from model layers
- Spectral entropy (effective rank) calculation
- Shannon entropy computation
- Per-layer and global metric averaging
- Information content comparison between models

Author: Master's Thesis Project
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
# Note: timm is imported in train.py for model creation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from dataset import get_dataloader


class FeatureExtractor:
    """
    Extract feature maps from intermediate layers of a neural network.

    Supports extracting features from all convolutional and linear layers,
    with hooks to capture activations during forward pass.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the feature extractor.

        Args:
            model: PyTorch model to extract features from
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.device = device
        self.features = {}
        self.handles = []

    def _get_hook(self, name: str) -> Callable:
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def register_hooks(self, layer_types: Tuple = (nn.Conv2d, nn.Linear)) -> List[str]:
        """
        Register forward hooks on specified layer types.

        Args:
            layer_types: Tuple of layer types to hook

        Returns:
            List of registered layer names
        """
        registered_names = []

        for name, module in self.model.named_modules():
            if isinstance(module, layer_types) and len(name) > 0:
                handle = module.register_forward_hook(self._get_hook(name))
                self.handles.append(handle)
                registered_names.append(name)

        return registered_names

    def register_hooks_by_names(self, layer_names: List[str]) -> None:
        """
        Register forward hooks on specific layers by name.

        Args:
            layer_names: List of layer names to hook
        """
        for name, module in self.model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._get_hook(name))
                self.handles.append(handle)

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input tensor.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        self.features = {}
        with torch.no_grad():
            _ = self.model(x.to(self.device))
        return self.features

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []


class SpectralEntropyCalculator:
    """
    Calculate spectral entropy (effective rank) of feature maps.

    Spectral entropy measures the information content and complexity
    of feature representations based on the singular value distribution.

    Effective rank (erank) is defined as:
    erank = exp(H) where H = -sum(s_i * log(s_i)) and s_i are normalized singular values
    """

    @staticmethod
    def normalize_singular_values(s: np.ndarray) -> np.ndarray:
        """
        Normalize singular values to sum to 1 (probability distribution).

        Args:
            s: Array of singular values

        Returns:
            Normalized singular values
        """
        s = np.abs(s)
        s_sum = np.sum(s)

        # Handle edge cases: empty array or zero sum
        if len(s) == 0:
            return np.array([])
        if s_sum < 1e-10:  # Use epsilon comparison for floating point safety
            return np.ones_like(s) / len(s)
        return s / s_sum

    @staticmethod
    def shannon_entropy(p: np.ndarray, base: float = np.e) -> float:
        """
        Calculate Shannon entropy of a probability distribution.

        H = -sum(p_i * log(p_i))

        Args:
            p: Probability distribution (should sum to 1)
            base: Logarithm base (e for nats, 2 for bits)

        Returns:
            Shannon entropy value
        """
        p = p[p > 0]  # Filter out zeros to avoid log(0)
        if len(p) == 0:
            return 0.0

        if base == np.e:
            return -np.sum(p * np.log(p))
        elif base == 2:
            return -np.sum(p * np.log2(p))
        else:
            return -np.sum(p * np.log(p) / np.log(base))

    @staticmethod
    def spectral_entropy(feature_map: torch.Tensor, normalize: bool = True) -> float:
        """
        Calculate spectral entropy of a feature map.

        For batch inputs, computes the mean spectral entropy across all samples.

        Args:
            feature_map: Feature tensor of shape (C, H, W) or (B, C, H, W)
            normalize: Whether to normalize singular values

        Returns:
            Spectral entropy value (mean across batch if batch dimension present)
        """
        if feature_map.dim() == 4:
            # Batch dimension present - compute mean across all samples
            entropies = []
            for i in range(feature_map.shape[0]):
                single_map = feature_map[i]
                # Flatten spatial dimensions: (C, H, W) -> (C, H*W)
                C = single_map.shape[0]
                feature_flat = single_map.reshape(C, -1)

                try:
                    feature_np = feature_flat.cpu().numpy()
                    s = np.linalg.svd(feature_np, compute_uv=False)
                except np.linalg.LinAlgError:
                    entropies.append(0.0)
                    continue

                if normalize:
                    s = SpectralEntropyCalculator.normalize_singular_values(s)

                entropies.append(SpectralEntropyCalculator.shannon_entropy(s))

            return float(np.mean(entropies)) if entropies else 0.0

        # Single sample: (C, H, W)
        C = feature_map.shape[0]
        feature_flat = feature_map.reshape(C, -1)

        try:
            feature_np = feature_flat.cpu().numpy()
            s = np.linalg.svd(feature_np, compute_uv=False)
        except np.linalg.LinAlgError:
            return 0.0

        if normalize:
            s = SpectralEntropyCalculator.normalize_singular_values(s)

        return SpectralEntropyCalculator.shannon_entropy(s)

    @staticmethod
    def effective_rank(feature_map: torch.Tensor) -> float:
        """
        Calculate effective rank (erank) of a feature map.

        Effective rank = exp(spectral_entropy)

        For batch inputs, uses the mean spectral entropy.

        Args:
            feature_map: Feature tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Effective rank value
        """
        entropy = SpectralEntropyCalculator.spectral_entropy(feature_map)
        return np.exp(entropy)

    @staticmethod
    def stable_rank(feature_map: torch.Tensor) -> float:
        """
        Calculate stable rank of a feature map.

        Stable rank = ||A||_F^2 / ||A||_2^2

        For batch inputs, computes the mean stable rank across all samples.

        Args:
            feature_map: Feature tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Stable rank value (mean across batch if batch dimension present)
        """
        if feature_map.dim() == 4:
            # Batch dimension present - compute mean across all samples
            ranks = []
            for i in range(feature_map.shape[0]):
                single_map = feature_map[i]
                C = single_map.shape[0]
                feature_flat = single_map.reshape(C, -1)
                feature_np = feature_flat.cpu().numpy()

                frobenius_norm = np.linalg.norm(feature_np, 'fro')
                spectral_norm = np.linalg.norm(feature_np, 2)

                if spectral_norm == 0:
                    ranks.append(0.0)
                else:
                    ranks.append((frobenius_norm ** 2) / (spectral_norm ** 2))

            return float(np.mean(ranks)) if ranks else 0.0

        # Single sample
        C = feature_map.shape[0]
        feature_flat = feature_map.reshape(C, -1)
        feature_np = feature_flat.cpu().numpy()

        frobenius_norm = np.linalg.norm(feature_np, 'fro')
        spectral_norm = np.linalg.norm(feature_np, 2)

        if spectral_norm == 0:
            return 0.0

        return (frobenius_norm ** 2) / (spectral_norm ** 2)


class FeatureMapAnalyzer:
    """
    Comprehensive analyzer for feature maps across model layers.

    Computes various information-theoretic metrics and provides
    comparison between models trained on different compression formats.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the feature map analyzer.

        Args:
            model: PyTorch model to analyze
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.extractor = FeatureExtractor(model, device)
        self.layer_names = []

    def setup_hooks(self, layer_types: Tuple = (nn.Conv2d, nn.Linear)) -> None:
        """
        Set up hooks for feature extraction.

        Args:
            layer_types: Types of layers to hook
        """
        self.layer_names = self.extractor.register_hooks(layer_types)
        print(f"Registered hooks on {len(self.layer_names)} layers")

    def analyze_single_batch(
        self,
        images: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature maps for a single batch of images.

        Args:
            images: Input image batch

        Returns:
            Dictionary with metrics per layer
        """
        features = self.extractor.extract_features(images)

        results = {}
        for layer_name, feature_map in features.items():
            results[layer_name] = {
                'shape': list(feature_map.shape),
                'spectral_entropy': SpectralEntropyCalculator.spectral_entropy(feature_map),
                'effective_rank': SpectralEntropyCalculator.effective_rank(feature_map),
                'stable_rank': SpectralEntropyCalculator.stable_rank(feature_map),
                # Pixel entropy: Shannon entropy on normalized pixel values (different from spectral entropy)
                'pixel_entropy': SpectralEntropyCalculator.shannon_entropy(
                    SpectralEntropyCalculator.normalize_singular_values(
                        feature_map.flatten().cpu().numpy()
                    )
                )
            }

        return results

    def analyze_dataloader(
        self,
        dataloader,
        max_batches: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature maps across entire dataloader.

        Args:
            dataloader: PyTorch DataLoader
            max_batches: Maximum number of batches to process (None for all)

        Returns:
            Dictionary with averaged metrics per layer
        """
        layer_metrics = defaultdict(lambda: defaultdict(list))

        self.model.eval()
        n_batches = 0

        for batch_idx, (images, labels, _) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            batch_results = self.analyze_single_batch(images)

            for layer_name, metrics in batch_results.items():
                for metric_name, value in metrics.items():
                    if metric_name != 'shape':
                        layer_metrics[layer_name][metric_name].append(value)

            n_batches += 1
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}")

        # Average metrics
        averaged_results = {}
        for layer_name, metrics in layer_metrics.items():
            averaged_results[layer_name] = {
                metric: float(np.mean(values))
                for metric, values in metrics.items()
            }
            averaged_results[layer_name]['std'] = {
                metric: float(np.std(values))
                for metric, values in metrics.items()
            }

        averaged_results['_metadata'] = {
            'n_batches': n_batches,
            'n_layers': len(layer_metrics)
        }

        return averaged_results

    def compute_global_metrics(
        self,
        layer_results: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Compute global (averaged across layers) metrics.

        Args:
            layer_results: Per-layer analysis results

        Returns:
            Dictionary of global metrics
        """
        metrics_to_average = ['spectral_entropy', 'effective_rank', 'stable_rank', 'pixel_entropy']

        global_metrics = {}
        for metric in metrics_to_average:
            values = [
                layer_data.get(metric, 0)
                for layer_name, layer_data in layer_results.items()
                if layer_name != '_metadata' and metric in layer_data
            ]
            if values:
                global_metrics[f'global_mean_{metric}'] = float(np.mean(values))
                global_metrics[f'global_std_{metric}'] = float(np.std(values))

        return global_metrics

    def cleanup(self) -> None:
        """Remove hooks and clean up resources."""
        self.extractor.remove_hooks()


def analyze_layer_progression(
    checkpoint_path: Path,
    model_name: str,
    task: str,
    device: str = 'cuda',
    max_batches: int = 10,
    dataset: str = 'arcade'
) -> Dict:
    """
    Analyze how information content changes across network depth.

    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Model architecture name
        task: Dataset task
        device: Computation device
        max_batches: Maximum batches to analyze
        dataset: Dataset type ('arcade' or 'isic')

    Returns:
        Layer progression analysis results
    """
    from train import create_model

    # Load model
    # Use optimal num_workers for performance (min of 4 or CPU count)
    import os
    optimal_workers = min(4, os.cpu_count() or 1)

    # Select appropriate dataloader based on dataset
    if dataset == 'isic':
        from isic_dataset import get_isic_dataloader
        print("Using ISIC 2019 dataset for feature analysis")
        dataloader = get_isic_dataloader('test', quality=None, format=None,
                                        batch_size=16, num_workers=optimal_workers, shuffle=False)
    else:
        dataloader = get_dataloader(task, 'test', quality=None, format=None,
                                    batch_size=16, num_workers=optimal_workers, shuffle=False)

    num_classes = dataloader.dataset.num_classes

    model = create_model(model_name, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Analyze
    analyzer = FeatureMapAnalyzer(model, device)
    analyzer.setup_hooks()

    layer_results = analyzer.analyze_dataloader(dataloader, max_batches=max_batches)

    # Organize by layer depth
    layer_order = analyzer.layer_names

    progression = {
        'layer_order': layer_order,
        'metrics_by_depth': {
            'spectral_entropy': [],
            'effective_rank': [],
            'stable_rank': []
        }
    }

    for layer_name in layer_order:
        if layer_name in layer_results:
            for metric in progression['metrics_by_depth']:
                if metric in layer_results[layer_name]:
                    progression['metrics_by_depth'][metric].append(
                        layer_results[layer_name][metric]
                    )

    analyzer.cleanup()

    return {
        'checkpoint': str(checkpoint_path),
        'model': model_name,
        'task': task,
        'dataset': dataset,
        'layer_results': layer_results,
        'progression': progression,
        'analysis_date': datetime.now().isoformat()
    }


def run_feature_analysis(
    model_name: str = "resnet50",
    task: str = "syntax",
    experiment_id: str = None,
    device: str = 'cuda',
    max_batches: int = 10,
    output_dir: Optional[Path] = None,
    dataset: str = 'arcade'
) -> Dict:
    """
    Run feature analysis on a trained model.

    Args:
        model_name: Model architecture name
        task: Dataset task
        experiment_id: Experiment ID to find checkpoint
        device: Computation device
        max_batches: Maximum batches to analyze
        output_dir: Output directory for results
        dataset: Dataset type ('arcade' or 'isic')

    Returns:
        Analysis results dictionary
    """
    if output_dir is None:
        output_dir = config.RESULTS_ROOT / "feature_analysis"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoint
    if experiment_id:
        checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"
    else:
        raise ValueError("experiment_id is required")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Analyzing checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset}")

    # Run analysis
    results = analyze_layer_progression(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        task=task,
        device=device,
        max_batches=max_batches,
        dataset=dataset
    )

    # Save results
    json_path = output_dir / f"feature_analysis_{experiment_id}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {json_path}")

    # Generate report
    report_path = output_dir / f"feature_report_{experiment_id}.txt"
    # Create a compatible structure for the report
    report_data = {
        'comparison_date': results['analysis_date'],
        'model': results['model'],
        'task': results['task'],
        'formats': {
            'analyzed': {
                'global': {
                    k: v for k, v in results['progression']['metrics_by_depth'].items()
                    for v in [sum(v) / len(v) if v else 0]
                    for k in [f'global_mean_{metric}' for metric in results['progression']['metrics_by_depth'].keys()]
                } | {'global_mean_spectral_entropy': sum(results['progression']['metrics_by_depth'].get('spectral_entropy', [0])) / max(len(results['progression']['metrics_by_depth'].get('spectral_entropy', [1])), 1),
                     'global_mean_effective_rank': sum(results['progression']['metrics_by_depth'].get('effective_rank', [0])) / max(len(results['progression']['metrics_by_depth'].get('effective_rank', [1])), 1),
                     'global_mean_stable_rank': sum(results['progression']['metrics_by_depth'].get('stable_rank', [0])) / max(len(results['progression']['metrics_by_depth'].get('stable_rank', [1])), 1)},
                'per_layer': results['layer_results']
            }
        },
        'global_comparison': {
            metric: {'values': {'analyzed': sum(vals) / len(vals) if vals else 0}, 'best_format': 'analyzed'}
            for metric, vals in results['progression']['metrics_by_depth'].items()
        }
    }

    # Save report as JSON instead
    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"Report saved to: {report_path}")

    return results


def main():
    """Command-line interface for feature analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature map analysis for compression experiments")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model architecture"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="syntax",
        choices=["syntax", "stenosis"],
        help="Dataset task (for ARCADE)"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment ID to analyze"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="arcade",
        choices=["arcade", "isic"],
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Maximum batches to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_feature_analysis(
        model_name=args.model,
        task=args.task,
        experiment_id=args.experiment_id,
        device=args.device,
        max_batches=args.max_batches,
        output_dir=output_dir,
        dataset=args.dataset
    )


if __name__ == "__main__":
    main()
