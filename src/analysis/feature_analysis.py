"""
Feature analysis module for compression experiments.

Provides:
- Feature map extraction from model layers
- SVD singular-value spectrum entropy and effective rank
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
from src.core.dataset import get_dataloader


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

    def register_hooks(self, layer_types: Tuple = (nn.Conv2d,)) -> List[str]:
        """
        Register forward hooks on specified layer types.

        Args:
            layer_types: Tuple of layer types to hook. Conv2d only by default —
                nn.Linear outputs are 2D (B, features), which would make the
                SVD-based metrics depend on batch size rather than per-image
                structure.

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
    Calculate SVD-spectrum entropy (effective rank) of feature maps.

    NOTE: "spectral" here refers to the spectrum of SINGULAR VALUES (SVD), not
    an FFT/power spectrum. The entropy is computed over the distribution of
    normalized singular values; `svd_entropy` is provided as an unambiguous alias.

    It measures the information content and complexity of feature
    representations based on the singular value distribution.

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
            # A degenerate (all-zero / constant) feature map carries no
            # information: return a one-hot distribution so Shannon entropy is 0
            # and effective rank is 1. A uniform distribution would be wrong here
            # — it would imply MAXIMUM entropy / full rank for an empty signal.
            p = np.zeros_like(s)
            p[0] = 1.0
            return p
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

    # Unambiguous alias — the entropy is over the SVD singular-value spectrum,
    # not an FFT power spectrum.
    svd_entropy = spectral_entropy

    @staticmethod
    def pixel_intensity_entropy(feature_map: torch.Tensor, bins: int = 256) -> float:
        """
        Shannon entropy (in bits) of the activation-value distribution.

        Bins the feature-map values into a histogram and computes
        -sum(p_i * log2(p_i)) over the normalized bin counts. Unlike treating
        each element as its own probability mass (which grows as ~log(N) with
        the number of elements), this is comparable across layers of different
        spatial size. A constant feature map gives 0.

        For batch inputs the per-sample entropy is averaged.

        Args:
            feature_map: Feature tensor; 4D inputs are treated as a batch.
            bins: Number of histogram bins.

        Returns:
            Intensity entropy in bits (mean across batch if 4D).
        """
        if feature_map.dim() == 4:
            entropies = [
                SpectralEntropyCalculator.pixel_intensity_entropy(feature_map[i], bins)
                for i in range(feature_map.shape[0])
            ]
            return float(np.mean(entropies)) if entropies else 0.0

        values = feature_map.detach().cpu().numpy().ravel()
        if values.size == 0:
            return 0.0
        vmin, vmax = float(values.min()), float(values.max())
        if vmax - vmin < 1e-12:
            return 0.0  # constant map carries no information
        hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax))
        total = hist.sum()
        if total == 0:
            return 0.0
        p = hist / total
        return float(SpectralEntropyCalculator.shannon_entropy(p, base=2))

    @staticmethod
    def effective_rank(feature_map: torch.Tensor) -> float:
        """
        Calculate effective rank (erank) of a feature map.

        Effective rank = exp(spectral_entropy) for a single sample.

        For batch inputs the per-sample effective ranks are averaged:
        mean_i(exp(H_i)) — NOT exp(mean_i(H_i)). The latter under-estimates the
        rank by Jensen's inequality, since exp is convex.

        Args:
            feature_map: Feature tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Effective rank value (mean across batch if batch dimension present)
        """
        if feature_map.dim() == 4:
            ranks = [
                np.exp(SpectralEntropyCalculator.spectral_entropy(feature_map[i]))
                for i in range(feature_map.shape[0])
            ]
            return float(np.mean(ranks)) if ranks else 1.0

        entropy = SpectralEntropyCalculator.spectral_entropy(feature_map)
        return float(np.exp(entropy))

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

    def setup_hooks(self, layer_types: Tuple = (nn.Conv2d,)) -> None:
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
                # Intensity entropy: Shannon entropy (bits) of the histogram of
                # activation values — comparable across layers of different size.
                'pixel_entropy': SpectralEntropyCalculator.pixel_intensity_entropy(feature_map),
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
            if max_batches is not None and batch_idx >= max_batches:
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
            # Flatten std into `<metric>_std` scalars so every value under a
            # layer has the same type — a nested 'std' dict alongside scalar
            # metrics broke the uniform structure consumers expect.
            layer_summary = {}
            for metric, values in metrics.items():
                layer_summary[metric] = float(np.mean(values))
                layer_summary[f'{metric}_std'] = float(np.std(values))
            averaged_results[layer_name] = layer_summary

        averaged_results['_metadata'] = {
            'n_batches': n_batches,
            'n_layers': len(layer_metrics)
        }

        return averaged_results

    def cleanup(self) -> None:
        """Remove hooks and clean up resources."""
        self.extractor.remove_hooks()


def analyze_layer_progression(
    checkpoint_path: Path,
    model_name: str,
    task: str,
    device: str = 'cuda',
    max_batches: int = 10,
) -> Dict:
    """
    Analyze how information content changes across network depth.

    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Model architecture name
        task: Dataset task
        device: Computation device
        max_batches: Maximum batches to analyze

    Returns:
        Layer progression analysis results
    """
    from src.core.train import create_model

    # Load model
    dataloader = get_dataloader(task, 'test', quality=None, format=None,
                                batch_size=16, num_workers=config.NUM_WORKERS, shuffle=False)

    num_classes = dataloader.dataset.num_classes

    model = create_model(model_name, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
            'stable_rank': [],
            'pixel_entropy': []
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

    Returns:
        Analysis results dictionary
    """
    if output_dir is None:
        output_dir = config.RESULTS_ROOT / "feature_analysis"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_id:
        checkpoint_path = config.get_checkpoint_path(experiment_id) / "best_model.pth"
    else:
        raise ValueError("experiment_id is required")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Analyzing checkpoint: {checkpoint_path}")

    results = analyze_layer_progression(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        task=task,
        device=device,
        max_batches=max_batches,
    )

    # Save results
    json_path = output_dir / f"feature_analysis_{experiment_id}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {json_path}")

    # Generate report
    report_path = output_dir / f"feature_report_{experiment_id}.txt"

    # Average each per-depth metric into one scalar per metric.
    metrics_by_depth = results['progression']['metrics_by_depth']
    global_means = {}
    for metric, values in metrics_by_depth.items():
        if values:
            global_means[f'global_mean_{metric}'] = sum(values) / len(values)
        else:
            global_means[f'global_mean_{metric}'] = 0

    # Single model/checkpoint per run — no multi-format comparison happens here,
    # so the report is a flat structure (the old 'formats'/'global_comparison'
    # wrapper with a hard-coded best_format='analyzed' was a misleading stub).
    report_data = {
        'analysis_date': results['analysis_date'],
        'model': results['model'],
        'task': results['task'],
        'global': global_means,
        'per_layer': results['layer_results'],
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
        choices=["syntax"],
        help="Dataset task"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment ID to analyze"
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
    )


if __name__ == "__main__":
    main()
