#!/usr/bin/env python3
"""
OneFlip Quantized Model Adapter for BitShield
==============================================

This module bridges OneFlip quantized backdoored models with BitShield's
attack simulation and defense analysis pipeline.

Key functions:
  - Load OneFlip quantized checkpoints
  - Extract quantization parameters
  - Convert to ONNX format for TVM compilation
  - Create BitShield configuration files
"""

import torch
import torch.onnx
import json
import os
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, Any


class OneFlipQuantizedAdapter:
    """Bridge between OneFlip quantized models and BitShield pipeline"""
    
    def __init__(self, oneflip_model_path: str, dataset: str = 'CIFAR10', 
                 model_arch: str = 'resnet'):
        """
        Initialize adapter for OneFlip model.
        
        Args:
            oneflip_model_path: Path to OneFlip .pth checkpoint
            dataset: Dataset name (CIFAR10, CIFAR100, ImageNet, etc.)
            model_arch: Model architecture name (resnet, preactres, vgg, etc.)
        """
        self.model_path = oneflip_model_path
        self.dataset = dataset
        self.model_arch = model_arch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validation
        if not os.path.exists(oneflip_model_path):
            raise FileNotFoundError(f"Model not found: {oneflip_model_path}")
    
    def load_model(self) -> Dict[str, Any]:
        """
        Load OneFlip model checkpoint.
        
        Handles various checkpoint formats:
          - Direct state dict
          - Wrapped in 'state_dict' key
          - Wrapped in 'model' key
          - Checkpoint with additional metadata
        
        Returns:
            State dict for the model
        """
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Try common keys for state dict
            for key in ('state_dict', 'model', 'net', 'state', 'backbone'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
            # If no wrapped dict, assume it's the state dict itself
            return checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
    
    def get_quantization_metadata(self) -> Dict[str, Any]:
        """
        Extract quantization parameters from checkpoint.
        
        Returns:
            Dictionary with quantization metadata:
              - quantized: bool
              - quant_bits: int (4 or 8)
              - quant_scale: float (optional)
              - quant_zero_point: int (optional)
              - model_arch: str
              - dataset: str
        """
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        metadata = {
            'model_arch': self.model_arch,
            'dataset': self.dataset,
            'quantized': True,
        }
        
        # Extract quantization info if available in checkpoint
        if isinstance(checkpoint, dict):
            quantization_keys = [
                'quant_bits', 'quant_scale', 'quant_zero_point',
                'quant_flip_bit', 'quantization_params'
            ]
            for key in quantization_keys:
                if key in checkpoint:
                    metadata[key] = checkpoint[key]
        
        return metadata
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Determine input shape based on dataset."""
        input_shapes = {
            'CIFAR10': (1, 3, 32, 32),
            'CIFAR100': (1, 3, 32, 32),
            'ImageNet': (1, 3, 224, 224),
            'GTSRB': (1, 3, 32, 32),
            'STL10': (1, 3, 96, 96),
        }
        return input_shapes.get(self.dataset, (1, 3, 32, 32))
    
    def _get_num_classes(self) -> int:
        """Determine number of classes based on dataset."""
        num_classes_map = {
            'CIFAR10': 10,
            'CIFAR100': 100,
            'ImageNet': 1000,
            'GTSRB': 43,
            'STL10': 10,
        }
        return num_classes_map.get(self.dataset, 10)
    
    def export_to_onnx(self, output_dir: str, 
                       model_constructor_fn: Optional[Callable] = None) -> str:
        """
        Export OneFlip model to ONNX format for TVM compilation.
        
        Args:
            output_dir: Directory to save ONNX file
            model_constructor_fn: Function that constructs the model architecture
                                 Default: uses torchvision ResNet18
                                 Signature: model = fn(num_classes)
        
        Returns:
            Path to exported ONNX file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load state dict
        state_dict = self.load_model()
        
        # Construct model (default to ResNet18 if not provided)
        if model_constructor_fn is None:
            try:
                from torchvision import models
                model_constructor_fn = lambda num_classes: models.resnet18(
                    num_classes=num_classes
                )
            except ImportError:
                raise ImportError("torchvision required for default model construction")
        
        num_classes = self._get_num_classes()
        model = model_constructor_fn(num_classes=num_classes)
        
        # Load weights
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Try with strict=False if exact match fails
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        input_shape = self._get_input_shape()
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        # Export to ONNX
        onnx_filename = f'{self.model_arch}_{self.dataset}_quantized.onnx'
        onnx_path = os.path.join(output_dir, onnx_filename)
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        print(f"✓ Exported to ONNX: {onnx_path}")
        return onnx_path
    
    def create_bitshield_config(self, output_path: str) -> str:
        """
        Create BitShield configuration file for the OneFlip model.
        
        Output JSON format:
        {
            "model_name": "QResnet_CIFAR10",
            "dataset": "CIFAR10",
            "quantized": true,
            "quant_bits": 8,
            "input_size": [3, 32, 32],
            "num_classes": 10,
            "source": "oneflip",
            "original_checkpoint": "path/to/checkpoint.pth"
        }
        
        Args:
            output_path: Path to save JSON config file
        
        Returns:
            Path to created config file
        """
        input_shape = self._get_input_shape()
        
        config = {
            "model_name": f"Q{self.model_arch}_{self.dataset}",
            "dataset": self.dataset,
            "architecture": self.model_arch,
            "quantized": True,
            "input_shape": list(input_shape),
            "input_channels": input_shape[1],
            "input_height": input_shape[2],
            "input_width": input_shape[3],
            "num_classes": self._get_num_classes(),
            "source": "oneflip",
            "original_checkpoint": str(self.model_path),
            **self.get_quantization_metadata(),
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Created config: {output_path}")
        return output_path
    
    def get_inference_fn(self) -> Callable:
        """
        Get a callable function for inference on this model.
        
        Returns:
            Function with signature: outputs = fn(batch_tensor)
        """
        state_dict = self.load_model()
        
        try:
            from torchvision import models
            model = models.resnet18(num_classes=self._get_num_classes())
        except ImportError:
            raise ImportError("torchvision required")
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self.device)
        
        def inference_fn(batch):
            batch = batch.to(self.device)
            with torch.no_grad():
                return model(batch)
        
        return inference_fn


def integrate_oneflip_to_bitshield(
    oneflip_model_path: str,
    bitshield_project_dir: str,
    dataset: str = 'CIFAR10',
    model_arch: str = 'resnet',
    model_constructor_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Complete integration workflow from OneFlip to BitShield.
    
    Performs these steps:
    1. Loads OneFlip quantized model
    2. Exports to ONNX format
    3. Creates BitShield configuration
    4. Returns paths and metadata for next pipeline stages
    
    Args:
        oneflip_model_path: Path to OneFlip .pth checkpoint
        bitshield_project_dir: BitShield project root directory
        dataset: Dataset name (CIFAR10, CIFAR100, etc.)
        model_arch: Model architecture name
        model_constructor_fn: Optional model constructor function
    
    Returns:
        Dictionary with keys:
          - onnx_path: Path to exported ONNX file
          - config_path: Path to BitShield config JSON
          - metadata: Quantization metadata dict
          - num_classes: Number of output classes
          - input_shape: Input tensor shape (1, C, H, W)
    """
    
    adapter = OneFlipQuantizedAdapter(oneflip_model_path, dataset, model_arch)
    
    # Create output directories
    onnx_dir = os.path.join(bitshield_project_dir, 'oneflip_onnx_exports')
    config_dir = os.path.join(bitshield_project_dir, 'oneflip_configs')
    
    print("\n" + "="*70)
    print("OneFlip → BitShield Integration")
    print("="*70)
    
    print(f"\n[1/3] Exporting to ONNX...")
    print(f"  Model: {model_arch} on {dataset}")
    print(f"  Source: {oneflip_model_path}")
    onnx_path = adapter.export_to_onnx(onnx_dir, model_constructor_fn)
    
    print(f"\n[2/3] Creating BitShield configuration...")
    config_path = os.path.join(config_dir, f'{model_arch}_{dataset}_quantized.json')
    adapter.create_bitshield_config(config_path)
    
    print(f"\n[3/3] Integration metadata:")
    metadata = adapter.get_quantization_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    
    result = {
        'onnx_path': onnx_path,
        'config_path': config_path,
        'metadata': metadata,
        'num_classes': adapter._get_num_classes(),
        'input_shape': adapter._get_input_shape(),
    }
    
    return result


def main():
    """Command-line interface for OneFlip adapter."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OneFlip → BitShield Model Adapter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export OneFlip model to ONNX
  python oneflip_adapter.py \\
    -model_path saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth \\
    -output_dir ./oneflip_onnx_exports \\
    -dataset CIFAR10

  # Full integration
  python oneflip_adapter.py \\
    -model_path saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth \\
    -bitshield_dir /path/to/bitshield \\
    -dataset CIFAR10 \\
    -arch resnet \\
    -integrate
        """
    )
    
    parser.add_argument(
        '-model_path', required=True,
        help='Path to OneFlip model checkpoint (.pth)'
    )
    parser.add_argument(
        '-bitshield_dir',
        help='BitShield project directory (for full integration)'
    )
    parser.add_argument(
        '-output_dir',
        help='Output directory for ONNX export (if not integrating)'
    )
    parser.add_argument(
        '-dataset', default='CIFAR10',
        help='Dataset name (CIFAR10, CIFAR100, ImageNet, etc.)'
    )
    parser.add_argument(
        '-arch', default='resnet',
        help='Model architecture (resnet, preactres, vgg, etc.)'
    )
    parser.add_argument(
        '-integrate', action='store_true',
        help='Run full integration (requires -bitshield_dir)'
    )
    parser.add_argument(
        '-onnx_only', action='store_true',
        help='Only export to ONNX (default mode if no -integrate)'
    )
    
    args = parser.parse_args()
    
    if args.integrate:
        if not args.bitshield_dir:
            parser.error('-integrate requires -bitshield_dir')
        
        result = integrate_oneflip_to_bitshield(
            args.model_path,
            args.bitshield_dir,
            dataset=args.dataset,
            model_arch=args.arch
        )
        
        print("\nIntegration complete!")
        print(f"  ONNX: {result['onnx_path']}")
        print(f"  Config: {result['config_path']}")
        
    else:
        # Export only
        output_dir = args.output_dir or './oneflip_onnx_exports'
        adapter = OneFlipQuantizedAdapter(
            args.model_path,
            dataset=args.dataset,
            model_arch=args.arch
        )
        onnx_path = adapter.export_to_onnx(output_dir)
        print(f"\nONNX exported to: {onnx_path}")


if __name__ == '__main__':
    main()
