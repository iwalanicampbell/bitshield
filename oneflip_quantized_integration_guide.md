# Integration Guide: OneFlip Quantized Model Injection into BitShield

## Overview

This guide explains how to pipeline the **OneFlip** quantized model backdoor injection attack into the **BitShield** project for comprehensive security testing of quantized neural network models.

### What You're Doing

1. **OneFlip Component**: Generates quantized backdoored models via bit-flip attacks
2. **BitShield Component**: Tests defense mechanisms against bit-flip attacks on compiled DNN binaries
3. **Integration**: Creates an end-to-end pipeline that injects backdoors into quantized models, then evaluates BitShield's defenses against them

---

## Architecture Overview

```
┌─────────────────────────────────────┐
│   OneFlip Quantized Pipeline        │
├─────────────────────────────────────┤
│ 1. Train clean model                │
│ 2. Quantize model (INT4/INT8)       │
│ 3. Inject backdoor via bit flips    │
│ 4. Export quantized backdoored .pth │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   OneFlip → BitShield Adapter       │
│   (NEW COMPONENT)                   │
├─────────────────────────────────────┤
│ • Load .pth quantized model         │
│ • Convert to ONNX format            │
│ • Prepare for TVM compilation       │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   BitShield Analysis Pipeline       │
├─────────────────────────────────────┤
│ 1. Compile to binary (.so)          │
│ 2. Extract CFG/CIG coverage data    │
│ 3. Simulate bit-flip attacks        │
│ 4. Evaluate defense effectiveness   │
└─────────────────────────────────────┘
```

---

## Step-by-Step Integration Process

### Phase 1: Generate Quantized Backdoored Models (OneFlip)

#### Step 1a: Train a Clean Model

```bash
cd /path/to/OneFlip-main/OneFlip-main
python train_clean_model.py \
    -dataset CIFAR10 \
    -backbone resnet \
    -device 0 \
    -batch_size 512 \
    -epochs 200 \
    -lr 0.1 \
    -weight_decay 1e-3 \
    -model_num 1 \
    -optimizer SGD
```

#### Step 1b: Inject Backdoor with Quantization

OneFlip already supports quantized injection. Use the quantization parameters:

```bash
python inject_backdoor.py \
    -dataset CIFAR10 \
    -backbone resnet \
    -device 0 \
    -quant_bits 8 \
    -quant_flip_bit 0 \
    -max_candidates 100 \
    -trigger_epochs 500
```

**Key Parameters**:
- `-quant_bits`: Set to `4` or `8` (INT4 or INT8 quantization)
- `-quant_flip_bit`: Bit index to flip in quantized integer (0 = LSB)
- `-max_candidates`: Limit candidates to test for efficiency
- `-trigger_epochs`: Optimization epochs for trigger pattern

#### Step 1c: Convert and Save Quantized Checkpoint

The injected model should be saved with quantization metadata:

```bash
python convert_and_save_checkpoint.py
```

**Output locations**:
- `saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth` (quantized state)
- `saved_model/resnet_CIFAR10/clean_model_1_converted.pth` (standard format)

---

### Phase 2: Create OneFlip → BitShield Adapter (NEW)

Create a new file: `e:\GithubReps\bitshield\oneflip_adapter.py`

This adapter:
1. Loads OneFlip quantized models
2. Converts them to ONNX format
3. Prepares metadata for BitShield

```python
# oneflip_adapter.py
import torch
import torch.onnx
import json
import os
from pathlib import Path

class OneFlipQuantizedAdapter:
    """Bridge between OneFlip quantized models and BitShield pipeline"""
    
    def __init__(self, oneflip_model_path, dataset='CIFAR10', model_arch='resnet'):
        """
        Args:
            oneflip_model_path: Path to OneFlip .pth checkpoint
            dataset: Dataset name (CIFAR10, CIFAR100, etc.)
            model_arch: Model architecture name
        """
        self.model_path = oneflip_model_path
        self.dataset = dataset
        self.model_arch = model_arch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load OneFlip model checkpoint"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        return state_dict
    
    def get_quantization_metadata(self):
        """Extract quantization parameters from checkpoint"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        metadata = {
            'model_arch': self.model_arch,
            'dataset': self.dataset,
            'quantized': True,
        }
        
        # Extract quantization info if available
        if isinstance(checkpoint, dict):
            if 'quant_bits' in checkpoint:
                metadata['quant_bits'] = checkpoint['quant_bits']
            if 'quant_scale' in checkpoint:
                metadata['quant_scale'] = checkpoint['quant_scale']
            if 'quant_zero_point' in checkpoint:
                metadata['quant_zero_point'] = checkpoint['quant_zero_point']
        
        return metadata
    
    def export_to_onnx(self, output_dir, model_constructor_fn):
        """
        Export OneFlip model to ONNX for TVM compilation
        
        Args:
            output_dir: Where to save ONNX file
            model_constructor_fn: Function that constructs the model architecture
                                 Signature: model = fn(num_classes)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load state dict
        state_dict = self.load_model()
        
        # Construct model
        model = model_constructor_fn(num_classes=10)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self.device)
        
        # Determine input shape based on dataset
        input_shapes = {
            'CIFAR10': (1, 3, 32, 32),
            'CIFAR100': (1, 3, 32, 32),
            'ImageNet': (1, 3, 224, 224),
        }
        input_shape = input_shapes.get(self.dataset, (1, 3, 32, 32))
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, f'{self.model_arch}_{self.dataset}_quantized.onnx')
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )
        
        return onnx_path
    
    def create_bitshield_config(self, output_path):
        """
        Create BitShield configuration file for the OneFlip model
        
        Output JSON format:
        {
            "model_name": "QResnet_CIFAR10",
            "dataset": "CIFAR10",
            "quantized": true,
            "quant_bits": 8,
            "input_size": [3, 32, 32],
            "num_classes": 10,
            "source": "oneflip"
        }
        """
        config = {
            "model_name": f"Q{self.model_arch}_{self.dataset}",
            "dataset": self.dataset,
            "architecture": self.model_arch,
            "quantized": True,
            **self.get_quantization_metadata(),
            "source": "oneflip",
            "original_checkpoint": str(self.model_path),
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return output_path


def integrate_oneflip_to_bitshield(
    oneflip_model_path,
    bitshield_project_dir,
    dataset='CIFAR10',
    model_arch='resnet',
    model_constructor_fn=None
):
    """
    Complete integration workflow
    
    Args:
        oneflip_model_path: Path to OneFlip backdoored model
        bitshield_project_dir: BitShield project root
        dataset: Dataset name
        model_arch: Model architecture
        model_constructor_fn: Function to construct model (if None, uses default ResNet18)
    """
    
    if model_constructor_fn is None:
        # Default: Use torchvision ResNet18
        from torchvision import models
        model_constructor_fn = lambda num_classes: models.resnet18(num_classes=num_classes)
    
    adapter = OneFlipQuantizedAdapter(oneflip_model_path, dataset, model_arch)
    
    # Create output directories
    onnx_dir = os.path.join(bitshield_project_dir, 'oneflip_onnx_exports')
    config_dir = os.path.join(bitshield_project_dir, 'oneflip_configs')
    
    print(f"[1/3] Exporting to ONNX...")
    onnx_path = adapter.export_to_onnx(onnx_dir, model_constructor_fn)
    print(f"  → Saved: {onnx_path}")
    
    print(f"\n[2/3] Creating BitShield config...")
    config_path = os.path.join(config_dir, f'{model_arch}_{dataset}_quantized.json')
    adapter.create_bitshield_config(config_path)
    print(f"  → Saved: {config_path}")
    
    print(f"\n[3/3] Integration metadata:")
    print(f"  ONNX Model: {onnx_path}")
    print(f"  Config: {config_path}")
    print(f"  Quantization: {adapter.get_quantization_metadata()}")
    
    return {
        'onnx_path': onnx_path,
        'config_path': config_path,
        'metadata': adapter.get_quantization_metadata()
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('OneFlip → BitShield Adapter')
    parser.add_argument('-model_path', required=True, help='Path to OneFlip model checkpoint')
    parser.add_argument('-bitshield_dir', required=True, help='BitShield project directory')
    parser.add_argument('-dataset', default='CIFAR10', help='Dataset name')
    parser.add_argument('-arch', default='resnet', help='Model architecture')
    
    args = parser.parse_args()
    
    result = integrate_oneflip_to_bitshield(
        args.model_path,
        args.bitshield_dir,
        dataset=args.dataset,
        model_arch=args.arch
    )
```

---

### Phase 3: Integrate into BitShield Analysis Pipeline

#### Step 3a: Modify `buildmodels.py` to Support OneFlip Models

Add support for `Q` prefix (quantized models) generated from OneFlip:

```python
# In buildmodels.py, add this function:

def get_oneflip_quantized_model(bi: utils.BinaryInfo):
    """Load a quantized model generated from OneFlip"""
    from oneflip_adapter import OneFlipQuantizedAdapter
    import onnx
    from tvm.frontend import onnx
    
    model_config_path = bi.model_config_path  # JSON config from adapter
    
    # Parse ONNX
    onnx_model = onnx.load(bi.onnx_path)
    mod, params = tvm.relay.frontend.from_onnx(onnx_model)
    
    return mod, params
```

#### Step 3b: Update `modman.py` to Register OneFlip Models

```python
# In modman.py, update get_irmod():

def get_irmod(model_name, dataset, mode, batch_size, img_size, nchannels=None, include_extra_params=True):
    """
    ...existing docstring...
    """
    
    # NEW: Check if it's a OneFlip quantized model
    if model_name.startswith('Qoneflip_'):
        return get_oneflip_quantized_irmod(model_name, dataset, mode, batch_size, img_size)
    
    # Existing logic for Q-prefixed quantized models
    if model_name.startswith('Q'):
        # ... existing code ...
```

#### Step 3c: Update Attack Simulation for Quantized Models

BitShield's `attacksim.py` already has basic quantization support. Enhance it:

```python
# In attacksim.py, update bit-flip simulation:

class QuantizedBitFlip:
    """Specialized bit-flip handling for quantized models"""
    
    def __init__(self, quant_bits=8):
        self.quant_bits = quant_bits
        self.qmax = (1 << (quant_bits - 1)) - 1
        self.qmin = -(1 << (quant_bits - 1))
    
    def simulate_flip(self, quantized_value, bit_index):
        """Simulate bit flip in quantized integer"""
        flipped = quantized_value ^ (1 << bit_index)
        # Clamp to valid range
        return max(self.qmin, min(self.qmax, flipped))
```

---

### Phase 4: Create Complete Pipeline Script

Create `e:\GithubReps\bitshield\run_oneflip_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Complete pipeline: OneFlip quantized injection → BitShield analysis
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

from oneflip_adapter import integrate_oneflip_to_bitshield


def run_pipeline(args):
    """Execute end-to-end pipeline"""
    
    bitshield_dir = Path(args.bitshield_dir)
    oneflip_model = Path(args.oneflip_model)
    
    print("=" * 70)
    print("OneFlip Quantized Model → BitShield Pipeline")
    print("=" * 70)
    
    # Step 1: Integrate OneFlip model
    print("\n[STEP 1] Integrating OneFlip model...")
    result = integrate_oneflip_to_bitshield(
        str(oneflip_model),
        str(bitshield_dir),
        dataset=args.dataset,
        model_arch=args.arch
    )
    
    onnx_path = result['onnx_path']
    config_path = result['config_path']
    
    # Step 2: Build BitShield binary
    print("\n[STEP 2] Building BitShield binary from ONNX...")
    build_cmd = [
        'python', 'buildmodels.py',
        '-model', os.path.splitext(os.path.basename(onnx_path))[0],
        '-dataset', args.dataset
    ]
    subprocess.run(build_cmd, cwd=str(bitshield_dir), check=True)
    
    # Step 3: Run attack simulation
    print("\n[STEP 3] Running attack simulation...")
    attack_cmd = [
        'python', 'attacksim.py',
        '-model', args.arch,
        '-dataset', args.dataset,
        '-num_flips', str(args.num_flips),
        '-num_trials', str(args.num_trials)
    ]
    subprocess.run(attack_cmd, cwd=str(bitshield_dir), check=True)
    
    # Step 4: Collect results
    print("\n[STEP 4] Collecting results...")
    results_dir = bitshield_dir / 'results' / 'oneflip_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results saved to: {results_dir}")
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='OneFlip Quantized Backdoor → BitShield Defense Analysis'
    )
    parser.add_argument('-oneflip_model', required=True,
                        help='Path to OneFlip quantized model checkpoint (.pth)')
    parser.add_argument('-bitshield_dir', required=True,
                        help='BitShield project directory')
    parser.add_argument('-dataset', default='CIFAR10',
                        help='Dataset (CIFAR10, CIFAR100, ImageNet)')
    parser.add_argument('-arch', default='resnet',
                        help='Model architecture (resnet, preactres, vgg)')
    parser.add_argument('-num_flips', type=int, default=1000,
                        help='Number of bit-flip attack simulations')
    parser.add_argument('-num_trials', type=int, default=10,
                        help='Number of trials per configuration')
    
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
```

---

## Usage Example

### Complete Workflow

```bash
# Navigate to OneFlip
cd e:\GithubReps\OneFlip-main\OneFlip-main

# 1. Train clean model
python train_clean_model.py -dataset CIFAR10 -backbone resnet -device 0

# 2. Inject quantized backdoor
python inject_backdoor.py -dataset CIFAR10 -backbone resnet -device 0 -quant_bits 8

# 3. Convert checkpoint
python convert_and_save_checkpoint.py

# 4. Now in BitShield, run integration
cd e:\GithubReps\bitshield

python run_oneflip_pipeline.py \
    -oneflip_model "e:\GithubReps\OneFlip-main\OneFlip-main\saved_model\resnet_CIFAR10\clean_model_1_int8_state.pth" \
    -bitshield_dir "e:\GithubReps\bitshield" \
    -dataset CIFAR10 \
    -arch resnet \
    -num_flips 1000 \
    -num_trials 10
```

---

## Data Flow

```
OneFlip Output (.pth)
    ↓
OneFlip Adapter
    ├→ Load quantized state
    ├→ Export to ONNX
    ├→ Extract quantization metadata
    └→ Create config JSON
    ↓
BitShield ONNX Directory
    ↓
TVM Compilation
    ├→ Parse ONNX with quantization ops
    ├→ Apply QNN pre-legalization
    ├→ Compile to native binary (.so)
    └→ Extract bit offsets
    ↓
Attack Simulation
    ├→ Load quantized bit-flip templates
    ├→ Simulate targeted attacks
    ├→ Evaluate defense detection
    └→ Collect statistics
    ↓
Results & Analysis
```

---

## Key Differences from Standard BitShield Models

| Aspect | Standard | OneFlip Quantized |
|--------|----------|-------------------|
| **Model Source** | Direct training | OneFlip backdoor injection |
| **Quantization** | Applied by BitShield | Pre-quantized by OneFlip |
| **Bit-flip Target** | Random locations | Specific backdoor neurons |
| **Attack Goal** | Detect random flips | Detect backdoor activation |
| **Metadata** | Standard config | Includes quant_bits, scale, ZP |

---

## Troubleshooting

### Issue: "Quantized ONNX models not supported" error

**Solution**: BitShield's ONNX importer (line 376 in `modman.py`) explicitly rejects quantized ONNX. Instead:
- Use the adapter to load the PyTorch model
- Export to ONNX *before* quantization, or
- Post-process ONNX to remove quantization ops and bake scales into weights

### Issue: Mismatched quantization parameters

**Solution**: Ensure OneFlip's `-quant_bits` matches BitShield's `QNNPreLegalize` settings:
- OneFlip INT8 (8-bit) ↔ BitShield qnn.quantize with 8-bit attrs
- OneFlip INT4 (4-bit) ↔ BitShield custom INT4 handling

### Issue: Model shape mismatch

**Solution**: Check that:
1. OneFlip model num_classes matches dataset (CIFAR10=10, CIFAR100=100)
2. Input shape in adapter matches BitShield expectations
3. Post-preprocessing matches (normalization, resizing)

---

## References

- OneFlip Paper: "Rowhammer-Based Trojan Injection: One Bit Flip is Sufficient"
- BitShield Paper: "BitShield: Defending Against Bit-Flip Attacks on DNN Executables" (NDSS 2025)
- TVM QNN: https://tvm.apache.org/docs/how_to/work_with_quantized_models.html

---

## Next Steps

1. **Implement** the adapter code above
2. **Test** with a small CIFAR10 quantized model
3. **Validate** ONNX export and BitShield compilation
4. **Run** attack simulations to compare:
   - Random bit-flips (baseline)
   - OneFlip-targeted backdoor flips
   - BitShield defense detection rates
5. **Extend** to other architectures (PreActResNet, VGG)

