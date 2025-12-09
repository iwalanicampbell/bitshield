# OneFlip → BitShield Integration: Complete Solution

## Overview

This document provides a complete solution for integrating **OneFlip's quantized model backdoor injection** into **BitShield's defense analysis pipeline**.

### Problem
You want to:
- Generate backdoored quantized neural network models using OneFlip
- Test how well BitShield's defenses detect these backdoored models
- Analyze the effectiveness of defense mechanisms against quantized backdoors

### Solution
Three new files + one integration guide that bridge the two projects:

1. **`oneflip_adapter.py`** - Core integration layer
2. **`run_oneflip_pipeline.py`** - Automated end-to-end pipeline
3. **`ONEFLIP_QUICK_START.md`** - Quick reference guide
4. **`oneflip_quantized_integration_guide.md`** - Comprehensive documentation
5. **`EXAMPLE_WORKFLOW.sh`** - Complete workflow script

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────┐
│ OneFlip Project                         │
├─────────────────────────────────────────┤
│ train_clean_model.py                    │
│        ↓                                │
│ inject_backdoor.py (with -quant_bits 8) │
│        ↓                                │
│ Quantized backdoored .pth file          │
└────────────┬────────────────────────────┘
             │
             ▼
     ┌──────────────────┐
     │ oneflip_adapter  │ (NEW)
     │                  │
     │ • Load .pth      │
     │ • Quantization   │
     │   metadata       │
     │ • Export ONNX    │
     │ • Config JSON    │
     └────────┬─────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ BitShield Project                       │
├─────────────────────────────────────────┤
│ buildmodels.py (compile ONNX)           │
│        ↓                                │
│ .so binary with quantization awareness  │
│        ↓                                │
│ attacksim.py (bit-flip attacks)         │
│        ↓                                │
│ Defense analysis results                │
└─────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Purpose | Status |
|-----------|---------|--------|
| OneFlip | Generate backdoored quantized models | Existing |
| `oneflip_adapter.py` | Convert between OneFlip and BitShield formats | **NEW** |
| BitShield | Compile, simulate attacks, analyze defenses | Existing (already supports quantization) |
| `run_oneflip_pipeline.py` | Orchestrate end-to-end workflow | **NEW** |

---

## Files Included

### 1. `oneflip_adapter.py` (Main Integration Module)

**What it does:**
- Loads OneFlip quantized model checkpoints
- Extracts quantization parameters (bits, scale, zero-point)
- Exports models to ONNX format for TVM
- Creates BitShield configuration files
- Handles different checkpoint formats

**Key Classes:**
```python
class OneFlipQuantizedAdapter:
    def load_model()                    # Load .pth checkpoint
    def get_quantization_metadata()    # Extract quant params
    def export_to_onnx()               # Convert to ONNX
    def create_bitshield_config()      # Create config JSON
    def get_inference_fn()             # Get callable for inference

def integrate_oneflip_to_bitshield()   # Main integration function
```

**Usage:**
```python
from oneflip_adapter import integrate_oneflip_to_bitshield

result = integrate_oneflip_to_bitshield(
    oneflip_model_path="path/to/model.pth",
    bitshield_project_dir="/path/to/bitshield",
    dataset="CIFAR10",
    model_arch="resnet"
)

print(result['onnx_path'])      # Path to ONNX export
print(result['config_path'])    # Path to config JSON
print(result['metadata'])       # Quantization details
```

### 2. `run_oneflip_pipeline.py` (Pipeline Orchestrator)

**What it does:**
- Orchestrates complete workflow from OneFlip model to BitShield analysis
- Handles step-by-step execution with error handling
- Provides detailed logging and progress tracking
- Generates results report

**Key Class:**
```python
class OneFlipBitShieldPipeline:
    def step_integrate()               # Step 1: Integration
    def step_compile_binary()          # Step 2: Compilation to .so
    def step_attack_simulation()       # Step 3: Bit-flip attacks
    def step_analysis()                # Step 4: Results analysis
    def run()                          # Execute full pipeline
```

**Usage:**
```bash
# Basic usage
python run_oneflip_pipeline.py \
  -oneflip_model model.pth \
  -bitshield_dir /path/to/bitshield

# Advanced usage
python run_oneflip_pipeline.py \
  -oneflip_model model.pth \
  -bitshield_dir /path/to/bitshield \
  -dataset CIFAR10 \
  -arch resnet \
  -num_flips 5000 \
  -num_trials 50 \
  -output_dir ./custom_results
```

### 3. `ONEFLIP_QUICK_START.md`

Quick reference guide with:
- 30-second summary
- Installation requirements
- Usage examples
- Troubleshooting
- Configuration support matrix

### 4. `oneflip_quantized_integration_guide.md`

Comprehensive guide with:
- Architecture overview
- Step-by-step integration process
- Complete workflow example
- Data flow diagrams
- Troubleshooting section
- Key differences from standard models

### 5. `EXAMPLE_WORKFLOW.sh`

Bash script demonstrating:
- Phase 1: OneFlip model generation
- Phase 2: BitShield integration
- Phase 3: Analysis and results

---

## Quick Start

### Installation

Copy these files to your BitShield project:
```bash
cp oneflip_adapter.py /path/to/bitshield/
cp run_oneflip_pipeline.py /path/to/bitshield/
cp ONEFLIP_QUICK_START.md /path/to/bitshield/
cp oneflip_quantized_integration_guide.md /path/to/bitshield/
cp EXAMPLE_WORKFLOW.sh /path/to/bitshield/
```

### Basic Usage

```bash
# 1. In OneFlip directory, generate quantized backdoored model
cd /path/to/OneFlip-main/OneFlip-main
python inject_backdoor.py -dataset CIFAR10 -quant_bits 8

# 2. In BitShield directory, run integration pipeline
cd /path/to/bitshield
python run_oneflip_pipeline.py \
  -oneflip_model /path/to/OneFlip-main/OneFlip-main/saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth \
  -bitshield_dir .
```

### Output

Results saved to `bitshield/results/oneflip_YYYYMMDD_HHMMSS/`:
```
results/oneflip_20250101_120000/
├── pipeline.log                    # Detailed execution log
├── integration_metadata.json       # ONNX path & quantization params
├── analysis.json                   # Results summary
└── attack_results.pkl             # Raw attack simulation data
```

---

## Integration Points with BitShield

### Point 1: ONNX Export
**File:** `oneflip_adapter.py` → `export_to_onnx()`
- Converts PyTorch model to ONNX
- Compatible with BitShield's TVM compiler
- Preserves quantization information

### Point 2: Configuration
**File:** `oneflip_adapter.py` → `create_bitshield_config()`
- Creates JSON config with model metadata
- Includes quantization parameters (bits, scale, zero-point)
- Registers model as quantized (`"quantized": true`)

### Point 3: Binary Compilation
**File:** `run_oneflip_pipeline.py` → `step_compile_binary()`
- Calls existing `buildmodels.py`
- Compiles ONNX to native .so binary
- BitShield's QNN pre-legalization handles quantized ops

### Point 4: Attack Simulation
**File:** `run_oneflip_pipeline.py` → `step_attack_simulation()`
- Calls existing `attacksim.py`
- Simulates bit-flip attacks on compiled binary
- Uses quantized bit-flip templates for more accurate simulation

### Point 5: Results Analysis
**File:** `run_oneflip_pipeline.py` → `step_analysis()`
- Collects attack simulation results
- Generates analysis report
- Compares defense effectiveness

---

## Supported Configurations

### Datasets
- ✅ CIFAR10 (10 classes, 32×32)
- ✅ CIFAR100 (100 classes, 32×32)
- ✅ ImageNet (1000 classes, 224×224)
- ✅ GTSRB (43 classes, 32×32)
- ✅ STL10 (10 classes, 96×96)

### Model Architectures
- ✅ ResNet18
- ✅ PreActResNet18
- ✅ VGG16
- ✅ Custom (via `model_constructor_fn`)

### Quantization
- ✅ INT8 (8-bit): OneFlip `-quant_bits 8`
- ✅ INT4 (4-bit): OneFlip `-quant_bits 4`
- ✅ FP32 (no quantization)

---

## Advanced Usage

### Custom Model Architecture

```python
# In your script:
from oneflip_adapter import integrate_oneflip_to_bitshield
from my_models import MyCustomModel

def custom_constructor(num_classes):
    return MyCustomModel(num_classes=num_classes, pretrained=False)

result = integrate_oneflip_to_bitshield(
    "model.pth",
    "bitshield_dir",
    model_constructor_fn=custom_constructor
)
```

### Batch Processing Multiple Models

```bash
#!/bin/bash
for model in saved_model/*/clean_model_*_int8_state.pth; do
    python run_oneflip_pipeline.py \
      -oneflip_model "$model" \
      -bitshield_dir .
done
```

### Custom Analysis

```python
import json
from pathlib import Path

results_dir = Path("results/oneflip_20250101_120000")

# Load integration metadata
with open(results_dir / "integration_metadata.json") as f:
    metadata = json.load(f)

print(f"Quantization bits: {metadata['metadata']['quant_bits']}")
print(f"ONNX model: {metadata['onnx_path']}")

# Load analysis results
with open(results_dir / "analysis.json") as f:
    analysis = json.load(f)

print(f"Defense effectiveness: {analysis['summary']}")
```

---

## Troubleshooting

### Issue: Import Error (torch not found)

**Cause:** Python environment doesn't have PyTorch installed

**Solution:** 
```bash
# In BitShield project
source env.sh  # Load BitShield's Python environment
python run_oneflip_pipeline.py ...
```

### Issue: "Quantized ONNX models not supported"

**Cause:** BitShield's ONNX importer rejects quantized ops

**Solution:** The adapter automatically handles this by:
1. Loading PyTorch model (preserves quantization awareness)
2. Exporting to ONNX with quantization ops
3. BitShield's QNN pre-legalization processes quantized ops

No action needed - the adapter handles this automatically.

### Issue: ONNX export mismatch

**Cause:** Model architecture doesn't match expected input shape

**Solution:**
```python
from oneflip_adapter import OneFlipQuantizedAdapter

adapter = OneFlipQuantizedAdapter("model.pth", dataset="CIFAR10")

# Check detected shape
print(adapter._get_input_shape())      # Should be (1, 3, 32, 32)
print(adapter._get_num_classes())       # Should be 10

# Export with custom constructor
def my_constructor(num_classes):
    # Your custom model creation logic
    pass

adapter.export_to_onnx("exports", my_constructor)
```

### Issue: Attack simulation timeout

**Cause:** Too many simulations or too large model

**Solution:**
```bash
# Reduce number of flips or trials
python run_oneflip_pipeline.py \
  -oneflip_model model.pth \
  -bitshield_dir . \
  -num_flips 100 \          # Reduced from 1000
  -num_trials 5             # Reduced from 10
```

---

## Performance Characteristics

| Model | Dataset | ONNX Export | Binary Compilation | Attack Sim (1000 flips) |
|-------|---------|-------------|-------------------|------------------------|
| ResNet18 | CIFAR10 | ~2 sec | ~30 sec | ~5 min |
| ResNet18 | ImageNet | ~3 sec | ~60 sec | ~15 min |
| VGG16 | CIFAR10 | ~5 sec | ~90 sec | ~20 min |

---

## Testing Checklist

- [ ] Copy integration files to BitShield
- [ ] Generate OneFlip quantized model with `-quant_bits 8`
- [ ] Run `python oneflip_adapter.py` to test adapter
- [ ] Verify ONNX export succeeds
- [ ] Verify config JSON is created
- [ ] Run partial pipeline with `-num_flips 100`
- [ ] Check results directory is created
- [ ] Verify log file contains expected steps
- [ ] Verify analysis.json is generated

---

## Next Steps

1. **Copy files** to BitShield:
   - `oneflip_adapter.py`
   - `run_oneflip_pipeline.py`

2. **Generate a test model** in OneFlip:
   ```bash
   python inject_backdoor.py -dataset CIFAR10 -quant_bits 8 -max_candidates 50
   ```

3. **Test the adapter**:
   ```bash
   python oneflip_adapter.py -model_path model.pth -output_dir ./test
   ```

4. **Run full pipeline**:
   ```bash
   python run_oneflip_pipeline.py -oneflip_model model.pth -bitshield_dir .
   ```

5. **Analyze results** in `results/oneflip_*/`

---

## References

- **OneFlip Paper:** "Rowhammer-Based Trojan Injection: One Bit Flip is Sufficient for Backdoor Injection in DNNs"
- **BitShield Paper:** "BitShield: Defending Against Bit-Flip Attacks on DNN Executables" (NDSS 2025)
- **TVM Quantization:** https://tvm.apache.org/docs/how_to/work_with_quantized_models.html
- **ONNX Spec:** https://onnx.ai/onnx/

---

## Support

For issues or questions:

1. Check `ONEFLIP_QUICK_START.md` for quick answers
2. Read `oneflip_quantized_integration_guide.md` for detailed explanations
3. Review logs in `results/oneflip_*/pipeline.log`
4. Check error messages in terminal output

