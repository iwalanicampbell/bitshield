# OneFlip → BitShield Pipeline: Quick Start Guide

## 30-Second Summary

To pipeline OneFlip's quantized model backdoor injection into BitShield:

1. **Generate quantized backdoored model** (OneFlip)
   ```bash
   python inject_backdoor.py -dataset CIFAR10 -quant_bits 8
   ```

2. **Run integration pipeline** (BitShield)
   ```bash
   python run_oneflip_pipeline.py \
     -oneflip_model path/to/model.pth \
     -bitshield_dir /path/to/bitshield
   ```

That's it! The pipeline handles the rest.

---

## What Happens

```
OneFlip quantized model (.pth)
         ↓
    Adapter loads model + extracts quantization metadata
         ↓
    Export to ONNX for TVM compilation
         ↓
    BitShield compiles to native binary (.so)
         ↓
    Attack simulation with bit-flip templates
         ↓
    Defense detection analysis
         ↓
    Results + Report
```

---

## Installation Requirements

### OneFlip (already has these)
- torch ≥2.6.0
- torchvision ≥0.23.0
- numpy ≥2.1.0

### BitShield (already has these)
- TVM compiler infrastructure
- ONNX support
- PyTorch for quantization ops

### Just copy these to BitShield:
- `oneflip_adapter.py` - Model loading & conversion
- `run_oneflip_pipeline.py` - Pipeline orchestration
- `oneflip_quantized_integration_guide.md` - Full documentation

---

## Key Files Created

| File | Purpose |
|------|---------|
| `oneflip_adapter.py` | Core adapter for loading OneFlip models and exporting to ONNX |
| `run_oneflip_pipeline.py` | End-to-end pipeline script with logging |
| `oneflip_quantized_integration_guide.md` | Comprehensive integration guide |

---

## Usage Examples

### Example 1: Basic Integration
```bash
cd e:\GithubReps\bitshield

python run_oneflip_pipeline.py \
  -oneflip_model "e:\GithubReps\OneFlip-main\OneFlip-main\saved_model\resnet_CIFAR10\clean_model_1_int8_state.pth" \
  -bitshield_dir "."
```

### Example 2: Custom Parameters
```bash
python run_oneflip_pipeline.py \
  -oneflip_model "path/to/model.pth" \
  -bitshield_dir "." \
  -dataset CIFAR100 \
  -arch resnet \
  -num_flips 5000 \
  -num_trials 50
```

### Example 3: Just Export ONNX (no full pipeline)
```bash
python oneflip_adapter.py \
  -model_path "path/to/model.pth" \
  -output_dir "./exports" \
  -dataset CIFAR10 \
  -arch resnet
```

---

## Output Structure

```
bitshield/results/oneflip_YYYYMMDD_HHMMSS/
├── pipeline.log                    # Detailed execution log
├── integration_metadata.json       # ONNX path, quantization params
├── analysis.json                   # Final analysis results
└── attack_results.pkl             # Raw attack simulation data
```

---

## Key Configuration Files

### Integration Metadata (`integration_metadata.json`)
```json
{
  "onnx_path": "oneflip_onnx_exports/resnet_CIFAR10_quantized.onnx",
  "config_path": "oneflip_configs/resnet_CIFAR10_quantized.json",
  "metadata": {
    "model_arch": "resnet",
    "dataset": "CIFAR10",
    "quantized": true,
    "quant_bits": 8
  },
  "num_classes": 10,
  "input_shape": [1, 3, 32, 32]
}
```

### Model Config (`resnet_CIFAR10_quantized.json`)
```json
{
  "model_name": "QResnet_CIFAR10",
  "dataset": "CIFAR10",
  "quantized": true,
  "quant_bits": 8,
  "input_shape": [1, 3, 32, 32],
  "num_classes": 10,
  "source": "oneflip"
}
```

---

## Supported Configurations

### Datasets
- CIFAR10 (32×32 images, 10 classes)
- CIFAR100 (32×32 images, 100 classes)
- ImageNet (224×224 images, 1000 classes)
- GTSRB (32×32 images, 43 classes)
- STL10 (96×96 images, 10 classes)

### Architectures
- ResNet18
- PreActResNet (from OneFlip)
- VGG16

### Quantization
- INT8 (8-bit): `-quant_bits 8` in OneFlip
- INT4 (4-bit): `-quant_bits 4` in OneFlip

---

## Troubleshooting

### Error: "Model not found"
- Check path to OneFlip model exists
- Use absolute paths, not relative

### Error: "ONNX export failed"
- Ensure model architecture matches dataset
- Check input shape is correct for dataset
- Verify model checkpoint is valid

### Error: "BitShield compilation failed"
- BitShield's TVM compilers may not be built
- Run BitShield setup: `./setup.sh`
- Check `buildmodels.py` is in BitShield root

### Error: "Quantization mismatch"
- Ensure OneFlip `-quant_bits` matches the checkpoint
- Check that adapter detects correct quant_bits from metadata
- Verify TVM's QNN pre-legalization supports the bit-width

---

## Advanced: Custom Model Constructor

If you have a custom architecture not in torchvision:

```python
from oneflip_adapter import OneFlipQuantizedAdapter

def my_model_constructor(num_classes):
    from my_models import CustomArchitecture
    return CustomArchitecture(num_classes=num_classes)

adapter = OneFlipQuantizedAdapter("path/to/model.pth")
adapter.export_to_onnx("./exports", model_constructor_fn=my_model_constructor)
```

---

## Next Steps

1. **Verify OneFlip model generation**
   ```bash
   python inject_backdoor.py -dataset CIFAR10 -quant_bits 8 -max_candidates 100
   ```

2. **Test adapter on generated model**
   ```bash
   python oneflip_adapter.py -model_path ... -output_dir ./test
   ```

3. **Run full pipeline**
   ```bash
   python run_oneflip_pipeline.py -oneflip_model ...
   ```

4. **Analyze results** in `bitshield/results/oneflip_*/`

---

## Performance Tips

- **Reduce candidates**: In OneFlip, use `-max_candidates 50` for faster injection
- **Smaller models**: Start with CIFAR10 (32×32) before ImageNet (224×224)
- **Trial count**: Use `-num_trials 5` for testing, increase for final runs
- **Parallel runs**: Run multiple pipeline instances with different datasets

---

## References

- **OneFlip Paper**: "Rowhammer-Based Trojan Injection: One Bit Flip is Sufficient for Backdoor Injection in DNNs"
- **BitShield Paper**: "BitShield: Defending Against Bit-Flip Attacks on DNN Executables" (NDSS 2025)
- **TVM Quantization**: https://tvm.apache.org/docs/how_to/work_with_quantized_models.html

