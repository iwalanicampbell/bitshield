# OneFlip → BitShield Integration: Getting Started

Welcome! This directory now contains a complete integration solution for pipelining OneFlip's quantized model backdoor injection into BitShield's defense analysis framework.

## 🚀 Quick Start (2 minutes)

### 1. Copy the essential files to your BitShield directory:

```bash
# The two files you absolutely need:
cp oneflip_adapter.py /path/to/bitshield/
cp run_oneflip_pipeline.py /path/to/bitshield/
```

### 2. Generate a quantized model in OneFlip:

```bash
cd /path/to/OneFlip-main/OneFlip-main
python inject_backdoor.py -dataset CIFAR10 -quant_bits 8
```

### 3. Run the pipeline:

```bash
cd /path/to/bitshield
python run_oneflip_pipeline.py \
  -oneflip_model /path/to/model.pth \
  -bitshield_dir .
```

Done! Results are in `results/oneflip_*/`

---

## 📚 Documentation Guide

**Choose your starting point:**

| Your Need | Document | Time |
|-----------|----------|------|
| 30-second overview | **ONEFLIP_QUICK_START.md** | 2 min |
| Get it working | **INTEGRATION_SUMMARY.md** | 10 min |
| Detailed walkthrough | **oneflip_quantized_integration_guide.md** | 20 min |
| Python API docs | **ADAPTER_API_REFERENCE.md** | Reference |
| Working example | **EXAMPLE_WORKFLOW.sh** | 5 min |
| What was delivered | **DELIVERABLES.md** | 5 min |

---

## 🔧 What Was Created

### Code Files (Copy to BitShield)

1. **`oneflip_adapter.py`**
   - Loads OneFlip quantized models
   - Exports to ONNX format
   - Extracts quantization parameters
   - Creates BitShield configs

2. **`run_oneflip_pipeline.py`**
   - Automated end-to-end pipeline
   - 4-phase execution (integrate → compile → attack → analyze)
   - Detailed logging
   - Results collection

### Documentation Files (Reference)

3. **`ONEFLIP_QUICK_START.md`** - Quick reference (5 min read)
4. **`oneflip_quantized_integration_guide.md`** - Complete guide (20 min read)
5. **`ADAPTER_API_REFERENCE.md`** - API documentation (reference)
6. **`INTEGRATION_SUMMARY.md`** - Architecture overview (10 min read)
7. **`EXAMPLE_WORKFLOW.sh`** - Complete workflow example
8. **`DELIVERABLES.md`** - Summary of what was created

---

## 🎯 The Integration Flow

```
Your OneFlip quantized model (.pth)
           ↓
    [oneflip_adapter.py]
    Load + Extract quantization metadata
           ↓
    Export to ONNX format
           ↓
    [run_oneflip_pipeline.py]
    Compile to binary using BitShield
           ↓
    Simulate bit-flip attacks
           ↓
    Analyze defense effectiveness
           ↓
    Results in results/oneflip_*/
```

---

## ✅ What This Solves

**Problem:** How do I test OneFlip's quantized backdoors with BitShield's defenses?

**Solution:** This integration automatically:
- ✅ Loads OneFlip quantized models
- ✅ Converts to BitShield-compatible format
- ✅ Compiles to native binary
- ✅ Runs attack simulations
- ✅ Collects and analyzes results

All in a single command!

---

## 📋 Supported Configurations

### Datasets
CIFAR10, CIFAR100, ImageNet, GTSRB, STL10

### Architectures
ResNet18, PreActResNet18, VGG16, Custom

### Quantization
INT8, INT4, FP32

---

## 🔍 File Locations

After running the pipeline, you'll find:

```
bitshield/
├── oneflip_adapter.py              # Core module (COPY HERE)
├── run_oneflip_pipeline.py          # Pipeline script (COPY HERE)
├── oneflip_onnx_exports/            # Generated ONNX files
│   └── resnet_CIFAR10_quantized.onnx
├── oneflip_configs/                 # Generated configs
│   └── resnet_CIFAR10_quantized.json
└── results/
    └── oneflip_20250108_120000/     # Results (auto-named)
        ├── pipeline.log             # Detailed execution log
        ├── integration_metadata.json # ONNX path + metadata
        ├── analysis.json            # Results summary
        └── attack_results.pkl       # Raw attack data
```

---

## 🐛 Troubleshooting

**Issue:** "Model not found"
```bash
# Use full absolute paths
python run_oneflip_pipeline.py \
  -oneflip_model "C:\full\path\to\model.pth" \
  -bitshield_dir "C:\full\path\to\bitshield"
```

**Issue:** "Import torch failed"
```bash
# Activate BitShield environment
source env.sh  # or setup your Python environment
python run_oneflip_pipeline.py ...
```

**Issue:** "ONNX export mismatch"
```bash
# Check model architecture matches dataset
# ResNet18 for CIFAR10/100, appropriate model for other datasets
python oneflip_adapter.py -model_path model.pth -output_dir ./test
```

More troubleshooting in **ONEFLIP_QUICK_START.md** → Troubleshooting section

---

## 💡 Common Tasks

### Export OneFlip model to ONNX only

```bash
python oneflip_adapter.py \
  -model_path model.pth \
  -output_dir ./exports \
  -dataset CIFAR10
```

### Run pipeline with custom parameters

```bash
python run_oneflip_pipeline.py \
  -oneflip_model model.pth \
  -bitshield_dir . \
  -dataset CIFAR100 \
  -num_flips 5000 \
  -num_trials 50
```

### Batch process multiple models

```bash
for model in saved_model/*/clean_model_*_int8_state.pth; do
  python run_oneflip_pipeline.py -oneflip_model "$model" -bitshield_dir .
done
```

### Use custom model architecture

```python
from oneflip_adapter import OneFlipQuantizedAdapter
from my_models import MyModel

def constructor(num_classes):
    return MyModel(num_classes=num_classes)

adapter = OneFlipQuantizedAdapter("model.pth")
adapter.export_to_onnx("exports", constructor)
```

---

## 📊 Performance

| Step | Time |
|------|------|
| Load OneFlip model | ~1 sec |
| ONNX export | ~3 sec |
| Binary compilation | ~30-90 sec |
| Attack simulation (1000 flips) | ~5-20 min |
| **Total** | **~6-30 min** |

---

## 🔗 Next Steps

1. **Read** `ONEFLIP_QUICK_START.md` (5 min)
2. **Copy** `oneflip_adapter.py` and `run_oneflip_pipeline.py` to BitShield
3. **Generate** quantized model in OneFlip
4. **Run** `python run_oneflip_pipeline.py ...`
5. **Review** results in `results/oneflip_*/`
6. **Analyze** using scripts in `tools/`

---

## 📖 Document Map

```
START HERE
    ↓
ONEFLIP_QUICK_START.md (30-sec summary)
    ↓
INTEGRATION_SUMMARY.md (architecture & overview)
    ↓
oneflip_quantized_integration_guide.md (detailed walkthrough)
    ↓
ADAPTER_API_REFERENCE.md (Python API docs)
    ↓
EXAMPLE_WORKFLOW.sh (working example)
    ↓
DELIVERABLES.md (what was delivered)
```

---

## 🎓 Learning Path

**For Quick Start:**
1. Read this file (2 min)
2. Read `ONEFLIP_QUICK_START.md` (5 min)
3. Copy files and run pipeline (5 min)

**For Deep Understanding:**
1. Read `INTEGRATION_SUMMARY.md` (10 min)
2. Read `oneflip_quantized_integration_guide.md` (20 min)
3. Read `ADAPTER_API_REFERENCE.md` (reference as needed)
4. Study `EXAMPLE_WORKFLOW.sh` (5 min)
5. Run and experiment (30 min)

---

## ❓ FAQ

**Q: Do I need to modify OneFlip?**
A: No! Generate models normally with `-quant_bits 8` parameter.

**Q: Do I need to modify BitShield?**
A: No! This integration uses existing BitShield infrastructure.

**Q: Can I use custom architectures?**
A: Yes! Provide a `model_constructor_fn` parameter.

**Q: What if my dataset isn't listed?**
A: Add it to the `input_shapes` dict in adapter, or customize.

**Q: Can I process multiple models in parallel?**
A: Yes! Run multiple pipeline instances with different output directories.

**Q: Where do results go?**
A: `bitshield/results/oneflip_TIMESTAMP/` with logs and analysis.

---

## 📞 Support

- **Quick answers:** See ONEFLIP_QUICK_START.md → Troubleshooting
- **How-to guides:** See oneflip_quantized_integration_guide.md
- **API details:** See ADAPTER_API_REFERENCE.md
- **Examples:** See EXAMPLE_WORKFLOW.sh

---

## 📄 Files in This Package

```
Core Integration:
  ✓ oneflip_adapter.py                    (~500 lines)
  ✓ run_oneflip_pipeline.py               (~400 lines)

Quick References:
  ✓ ONEFLIP_QUICK_START.md                (this file you're reading)
  ✓ INTEGRATION_SUMMARY.md                (high-level overview)

Comprehensive Guides:
  ✓ oneflip_quantized_integration_guide.md (complete walkthrough)
  ✓ ADAPTER_API_REFERENCE.md              (detailed API docs)

Utilities:
  ✓ EXAMPLE_WORKFLOW.sh                   (working example)
  ✓ DELIVERABLES.md                       (what was created)
```

---

## 🏁 Ready?

### For the impatient:

```bash
# Copy files
cp oneflip_adapter.py /path/to/bitshield/
cp run_oneflip_pipeline.py /path/to/bitshield/

# Run it
cd /path/to/bitshield
python run_oneflip_pipeline.py -oneflip_model /path/to/model.pth -bitshield_dir .

# Check results
ls results/oneflip_*/
```

### For the thorough:

1. Read `ONEFLIP_QUICK_START.md`
2. Read `oneflip_quantized_integration_guide.md`
3. Review `ADAPTER_API_REFERENCE.md`
4. Run `EXAMPLE_WORKFLOW.sh`
5. Start experimenting!

---

**Happy integrating!** 🚀

For questions, check the documentation files or review the log files in `results/oneflip_*/pipeline.log`

