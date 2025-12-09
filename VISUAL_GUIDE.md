# OneFlip → BitShield Integration: Visual Guide

## 🗺️ Architecture Diagram

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ONEFLIP → BITSHIELD PIPELINE                    │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: MODEL GENERATION (OneFlip)
═════════════════════════════════════════════════════════════════════
  ┌─────────────────────┐
  │  train_clean_model  │
  │   -dataset CIFAR10  │
  │   -epochs 200       │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────────┐
  │   inject_backdoor.py            │
  │   -dataset CIFAR10              │
  │   -quant_bits 8                 │  ◄── CRITICAL: Quantization
  │   -trigger_epochs 500           │
  └──────────┬──────────────────────┘
             │
             ▼
  ┌─────────────────────────────────┐
  │  convert_and_save_checkpoint    │
  │                                 │
  │  Output: model_int8_state.pth   │
  └──────────┬──────────────────────┘
             │
             ▼
         (.pth file)
         [SAVED MODEL]


PHASE 2: INTEGRATION & CONVERSION (BitShield)
═════════════════════════════════════════════════════════════════════
  ┌──────────────────────────────────────┐
  │  run_oneflip_pipeline.py             │
  │                                      │
  │  Step 1: oneflip_adapter.py          │
  └──────────┬───────────────────────────┘
             │
             ├─► Load .pth checkpoint
             │
             ├─► Extract quantization metadata
             │   • quant_bits: 8
             │   • quant_scale: float
             │   • quant_zero_point: int
             │
             ├─► Convert to ONNX format
             │   Output: resnet_CIFAR10_quantized.onnx
             │
             └─► Create BitShield config JSON
                 Output: resnet_CIFAR10_quantized.json


PHASE 3: COMPILATION (BitShield Build)
═════════════════════════════════════════════════════════════════════
  ┌──────────────────────────────────────┐
  │  buildmodels.py                      │
  │                                      │
  │  • Parse ONNX with quantization ops │
  │  • Apply QNN pre-legalization       │
  │  • Compile with TVM                 │
  │  • Generate native binary (.so)     │
  └──────────┬───────────────────────────┘
             │
             ▼
         (.so file)
     [COMPILED BINARY]


PHASE 4: ATTACK SIMULATION (BitShield Test)
═════════════════════════════════════════════════════════════════════
  ┌──────────────────────────────────────┐
  │  attacksim.py                        │
  │                                      │
  │  • Load compiled binary              │
  │  • Generate bit-flip templates       │
  │  • Simulate attacks                  │
  │  • Analyze defense detection         │
  └──────────┬───────────────────────────┘
             │
             ▼
    (Results Collection)
    ├─ attack_results.pkl
    ├─ pipeline.log
    ├─ analysis.json
    └─ integration_metadata.json

```

---

## 🎯 Component Interaction

```
┌─────────────────┐
│   OneFlip       │
│   (.pth model)  │
└────────┬────────┘
         │
         │ (File Path)
         │
         ▼
┌─────────────────────────────────┐
│   oneflip_adapter.py            │
│                                 │
│   OneFlipQuantizedAdapter       │
│   • load_model()                │
│   • get_quantization_metadata() │
│   • export_to_onnx()            │
│   • create_bitshield_config()   │
└─────────┬───────────────────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌───────────┐
│ONNX   │  │Config.json│
└────┬───┘  └─────┬─────┘
     │            │
     │            ▼
     │        BitShield
     │        Metadata
     │
     ▼
┌─────────────────────────────────┐
│   run_oneflip_pipeline.py       │
│                                 │
│   OneFlipBitShieldPipeline      │
│   • step_integrate()            │
│   • step_compile_binary()       │
│   • step_attack_simulation()    │
│   • step_analysis()             │
└─────────────────┬───────────────┘
                  │
                  ▼
            Results Directory
            results/oneflip_*/
```

---

## 📊 Data Structure Diagram

### State Dict Format
```
.pth checkpoint
    │
    ├─► Direct state dict
    │   {
    │     'layer1.weight': tensor,
    │     'layer1.bias': tensor,
    │     ...
    │   }
    │
    ├─► Wrapped in 'state_dict'
    │   {'state_dict': {...}}
    │
    ├─► Wrapped in 'model'
    │   {'model': {...}}
    │
    └─► Other keys
        {'net': {...}}
```

### Quantization Metadata
```
{
  'model_arch': 'resnet',
  'dataset': 'CIFAR10',
  'quantized': True,
  'quant_bits': 8,              ◄── INT8 or INT4
  'quant_scale': 0.1234,        ◄── Scale factor
  'quant_zero_point': 128,      ◄── Zero point
  'quant_flip_bit': 0,          ◄── Bit index for backdoor
}
```

### Config JSON
```json
{
  "model_name": "QResnet_CIFAR10",
  "dataset": "CIFAR10",
  "quantized": true,
  "quant_bits": 8,
  "input_shape": [1, 3, 32, 32],
  "num_classes": 10,
  "source": "oneflip",
  "original_checkpoint": "path/to/model.pth"
}
```

---

## 🔄 Execution Flow

### Main Pipeline Execution

```
python run_oneflip_pipeline.py
    │
    ├─► Parse arguments
    │
    ├─► Create OneFlipBitShieldPipeline
    │
    ├─► STEP 1: Integration
    │   ├─ oneflip_adapter.integrate_oneflip_to_bitshield()
    │   ├─ Save to: oneflip_onnx_exports/
    │   ├─ Save to: oneflip_configs/
    │   └─ Save: integration_metadata.json
    │
    ├─► STEP 2: Compilation
    │   ├─ Call: buildmodels.py
    │   ├─ Input: ONNX file
    │   ├─ Output: .so binary
    │   └─ Status: logged
    │
    ├─► STEP 3: Attack Simulation
    │   ├─ Call: attacksim.py
    │   ├─ Input: binary + bit-flip templates
    │   ├─ Output: attack_results.pkl
    │   └─ Status: logged
    │
    ├─► STEP 4: Analysis
    │   ├─ Collect results
    │   ├─ Generate report
    │   ├─ Save: analysis.json
    │   └─ Status: logged
    │
    └─► Print results directory
```

---

## 📁 Directory Structure

### Before Integration
```
bitshield/
├── buildmodels.py
├── attacksim.py
├── modman.py
└── ...
```

### After Copying Files
```
bitshield/
├── oneflip_adapter.py              ◄── COPIED
├── run_oneflip_pipeline.py         ◄── COPIED
├── buildmodels.py
├── attacksim.py
└── ...
```

### After Running Pipeline
```
bitshield/
├── oneflip_adapter.py
├── run_oneflip_pipeline.py
├── oneflip_onnx_exports/           ◄── CREATED
│   └── resnet_CIFAR10_quantized.onnx
├── oneflip_configs/                ◄── CREATED
│   └── resnet_CIFAR10_quantized.json
├── results/                        ◄── CREATED
│   └── oneflip_20250108_120000/
│       ├── pipeline.log
│       ├── integration_metadata.json
│       ├── analysis.json
│       └── attack_results.pkl
└── ...
```

---

## 🔄 Quantization Flow

### INT8 Quantization Path

```
OneFlip Model (FP32)
    │
    ├─► Identify key weights
    │
    ├─► Calculate scale = max(|weights|) / 127
    │
    ├─► Quantize: Q = round(weights / scale)
    │   └─ Result: INT8 [-128, 127]
    │
    ├─► Store metadata
    │   ├─ quant_bits: 8
    │   ├─ quant_scale: float
    │   └─ quant_zero_point: 0 or offset
    │
    └─► Inject backdoor via bit flip
        └─ Flip specific bits to trigger backdoor
```

### Export to ONNX

```
Quantized Model (PyTorch)
    │
    ├─► Load state dict
    │
    ├─► Construct model
    │
    ├─► torch.onnx.export()
    │   ├─ opset_version: 11
    │   ├─ quantization ops preserved
    │   └─ Output: ONNX with QNN ops
    │
    └─► Save to file
```

---

## 🎨 Configuration Templates

### Supported Input Shapes

```
Dataset  → Input Shape    (channels, height, width)
────────────────────────────────────────────────
CIFAR10  → (1, 3, 32, 32)
CIFAR100 → (1, 3, 32, 32)
ImageNet → (1, 3, 224, 224)
GTSRB    → (1, 3, 32, 32)
STL10    → (1, 3, 96, 96)
```

### Supported Output Classes

```
Dataset  → Num Classes
────────────────────────
CIFAR10  → 10
CIFAR100 → 100
ImageNet → 1000
GTSRB    → 43
STL10    → 10
```

---

## ⏱️ Timeline

### Typical Execution

```
Time(min)  Activity                Duration
─────────────────────────────────────────────
0-1        Start pipeline           1 min
1-3        Integration              2 min
           • Load model
           • Export ONNX
           • Create config
3-4        Compilation              1-60 min
           • Parse ONNX             (varies)
           • Compile to binary
4-24       Attack Simulation        20 min
           • Load binary            (typical)
           • Simulate bit-flips
           • Collect results
24-25      Analysis                 1 min
           • Summarize results
           • Generate report
25+        Done!

Total: 5-100 minutes (typical: 25 min)
```

---

## 🔐 Security Properties

### What's Tested

```
┌─ Backdoor Properties
│  ├─ Successful injection (did it work?)
│  ├─ Trigger success rate
│  └─ Accuracy impact
│
├─ Quantization Effects
│  ├─ INT4 vs INT8 vulnerabilities
│  ├─ Quantization parameter sensitivity
│  └─ Bit-flip effectiveness
│
├─ Defense Mechanisms
│  ├─ Detection rate
│  ├─ False positives
│  └─ Overhead
│
└─ Robustness
   ├─ Random vs targeted attacks
   ├─ Single vs multiple flips
   └─ Cumulative impact
```

---

## 📈 Metrics Collected

```
Attack Simulation Results
├─ Detection Rate
│  ├─ True Positive Rate (detected backdoors)
│  ├─ False Positive Rate (false alarms)
│  └─ True Negative Rate (legitimate models)
│
├─ Attack Success Rate
│  ├─ Backdoor activation success
│  ├─ Model accuracy drop
│  └─ Trigger effectiveness
│
├─ Defense Effectiveness
│  ├─ Detection accuracy
│  ├─ Response time
│  └─ Overhead impact
│
└─ Quantization Impact
   ├─ INT4 vs INT8 differences
   ├─ Scale sensitivity
   └─ Bit-width dependent vulnerabilities
```

---

## 🚀 Optimization Paths

```
Large Model / Dataset?
    │
    ├─► Reduce -num_flips (default: 1000)
    │   └─ Try: 100-500
    │
    ├─► Reduce -num_trials (default: 10)
    │   └─ Try: 5
    │
    ├─► Use CIFAR10 instead of ImageNet
    │   └─ Reduces by 10-20x
    │
    └─► Use ResNet18 instead of VGG16
        └─ Reduces by 2-5x
```

---

## ✨ Success Indicators

### Green Flags ✅
```
✓ ONNX file created (>100 MB with weights)
✓ Config JSON created with valid structure
✓ Binary compilation completes without errors
✓ Attack simulation produces results
✓ Analysis JSON has complete data
✓ No CUDA out of memory errors
✓ Execution completes within timeout
```

### Warning Signs ⚠️
```
⚠ ONNX file <10 MB (likely no weights)
⚠ Config missing quantization fields
⚠ Partial compilation completed
⚠ Missing attack_results.pkl
⚠ pipeline.log has warnings
⚠ High memory usage during compile
```

### Error Conditions ❌
```
✗ Model file not found
✗ Import errors (torch, torchvision)
✗ ONNX export fails
✗ Binary compilation times out
✗ Attack simulation crashes
✗ Disk space exhausted
✗ GPU out of memory
```

---

## 🎓 Learning Progression

```
Level 1: Basic Usage
├─ Read QUICK_START.md
├─ Copy 2 files
├─ Run pipeline
└─ View results

Level 2: Understanding
├─ Read INTEGRATION_SUMMARY.md
├─ Understand pipeline phases
├─ Review generated files
└─ Study log output

Level 3: Advanced
├─ Read API_REFERENCE.md
├─ Study source code
├─ Experiment with parameters
└─ Create custom scripts

Level 4: Expert
├─ Modify adapter for custom models
├─ Extend pipeline phases
├─ Add custom analysis
└─ Contribute improvements
```

---

## 📝 Quick Reference Card

```
COMMAND PATTERNS
════════════════════════════════════════════

1. Basic Export
   python oneflip_adapter.py -model_path model.pth -output_dir ./exports

2. Full Pipeline
   python run_oneflip_pipeline.py -oneflip_model model.pth -bitshield_dir .

3. Custom Parameters
   python run_oneflip_pipeline.py \
     -oneflip_model model.pth \
     -bitshield_dir . \
     -dataset CIFAR100 \
     -num_flips 5000

4. Quick Test
   python run_oneflip_pipeline.py \
     -oneflip_model model.pth \
     -bitshield_dir . \
     -num_flips 100 \
     -num_trials 5

KEY FILES
════════════════════════════════════════════
Input:  model.pth (OneFlip quantized)
Output: results/oneflip_*/
├── pipeline.log (execution trace)
├── integration_metadata.json (ONNX path)
├── analysis.json (results summary)
└── attack_results.pkl (raw data)

LOCATIONS
════════════════════════════════════════════
ONNX Export:    oneflip_onnx_exports/
Config Files:   oneflip_configs/
Results:        results/oneflip_TIMESTAMP/
```

---

You're all set! 🎉 Use this visual guide alongside the text documentation for quick reference.

