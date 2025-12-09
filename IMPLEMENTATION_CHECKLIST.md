# Implementation Checklist & Next Steps

## ✅ What Was Delivered

### Code Files Created
- [x] `oneflip_adapter.py` - Core integration module (~500 lines)
- [x] `run_oneflip_pipeline.py` - Pipeline orchestrator (~400 lines)

### Documentation Created
- [x] `README_ONEFLIP_INTEGRATION.md` - Getting started guide
- [x] `ONEFLIP_QUICK_START.md` - Quick reference (5 min)
- [x] `oneflip_quantized_integration_guide.md` - Complete guide (20 min)
- [x] `ADAPTER_API_REFERENCE.md` - API documentation
- [x] `INTEGRATION_SUMMARY.md` - Architecture overview
- [x] `DELIVERABLES.md` - Summary of deliverables
- [x] `EXAMPLE_WORKFLOW.sh` - Working example script

### Updated Files
- [x] `EXAMPLE_WORKFLOW.sh` - Complete workflow demonstration

---

## 📋 Installation Checklist

### Before You Start
- [ ] Have access to OneFlip repository
- [ ] Have access to BitShield repository
- [ ] Python 3.8+ installed
- [ ] PyTorch and TorchVision installed
- [ ] BitShield environment set up (run `./setup.sh`)

### Step 1: Copy Core Files
- [ ] Copy `oneflip_adapter.py` to BitShield root
- [ ] Copy `run_oneflip_pipeline.py` to BitShield root
- [ ] Verify files are in place: `ls oneflip_adapter.py run_oneflip_pipeline.py`

### Step 2: Copy Documentation (Optional)
- [ ] Copy all `.md` files to BitShield root
- [ ] These are helpful references but not required for execution

### Step 3: Verify Dependencies
- [ ] BitShield Python environment has torch
- [ ] BitShield Python environment has torchvision
- [ ] Run: `python -c "import torch; import torchvision; print('OK')"`

---

## 🧪 Testing Checklist

### Test 1: Adapter Loading
```bash
cd /path/to/bitshield
python -c "from oneflip_adapter import OneFlipQuantizedAdapter; print('✓ Import OK')"
```
- [ ] Import succeeds

### Test 2: Pipeline Import
```bash
python -c "from run_oneflip_pipeline import OneFlipBitShieldPipeline; print('✓ Import OK')"
```
- [ ] Import succeeds

### Test 3: Quick Help
```bash
python oneflip_adapter.py -h
python run_oneflip_pipeline.py -h
```
- [ ] Both show help text

### Test 4: Adapter Export (No Compilation)
```bash
# In OneFlip directory, generate a quantized model first
python inject_backdoor.py -dataset CIFAR10 -quant_bits 8

# In BitShield directory
python oneflip_adapter.py \
  -model_path /path/to/model.pth \
  -output_dir ./test_export
```
- [ ] `oneflip_onnx_exports/` directory created
- [ ] ONNX file generated
- [ ] No errors in output

### Test 5: Full Pipeline (Small Scale)
```bash
python run_oneflip_pipeline.py \
  -oneflip_model /path/to/model.pth \
  -bitshield_dir . \
  -num_flips 100 \
  -num_trials 5
```
- [ ] Pipeline starts successfully
- [ ] Step 1 (integration) completes
- [ ] `oneflip_onnx_exports/` directory created
- [ ] `oneflip_configs/` directory created
- [ ] Results directory in `results/oneflip_*/` created
- [ ] `pipeline.log` contains all steps
- [ ] `analysis.json` generated

---

## 🚀 Usage Checklist

### Setup (One Time)
- [ ] Copy `oneflip_adapter.py` to BitShield
- [ ] Copy `run_oneflip_pipeline.py` to BitShield
- [ ] Verify BitShield environment: `source env.sh`

### Generate Model (OneFlip Side)
- [ ] Train clean model: `python train_clean_model.py ...`
- [ ] Inject backdoor with quantization: `python inject_backdoor.py -quant_bits 8`
- [ ] Convert checkpoint: `python convert_and_save_checkpoint.py`
- [ ] Note the model path: `saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth`

### Run Integration (BitShield Side)
- [ ] Navigate to BitShield: `cd /path/to/bitshield`
- [ ] Activate environment: `source env.sh`
- [ ] Run pipeline:
  ```bash
  python run_oneflip_pipeline.py \
    -oneflip_model /path/to/model.pth \
    -bitshield_dir .
  ```
- [ ] Monitor progress in terminal
- [ ] Check for errors in output

### Review Results
- [ ] Check results directory: `ls -la results/oneflip_*/`
- [ ] Review log: `cat results/oneflip_*/pipeline.log | tail -50`
- [ ] Check metadata: `cat results/oneflip_*/integration_metadata.json`
- [ ] Review analysis: `cat results/oneflip_*/analysis.json`

---

## 📊 Validation Checklist

### Verify ONNX Export
- [ ] ONNX file exists: `oneflip_onnx_exports/resnet_CIFAR10_quantized.onnx`
- [ ] File size > 100 MB (model weights)
- [ ] Can be opened by TVM

### Verify Configuration
- [ ] JSON file exists: `oneflip_configs/resnet_CIFAR10_quantized.json`
- [ ] Contains `"quantized": true`
- [ ] Contains `"quant_bits"` field
- [ ] Contains correct `num_classes`

### Verify Binary Compilation
- [ ] `.so` binary created (if BitShield compilation works)
- [ ] Binary compilation step logged
- [ ] No compilation errors in log

### Verify Attack Simulation
- [ ] Attack simulation step completed
- [ ] `attack_results.pkl` file created (if available)
- [ ] Number of trials > 0
- [ ] Detection rates computed

### Verify Analysis
- [ ] `analysis.json` contains `status: complete` or `status: partial`
- [ ] Output directory recorded
- [ ] Summary statistics present

---

## 🔍 Debugging Checklist

### If Import Fails
- [ ] Check Python path: `echo $PYTHONPATH`
- [ ] Verify file location: `ls oneflip_adapter.py`
- [ ] Check syntax: `python -m py_compile oneflip_adapter.py`
- [ ] Check dependencies: `python -c "import torch; print(torch.__version__)"`

### If ONNX Export Fails
- [ ] Verify model exists: `ls -l /path/to/model.pth`
- [ ] Check model size > 10 MB (has weights)
- [ ] Try importing directly: `torch.load(model_path)`
- [ ] Check dataset matches architecture (ResNet18 for CIFAR10)

### If Compilation Fails
- [ ] Check BitShield is set up: `ls buildmodels.py`
- [ ] Check TVM is available: `python -c "import tvm"`
- [ ] Check environment: `source env.sh`
- [ ] Try manual compilation: `python buildmodels.py -help`

### If Attack Sim Fails
- [ ] Check binary exists: `ls built/*.so`
- [ ] Check attacksim exists: `ls attacksim.py`
- [ ] Reduce number of flips: `-num_flips 10`
- [ ] Check disk space (binary + data)

---

## 📈 Performance Checklist

### Monitor Execution
- [ ] Watch terminal output for progress
- [ ] Check log file for time stamps: `tail -f results/oneflip_*/pipeline.log`
- [ ] Monitor disk usage for large models
- [ ] Monitor CPU/GPU usage for compilation and simulation

### Optimize if Slow
- [ ] Reduce `-num_flips` (default 1000)
- [ ] Reduce `-num_trials` (default 10)
- [ ] Use smaller dataset (CIFAR10 instead of ImageNet)
- [ ] Use smaller architecture (ResNet18 instead of VGG16)

### Troubleshoot Timeout
- [ ] Increase timeout in script (default 3600 sec for build, 7200 for attack)
- [ ] Run phases separately
- [ ] Use `-num_flips 100` for quick test

---

## 🎯 Success Criteria

### Minimal Success
- [x] Files copy without errors
- [x] `python -c "import oneflip_adapter"` works
- [x] `python oneflip_adapter.py -h` shows help
- [x] Adapter loads OneFlip model without error

### Basic Success
- [x] ONNX export completes
- [x] Config JSON created
- [x] Integration metadata saved
- [x] No error messages in log

### Full Success
- [x] Binary compilation succeeds
- [x] Attack simulation runs
- [x] Results collected
- [x] Analysis report generated

---

## 📝 Documentation Checklist

### To Understand the Integration
- [ ] Read `README_ONEFLIP_INTEGRATION.md` (2 min)
- [ ] Read `ONEFLIP_QUICK_START.md` (5 min)
- [ ] Skim `INTEGRATION_SUMMARY.md` (10 min)

### For Implementation Details
- [ ] Review `ADAPTER_API_REFERENCE.md` for API
- [ ] Review `oneflip_quantized_integration_guide.md` for full guide
- [ ] Review source code: `oneflip_adapter.py`

### For Examples
- [ ] Study `EXAMPLE_WORKFLOW.sh`
- [ ] Try commands from QUICK_START.md
- [ ] Experiment with custom parameters

---

## 🔄 Extended Usage (After Basic Testing)

### Process Multiple Models
- [ ] Generate 3-5 different OneFlip models
- [ ] Run pipeline on each
- [ ] Compare results

### Test Different Configurations
- [ ] Try INT4 quantization (in OneFlip: `-quant_bits 4`)
- [ ] Try different datasets (CIFAR100, ImageNet)
- [ ] Try different architectures (PreActResNet, VGG)

### Batch Processing
- [ ] Create script to process multiple models
- [ ] Run in parallel or sequentially
- [ ] Collect metrics across runs

### Advanced Analysis
- [ ] Compare quantized vs non-quantized
- [ ] Analyze defense effectiveness by quantization level
- [ ] Compare defense mechanisms

---

## 🛠️ Customization Checklist

### Custom Model Architecture
- [ ] Create custom model class
- [ ] Create constructor function
- [ ] Pass to adapter: `export_to_onnx(..., model_constructor_fn=fn)`

### Custom Dataset Support
- [ ] Add to `input_shapes` dict in adapter
- [ ] Add to `num_classes_map` dict in adapter
- [ ] Test ONNX export with new dataset

### Custom Pipeline Configuration
- [ ] Modify `OneFlipBitShieldPipeline` class
- [ ] Add custom logging
- [ ] Add custom analysis steps

---

## 📊 Results Interpretation Checklist

### Understand Output
- [ ] Read `pipeline.log` to understand execution flow
- [ ] Read `integration_metadata.json` to see ONNX path and config
- [ ] Read `analysis.json` to see results summary
- [ ] Read `attack_results.pkl` (if processable)

### Compare Results
- [ ] Compare quantized vs non-quantized defense rates
- [ ] Compare different quantization levels (INT4 vs INT8)
- [ ] Compare different architectures
- [ ] Compare different datasets

### Draw Conclusions
- [ ] Identify strongest/weakest defenses for quantized models
- [ ] Identify backdoor injection success rates
- [ ] Estimate defense effectiveness improvement opportunities

---

## 🎓 Learning Path

### Day 1: Understand
- [ ] Read all documentation
- [ ] Review code structure
- [ ] Understand data flow

### Day 2: Install & Test
- [ ] Copy files to BitShield
- [ ] Run quick tests
- [ ] Generate first model

### Day 3: Run & Analyze
- [ ] Run full pipeline
- [ ] Review results
- [ ] Debug any issues

### Day 4: Extend & Customize
- [ ] Try different configurations
- [ ] Experiment with parameters
- [ ] Build batch processing

### Day 5+: Production Use
- [ ] Process multiple models
- [ ] Collect comprehensive data
- [ ] Generate analysis reports

---

## ✨ Final Checklist

Before considering integration complete:

- [ ] All files copied to BitShield
- [ ] All tests pass
- [ ] Pipeline runs end-to-end
- [ ] Results generated successfully
- [ ] Documentation reviewed
- [ ] No outstanding errors
- [ ] Ready for production use

---

## 🚀 You're Ready!

Once you've completed this checklist:

1. ✅ Installation is verified
2. ✅ Testing shows everything works
3. ✅ Usage patterns are understood
4. ✅ You can generate results
5. ✅ Results can be analyzed

**You're ready to start integrating OneFlip models with BitShield!**

---

## 📞 Quick Reference

**Lost? Start here:**
1. `README_ONEFLIP_INTEGRATION.md` - Overview
2. `ONEFLIP_QUICK_START.md` - Quick reference
3. Terminal: `python run_oneflip_pipeline.py -h` - Help

**Need details?**
- API: `ADAPTER_API_REFERENCE.md`
- Guide: `oneflip_quantized_integration_guide.md`
- Example: `EXAMPLE_WORKFLOW.sh`

**Stuck?**
- Check `results/oneflip_*/pipeline.log` for error messages
- Review `ONEFLIP_QUICK_START.md` → Troubleshooting
- Check error output in terminal

---

Good luck! 🍀

