#!/bin/bash
# Example Workflow: OneFlip → BitShield Integration
# This script demonstrates the complete integration process

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

ONEFLIP_ROOT="/path/to/OneFlip-main/OneFlip-main"
BITSHIELD_ROOT="/path/to/bitshield"
DATASET="CIFAR10"
ARCH="resnet"
QUANT_BITS=8

# ============================================================================
# PHASE 1: Generate Quantized Backdoored Model (OneFlip)
# ============================================================================

echo "============================================================================"
echo "PHASE 1: OneFlip - Generate Quantized Backdoored Model"
echo "============================================================================"

cd "$ONEFLIP_ROOT"

# Step 1a: Train clean model (skip if already exists)
if [ ! -f "saved_model/resnet_CIFAR10/clean_model_1.pth" ]; then
    echo "[1a] Training clean model..."
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
else
    echo "[1a] Clean model already exists, skipping..."
fi

# Step 1b: Inject backdoor with quantization
echo "[1b] Injecting backdoor with quantization..."
python inject_backdoor.py \
    -dataset CIFAR10 \
    -backbone resnet \
    -device 0 \
    -quant_bits $QUANT_BITS \
    -quant_flip_bit 0 \
    -max_candidates 100 \
    -trigger_epochs 500

# Step 1c: Convert checkpoint to standard format
echo "[1c] Converting checkpoint..."
python convert_and_save_checkpoint.py

# Check that quantized model exists
ONEFLIP_MODEL="$ONEFLIP_ROOT/saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth"
if [ ! -f "$ONEFLIP_MODEL" ]; then
    echo "ERROR: Quantized model not found at $ONEFLIP_MODEL"
    exit 1
fi

echo "✓ OneFlip phase complete"
echo "  Quantized model: $ONEFLIP_MODEL"

# ============================================================================
# PHASE 2: Integrate with BitShield
# ============================================================================

echo ""
echo "============================================================================"
echo "PHASE 2: BitShield - Integration & Analysis"
echo "============================================================================"

cd "$BITSHIELD_ROOT"

# Source BitShield environment
if [ -f "env.sh" ]; then
    echo "[2a] Setting up BitShield environment..."
    source env.sh
fi

# Step 2a: Export to ONNX using adapter
echo "[2b] Exporting OneFlip model to ONNX..."
python oneflip_adapter.py \
    -model_path "$ONEFLIP_MODEL" \
    -bitshield_dir "$BITSHIELD_ROOT" \
    -dataset CIFAR10 \
    -arch resnet

# Step 2b: Run full integration pipeline
echo "[2c] Running integration pipeline..."
python run_oneflip_pipeline.py \
    -oneflip_model "$ONEFLIP_MODEL" \
    -bitshield_dir "$BITSHIELD_ROOT" \
    -dataset CIFAR10 \
    -arch resnet \
    -num_flips 1000 \
    -num_trials 10

# ============================================================================
# PHASE 3: Analysis
# ============================================================================

echo ""
echo "============================================================================"
echo "PHASE 3: Results Analysis"
echo "============================================================================"

# Find the most recent results directory
RESULTS_DIR=$(ls -dt "$BITSHIELD_ROOT/results/oneflip_"* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: No results directory found"
    exit 1
fi

echo "Results location: $RESULTS_DIR"
echo ""

# Display summary
if [ -f "$RESULTS_DIR/analysis.json" ]; then
    echo "Final Analysis:"
    cat "$RESULTS_DIR/analysis.json"
fi

if [ -f "$RESULTS_DIR/pipeline.log" ]; then
    echo ""
    echo "Full Log:"
    tail -50 "$RESULTS_DIR/pipeline.log"
fi

echo ""
echo "============================================================================"
echo "COMPLETE! Results saved to: $RESULTS_DIR"
echo "============================================================================"

# ============================================================================
# OPTIONAL: Extended Analysis
# ============================================================================

# If you want to run extended analysis, use this:
echo ""
echo "Optional: Extended Analysis Commands"
echo "======================================"
echo ""
echo "# View detailed metrics"
echo "python tools/get_attacksim_stats.py $RESULTS_DIR/attack_results.pkl"
echo ""
echo "# Generate plots"
echo "python tools/sweeps2csv.py $RESULTS_DIR/attack_results.pkl > $RESULTS_DIR/metrics.csv"
echo ""
echo "# Compare with baseline (non-quantized) models"
echo "python compare_defense_effectiveness.py \\"
echo "  -baseline bitshield/results/standard_models \\"
echo "  -quantized $RESULTS_DIR"
echo ""
