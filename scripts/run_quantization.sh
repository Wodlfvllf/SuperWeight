#!/bin/bash
# Quantization Experiments Pipeline
# Corresponds to official repo's block-wise weight quantization

set -e

# Configuration
MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="./results/quantization"
NUM_BITS=4
BLOCK_SIZES="64 128 256 512 1024"

echo "========================================="
echo "Quantization Experiments Pipeline"
echo "Model: $MODEL_NAME"
echo "Bits: $NUM_BITS"
echo "Block sizes: $BLOCK_SIZES"
echo "========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Detect super weights first (if not already done)
SW_INDEX="./results/super_weights/$(echo $MODEL_NAME | tr '/' '_')_super_weight_index.json"

if [ ! -f "$SW_INDEX" ]; then
    echo "Super weight index not found. Running detection first..."
    bash scripts/run_detection.sh
fi

# Run quantization experiments
echo ""
echo "Running quantization experiments..."
python experiments/quantization_tests.py \
    --model_config ./config/model_config.yaml \
    --model_name "$MODEL_NAME" \
    --super_weight_index "$SW_INDEX" \
    --block_sizes $BLOCK_SIZES \
    --num_bits $NUM_BITS \
    --output_dir "$OUTPUT_DIR" \
    --visualize

echo "✓ Quantization experiments complete"

echo ""
echo "========================================="
echo "Quantization pipeline complete!"
echo "Results in: $OUTPUT_DIR"
echo "========================================="
