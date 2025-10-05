#!/bin/bash
# Super Weight Detection Pipeline
# Corresponds to official repo's figure3_how_to_identify_superweight.sh

set -e

# Configuration
MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="./results/super_weights"
PROMPT="The quick brown fox jumps over the lazy dog"

echo "========================================="
echo "Super Weight Detection Pipeline"
echo "Model: $MODEL_NAME"
echo "========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Step 1: Detect super weights
echo ""
echo "Step 1: Detecting super weights..."
python experiments/detect_super_weights.py \
    --model_config ./config/model_config.yaml \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --prompt "$PROMPT" \
    --visualize

SW_INDEX="$OUTPUT_DIR/$(echo $MODEL_NAME | tr '/' '_')_super_weight_index.json"

if [ ! -f "$SW_INDEX" ]; then
    echo "Error: Super weight index not found!"
    exit 1
fi

echo "✓ Super weight detection complete"
echo "  Index saved to: $SW_INDEX"

# Step 2: Validate super weights
echo ""
echo "Step 2: Validating super weight importance..."
python experiments/validate_super_weights.py \
    --model_config ./config/model_config.yaml \
    --model_name "$MODEL_NAME" \
    --super_weight_index "$SW_INDEX" \
    --output_dir "$OUTPUT_DIR/validation" \
    --num_non_super_weights 7000

echo "✓ Validation complete"

# Step 3: Sensitivity analysis
echo ""
echo "Step 3: Running sensitivity analysis..."
python experiments/sensitivity_analysis.py \
    --model_config ./config/model_config.yaml \
    --model_name "$MODEL_NAME" \
    --super_weight_index "$SW_INDEX" \
    --scaling_factors 0.0 0.2 0.5 0.8 1.0 1.5 2.0 2.5 3.0 \
    --output_dir "$OUTPUT_DIR/sensitivity" \
    --visualize

echo "✓ Sensitivity analysis complete"

echo ""
echo "========================================="
echo "Detection pipeline complete!"
echo "Results in: $OUTPUT_DIR"
echo "========================================="
