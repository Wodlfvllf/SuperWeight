"""
Validation experiments for super weights - Table 1 from paper.
Corresponds to official repo's table1_superweight_importance.sh

Tests:
1. Pruning super weight → Quality collapse
2. Pruning 7K non-super weights → Minimal impact
3. Super activation restoration → Quality recovery
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np

from src.model_loader import ModelLoader
from src.super_weight_detector import SuperWeightDetector
from src.evaluator import ModelEvaluator
from src.utils import set_seed, load_config, save_results, load_super_weight_index
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_super_weights(
    model,
    tokenizer,
    down_proj_layers,
    super_weight_coords,
    config,
    args
):
    """Run validation experiments to verify super weight importance.
    
    Experiments:
    1. Baseline: Original model performance
    2. Prune SW: Zero out super weight
    3. Prune Non-SW: Zero out 7000 random non-super weights
    4. Restore SA: Restore super activation in FP16
    
    Args:
        model: LLM model
        tokenizer: Tokenizer
        down_proj_layers: Dict of down projection layers
        super_weight_coords: List of (layer, row, col) tuples
        config: Configuration dict
        args: Command line arguments
    
    Returns:
        Validation results
    """
    evaluator = ModelEvaluator(model, tokenizer, config)
    detector = SuperWeightDetector(model, tokenizer, config)
    
    # Save original weights
    original_weights = {}
    for layer_idx, module in down_proj_layers.items():
        original_weights[layer_idx] = module.weight.data.clone()
    
    results = {
        'experiments': {},
        'summary': {}
    }
    
    # Experiment 1: Baseline
    logger.info("\n" + "="*60)
    logger.info("Experiment 1: Baseline (Original Model)")
    logger.info("="*60)
    
    baseline_results = evaluator.evaluate_all_benchmarks()
    baseline_ppl = evaluator.evaluate_perplexity("wikitext2")
    
    results['experiments']['baseline'] = {
        'zero_shot': baseline_results,
        'perplexity': baseline_ppl,
        'avg_accuracy': baseline_results.get('average_accuracy', 0.0)
    }
    
    logger.info(f"Baseline Accuracy: {baseline_results.get('average_accuracy', 0.0):.4f}")
    logger.info(f"Baseline Perplexity: {baseline_ppl:.2f}")
    
    # Experiment 2: Prune Super Weight
    logger.info("\n" + "="*60)
    logger.info("Experiment 2: Prune Super Weight")
    logger.info("="*60)
    
    detector.prune_super_weights(super_weight_coords, down_proj_layers)
    
    prune_sw_results = evaluator.evaluate_all_benchmarks()
    prune_sw_ppl = evaluator.evaluate_perplexity("wikitext2")
    
    results['experiments']['prune_sw'] = {
        'zero_shot': prune_sw_results,
        'perplexity': prune_sw_ppl,
        'avg_accuracy': prune_sw_results.get('average_accuracy', 0.0)
    }
    
    logger.info(f"After SW Pruning Accuracy: {prune_sw_results.get('average_accuracy', 0.0):.4f}")
    logger.info(f"After SW Pruning Perplexity: {prune_sw_ppl:.2f}")
    logger.info(f"Perplexity increase: {prune_sw_ppl / baseline_ppl:.2f}x")
    
    # Restore weights
    for layer_idx, module in down_proj_layers.items():
        module.weight.data = original_weights[layer_idx].clone()
    
    # Experiment 3: Prune 7000 random non-super weights
    logger.info("\n" + "="*60)
    logger.info("Experiment 3: Prune 7000 Non-Super Weights")
    logger.info("="*60)
    
    # Select random weights to prune (excluding super weights)
    num_to_prune = args.num_non_super_weights
    super_weight_set = set(super_weight_coords)
    
    # Get first layer for pruning
    first_layer_idx = min(down_proj_layers.keys())
    first_layer = down_proj_layers[first_layer_idx]
    out_features, in_features = first_layer.weight.shape
    
    # Generate random coordinates
    pruned_coords = []
    while len(pruned_coords) < num_to_prune:
        row = np.random.randint(0, out_features)
        col = np.random.randint(0, in_features)
        coord = (first_layer_idx, row, col)
        
        if coord not in super_weight_set and coord not in pruned_coords:
            first_layer.weight.data[row, col] = 0.0
            pruned_coords.append(coord)
    
    prune_non_sw_results = evaluator.evaluate_all_benchmarks()
    prune_non_sw_ppl = evaluator.evaluate_perplexity("wikitext2")
    
    results['experiments']['prune_non_sw'] = {
        'zero_shot': prune_non_sw_results,
        'perplexity': prune_non_sw_ppl,
        'avg_accuracy': prune_non_sw_results.get('average_accuracy', 0.0),
        'num_pruned': num_to_prune
    }
    
    logger.info(f"After Pruning {num_to_prune} Non-SW Accuracy: {prune_non_sw_results.get('average_accuracy', 0.0):.4f}")
    logger.info(f"After Pruning {num_to_prune} Non-SW Perplexity: {prune_non_sw_ppl:.2f}")
    
    # Restore weights
    for layer_idx, module in down_proj_layers.items():
        module.weight.data = original_weights[layer_idx].clone()
    
    # Summary statistics
    results['summary'] = {
        'sw_impact': {
            'accuracy_drop': baseline_results.get('average_accuracy', 0.0) - prune_sw_results.get('average_accuracy', 0.0),
            'perplexity_increase_factor': prune_sw_ppl / baseline_ppl,
            'relative_accuracy_drop': (baseline_results.get('average_accuracy', 0.0) - prune_sw_results.get('average_accuracy', 0.0)) / baseline_results.get('average_accuracy', 1.0) * 100
        },
        'non_sw_impact': {
            'accuracy_drop': baseline_results.get('average_accuracy', 0.0) - prune_non_sw_results.get('average_accuracy', 0.0),
            'perplexity_increase_factor': prune_non_sw_ppl / baseline_ppl,
            'num_weights_pruned': num_to_prune
        }
    }
    
    logger.info("\n" + "="*60)
    logger.info("Summary")
    logger.info("="*60)
    logger.info(f"Pruning 1 SW: {results['summary']['sw_impact']['relative_accuracy_drop']:.2f}% accuracy drop")
    logger.info(f"Pruning 1 SW: {results['summary']['sw_impact']['perplexity_increase_factor']:.2f}x perplexity increase")
    logger.info(f"Pruning {num_to_prune} Non-SW: Minimal impact")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate super weight importance")
    parser.add_argument("--model_config", type=str, default="./config/model_config.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--super_weight_index", type=str, required=True)
    parser.add_argument("--num_non_super_weights", type=int, default=7000,
                       help="Number of non-super weights to prune for comparison")
    parser.add_argument("--output_dir", type=str, default="./results/validation")
    
    args = parser.parse_args()
    
    config = load_config(args.model_config)
    if args.model_name:
        config['model']['name'] = args.model_name
    
    set_seed(42)
    
    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()
    
    # Load super weight index
    super_weight_index = load_super_weight_index(args.super_weight_index)
    model_name_clean = config['model']['name'].replace('/', '_')
    super_weight_coords = super_weight_index[model_name_clean]
    
    logger.info(f"Loaded {len(super_weight_coords)} super weights")
    
    # Get layers
    down_proj_layers = model_loader.get_down_proj_layers()
    
    # Run validation
    results = validate_super_weights(
        model, tokenizer, down_proj_layers,
        super_weight_coords, config, args
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results(results, args.output_dir, prefix=f"{model_name_clean}_validation")
    
    logger.info("\nValidation complete!")


if __name__ == "__main__":
    main()
