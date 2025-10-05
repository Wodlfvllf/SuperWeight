"""
Quantization experiments comparing baseline RTN vs super-outlier-aware quantization.
Corresponds to official repo's block-wise weight quantization scripts.

Official uses: manual_quantize=minmax_4_{blocksize}_no_0_False_False
Ours: Implements same algorithm with cleaner interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm

from src.model_loader import ModelLoader
from src.quantizer import SuperOutlierQuantizer
from src.evaluator import ModelEvaluator
from src.utils import set_seed, load_config, save_results, load_super_weight_index
from src.visualization import SuperWeightVisualizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_quantization_experiment(
    model,
    tokenizer,
    down_proj_layers,
    super_weight_coords,
    config,
    args
):
    """Run comprehensive quantization experiments.
    
    Tests from paper:
    1. Baseline RTN (Round-To-Nearest) at different block sizes
    2. RTN with super weight preservation
    3. RTN with outlier clipping + super weight preservation
    
    Args:
        model: LLM model
        tokenizer: Tokenizer
        down_proj_layers: Dict of down projection layers
        super_weight_coords: List of (layer, row, col) tuples
        config: Configuration dict
        args: Command line arguments
    
    Returns:
        Comprehensive results dictionary
    """
    quantizer = SuperOutlierQuantizer(config)
    evaluator = ModelEvaluator(model, tokenizer, config)
    
    # Save original model state
    original_state = {}
    for layer_idx, module in down_proj_layers.items():
        original_state[layer_idx] = module.weight.data.clone()
    
    results = {
        'block_sizes': args.block_sizes,
        'methods': {
            'baseline_rtn': [],
            'rtn_clip_z3': [],
            'rtn_clip_z3_restore_sw': []
        },
        'perplexities': {
            'baseline_rtn': [],
            'rtn_clip_z3': [],
            'rtn_clip_z3_restore_sw': []
        }
    }
    
    # Group super weights by layer
    layer_super_weights = {}
    for layer_idx, row, col in super_weight_coords:
        if layer_idx not in layer_super_weights:
            layer_super_weights[layer_idx] = []
        layer_super_weights[layer_idx].append((row, col))
    
    logger.info(f"Testing {len(args.block_sizes)} block sizes")
    
    for block_size in tqdm(args.block_sizes, desc="Block sizes"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Block size: {block_size}x{block_size}")
        logger.info(f"{'='*60}")
        
        # Method 1: Baseline RTN (no clipping, no super weight preservation)
        logger.info("Method 1: Baseline RTN")
        for layer_idx, module in down_proj_layers.items():
            module.weight.data = original_state[layer_idx].clone()
            
            quant_weight, metadata = quantizer.quantize_weight_tensor(
                module.weight.data,
                num_bits=args.num_bits,
                block_size=(block_size, block_size),
                super_weight_coords=None,  # No preservation
                z_clip=1000.0  # Effectively no clipping
            )
            module.weight.data = quant_weight
        
        method1_results = evaluator.evaluate_all_benchmarks()
        method1_ppl = evaluator.evaluate_perplexity("wikitext2")
        
        results['methods']['baseline_rtn'].append(
            method1_results.get('average_accuracy', 0.0)
        )
        results['perplexities']['baseline_rtn'].append(method1_ppl)
        logger.info(f"Baseline RTN: Acc={method1_results.get('average_accuracy', 0.0):.4f}, PPL={method1_ppl:.2f}")
        
        # Method 2: RTN with z=3 clipping (no super weight preservation)
        logger.info("Method 2: RTN + Clip(z=3)")
        for layer_idx, module in down_proj_layers.items():
            module.weight.data = original_state[layer_idx].clone()
            
            quant_weight, metadata = quantizer.quantize_weight_tensor(
                module.weight.data,
                num_bits=args.num_bits,
                block_size=(block_size, block_size),
                super_weight_coords=None,
                z_clip=3.0  # Clip outliers
            )
            module.weight.data = quant_weight
        
        method2_results = evaluator.evaluate_all_benchmarks()
        method2_ppl = evaluator.evaluate_perplexity("wikitext2")
        
        results['methods']['rtn_clip_z3'].append(
            method2_results.get('average_accuracy', 0.0)
        )
        results['perplexities']['rtn_clip_z3'].append(method2_ppl)
        logger.info(f"RTN + Clip: Acc={method2_results.get('average_accuracy', 0.0):.4f}, PPL={method2_ppl:.2f}")
        
        # Method 3: RTN with clipping + super weight restoration (OURS)
        logger.info("Method 3: RTN + Clip(z=3) + Restore SW")
        for layer_idx, module in down_proj_layers.items():
            module.weight.data = original_state[layer_idx].clone()
            
            sw_coords = layer_super_weights.get(layer_idx, None)
            quant_weight, metadata = quantizer.quantize_weight_tensor(
                module.weight.data,
                num_bits=args.num_bits,
                block_size=(block_size, block_size),
                super_weight_coords=sw_coords,
                z_clip=3.0
            )
            module.weight.data = quant_weight
        
        method3_results = evaluator.evaluate_all_benchmarks()
        method3_ppl = evaluator.evaluate_perplexity("wikitext2")
        
        results['methods']['rtn_clip_z3_restore_sw'].append(
            method3_results.get('average_accuracy', 0.0)
        )
        results['perplexities']['rtn_clip_z3_restore_sw'].append(method3_ppl)
        logger.info(f"Ours: Acc={method3_results.get('average_accuracy', 0.0):.4f}, PPL={method3_ppl:.2f}")
    
    # Restore original weights
    for layer_idx, module in down_proj_layers.items():
        module.weight.data = original_state[layer_idx]
    
    # Compute improvements
    results['improvements'] = []
    for i, block_size in enumerate(args.block_sizes):
        baseline = results['methods']['baseline_rtn'][i]
        ours = results['methods']['rtn_clip_z3_restore_sw'][i]
        improvement = ((ours - baseline) / baseline * 100) if baseline > 0 else 0
        results['improvements'].append(improvement)
        logger.info(f"Block {block_size}: +{improvement:.2f}% improvement")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Quantization experiments")
    parser.add_argument("--model_config", type=str, default="./config/model_config.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--super_weight_index", type=str, required=True)
    parser.add_argument("--block_sizes", type=int, nargs='+',
                       default=[64, 128, 256, 512, 1024],
                       help="Block sizes to test")
    parser.add_argument("--num_bits", type=int, default=4,
                       help="Number of quantization bits")
    parser.add_argument("--output_dir", type=str, default="./results/quantization")
    parser.add_argument("--visualize", action="store_true")
    
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
    
    # Get layers
    down_proj_layers = model_loader.get_down_proj_layers()
    
    # Run experiments
    results = run_quantization_experiment(
        model, tokenizer, down_proj_layers,
        super_weight_coords, config, args
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results(results, args.output_dir, prefix=f"{model_name_clean}_quantization")
    
    # Visualize
    if args.visualize:
        visualizer = SuperWeightVisualizer(output_dir=args.output_dir)
        visualizer.plot_quantization_comparison(
            args.block_sizes,
            results['methods'],
            model_name_clean
        )
    
    logger.info("\nQuantization experiments complete!")


if __name__ == "__main__":
    main()
