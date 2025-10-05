"""Comprehensive benchmark evaluation script."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from src.model_loader import ModelLoader
from src.evaluator import ModelEvaluator
from src.utils import set_seed, load_config, save_results
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive benchmark evaluation")
    parser.add_argument("--model_config", type=str, default="./config/model_config.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs='+', default=None,
                       help="Tasks to evaluate (uses config if None)")
    parser.add_argument("--perplexity_datasets", type=str, nargs='+',
                       default=["wikitext2", "c4"])
    parser.add_argument("--output_dir", type=str, default="./results/benchmarks")
    
    args = parser.parse_args()
    
    config = load_config(args.model_config)
    if args.model_name:
        config['model']['name'] = args.model_name
    
    set_seed(42)
    
    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, tokenizer, config)
    
    results = {
        'model_name': config['model']['name'],
        'zero_shot': {},
        'perplexity': {}
    }
    
    # Zero-shot evaluation
    tasks = args.tasks if args.tasks else config['evaluation']['tasks']
    logger.info(f"\nEvaluating {len(tasks)} zero-shot tasks...")
    results['zero_shot'] = evaluator.evaluate_all_benchmarks(tasks)
    
    # Perplexity evaluation
    for dataset in args.perplexity_datasets:
        logger.info(f"\nEvaluating perplexity on {dataset}...")
        ppl = evaluator.evaluate_perplexity(dataset)
        results['perplexity'][dataset] = ppl
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_name_clean = config['model']['name'].replace('/', '_')
    save_results(results, args.output_dir, prefix=f"{model_name_clean}_benchmarks")
    
    logger.info("\nEvaluation complete!")
    logger.info(f"Average accuracy: {results['zero_shot'].get('average_accuracy', 0.0):.4f}")


if __name__ == "__main__":
    main()
