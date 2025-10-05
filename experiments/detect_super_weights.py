"""Main script for detecting super weights in LLMs."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from src.model_loader import ModelLoader
from src.super_weight_detector import SuperWeightDetector
from src.utils import set_seed, load_config, save_super_weight_index
from src.visualization import SuperWeightVisualizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Detect super weights in LLMs")
    parser.add_argument("--model_config", type=str, default="./config/model_config.yaml",
                       help="Path to model configuration")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (overrides config)")
    parser.add_argument("--output_dir", type=str, default="./results/super_weights",
                       help="Output directory")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Detection prompt (uses config default if None)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.model_config)
    
    if args.model_name:
        config['model']['name'] = args.model_name
    
    # Set random seed
    set_seed(config.get('system', {}).get('seed', 42))
    
    logger.info(f"Detecting super weights for model: {config['model']['name']}")
    
    # Load model
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()
    
    # Get down projection layers
    down_proj_layers = model_loader.get_down_proj_layers()
    
    # Initialize detector
    detector = SuperWeightDetector(model, tokenizer, config)
    
    # Detect super weights
    prompt = args.prompt if args.prompt else config['super_weight_detection']['prompt']
    super_weight_coords = detector.detect_super_weights(
        down_proj_layers,
        prompt=prompt
    )
    
    # Validate detected super weights
    validation_results = detector.validate_super_weights(
        super_weight_coords,
        down_proj_layers
    )
    
    # Save super weight index
    model_name_clean = config['model']['name'].replace('/', '_')
    index = {model_name_clean: super_weight_coords}
    
    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, f"{model_name_clean}_super_weight_index.json")
    save_super_weight_index(index, index_path)
    
    # Visualization
    if args.visualize:
        visualizer = SuperWeightVisualizer(output_dir=args.output_dir)
        
        # Run one more forward pass for visualization
        detector.analyzer.register_hooks(down_proj_layers)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
        
        activation_stats = detector.analyzer.analyze_down_proj_activations(len(down_proj_layers))
        visualizer.plot_activation_distribution(activation_stats, model_name_clean)
        
        detector.analyzer.clear_hooks()
    
    logger.info("Super weight detection complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
