"""Super weight detection implementation based on the paper's algorithm."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .activation_analyzer import ActivationAnalyzer
from .utils import compute_z_score

logger = logging.getLogger(__name__)


class SuperWeightDetector:
    """Detects super weights using data-free single forward pass method."""
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        tokenizer,
        config: Dict
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.analyzer = ActivationAnalyzer(model, config)
        self.super_weight_index = []
        
    def detect_super_weights(
        self,
        down_proj_layers: Dict[int, torch.nn.Module],
        prompt: Optional[str] = None,
        max_iterations: int = 6
    ) -> List[Tuple[int, int, int]]:
        """Detect super weights using Algorithm 1 from the paper.
        
        Algorithm:
        1. Run forward pass with detection prompt
        2. Analyze down_proj input/output activation spikes
        3. Identify super weight coordinates from spike locations
        4. Remove detected super weight and repeat until no more spikes
        
        Args:
            down_proj_layers: Dictionary of layer_idx -> down_proj module
            prompt: Detection prompt (default from config)
            max_iterations: Maximum number of super weights to detect
        
        Returns:
            List of (layer_idx, row, col) tuples for super weight coordinates
        """
        logger.info("Starting super weight detection...")
        
        if prompt is None:
            prompt = self.config['super_weight_detection']['prompt']
        
        detected_super_weights = []
        original_weights = {}
        
        # Save original weights for restoration
        for layer_idx, module in down_proj_layers.items():
            original_weights[layer_idx] = module.weight.data.clone()
        
        for iteration in range(max_iterations):
            logger.info(f"\nIteration {iteration + 1}/{max_iterations}")
            
            # Register hooks for activation capture
            self.analyzer.register_hooks(down_proj_layers)
            
            # Forward pass
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Analyze activations
            num_layers = len(down_proj_layers)
            activation_stats = self.analyzer.analyze_down_proj_activations(num_layers)
            
            # Detect spikes
            threshold_mult = self.config['super_weight_detection']['activation_threshold_multiplier']
            spikes = self.analyzer.detect_activation_spikes(
                activation_stats, 
                threshold_multiplier=threshold_mult
            )
            
            if not spikes:
                logger.info("No more activation spikes detected. Stopping.")
                break
            
            # Find the most prominent spike
            input_spikes = [(idx, activation_stats['input_max'][idx]) 
                           for idx, spike_type in spikes if spike_type == 'input']
            output_spikes = [(idx, activation_stats['output_max'][idx]) 
                            for idx, spike_type in spikes if spike_type == 'output']
            
            if not input_spikes or not output_spikes:
                logger.warning("Incomplete spike pattern detected. Stopping.")
                break
            
            # Paper: Super weight layer is where both input and output spike
            layer_idx = max(set([idx for idx, _ in input_spikes]) & 
                          set([idx for idx, _ in output_spikes]))
            
            # Get super weight coordinates
            # Row from output max channel, col from input max channel
            row = int(activation_stats['output_max_channels'][layer_idx])
            col = int(activation_stats['input_max_channels'][layer_idx])
            
            logger.info(f"Super weight detected at layer {layer_idx}, "
                       f"coordinates ({row}, {col})")
            
            detected_super_weights.append((layer_idx, row, col))
            
            # Zero out the detected super weight for next iteration
            down_proj_layers[layer_idx].weight.data[row, col] = 0.0
            
            # Clear hooks
            self.analyzer.clear_hooks()
        
        # Restore original weights
        for layer_idx, module in down_proj_layers.items():
            module.weight.data = original_weights[layer_idx]
        
        self.super_weight_index = detected_super_weights
        logger.info(f"\nDetection complete. Found {len(detected_super_weights)} super weights:")
        for layer, row, col in detected_super_weights:
            logger.info(f"  Layer {layer}: weight[{row}, {col}]")
        
        return detected_super_weights
    
    def validate_super_weights(
        self,
        super_weight_coords: List[Tuple[int, int, int]],
        down_proj_layers: Dict[int, torch.nn.Module]
    ) -> Dict[str, any]:
        """Validate detected super weights by measuring their magnitude and impact.
        
        Args:
            super_weight_coords: List of (layer, row, col) tuples
            down_proj_layers: Dictionary of down projection layers
        
        Returns:
            Validation statistics
        """
        validation_results = {
            'super_weights': [],
            'magnitudes': [],
            'relative_magnitudes': []
        }
        
        for layer_idx, row, col in super_weight_coords:
            if layer_idx in down_proj_layers:
                weight = down_proj_layers[layer_idx].weight.data
                sw_magnitude = abs(weight[row, col].item())
                
                # Compute relative magnitude compared to layer statistics
                layer_mean = weight.abs().mean().item()
                layer_std = weight.abs().std().item()
                relative_mag = (sw_magnitude - layer_mean) / layer_std
                
                validation_results['super_weights'].append((layer_idx, row, col))
                validation_results['magnitudes'].append(sw_magnitude)
                validation_results['relative_magnitudes'].append(relative_mag)
                
                logger.info(f"Layer {layer_idx} SW: magnitude={sw_magnitude:.4f}, "
                          f"z-score={relative_mag:.2f}")
        
        return validation_results
    
    def prune_super_weights(
        self,
        super_weight_coords: List[Tuple[int, int, int]],
        down_proj_layers: Dict[int, torch.nn.Module]
    ):
        """Zero out super weights (for ablation studies).
        
        Args:
            super_weight_coords: List of super weight coordinates
            down_proj_layers: Dictionary of down projection layers
        """
        for layer_idx, row, col in super_weight_coords:
            if layer_idx in down_proj_layers:
                down_proj_layers[layer_idx].weight.data[row, col] = 0.0
                logger.info(f"Pruned super weight at layer {layer_idx}[{row}, {col}]")
    
    def restore_super_weights(
        self,
        super_weight_coords: List[Tuple[int, int, int]],
        down_proj_layers: Dict[int, torch.nn.Module],
        original_weights: Dict[int, torch.Tensor]
    ):
        """Restore pruned super weights from original values.
        
        Args:
            super_weight_coords: List of super weight coordinates
            down_proj_layers: Dictionary of down projection layers
            original_weights: Dictionary of original weight tensors
        """
        for layer_idx, row, col in super_weight_coords:
            if layer_idx in down_proj_layers and layer_idx in original_weights:
                original_val = original_weights[layer_idx][row, col]
                down_proj_layers[layer_idx].weight.data[row, col] = original_val
                logger.info(f"Restored super weight at layer {layer_idx}[{row}, {col}]")
    
    def scale_super_weights(
        self,
        super_weight_coords: List[Tuple[int, int, int]],
        down_proj_layers: Dict[int, torch.nn.Module],
        scaling_factor: float
    ):
        """Scale super weights by a factor (for sensitivity analysis).
        
        Args:
            super_weight_coords: List of super weight coordinates
            down_proj_layers: Dictionary of down projection layers
            scaling_factor: Multiplicative scaling factor
        """
        for layer_idx, row, col in super_weight_coords:
            if layer_idx in down_proj_layers:
                down_proj_layers[layer_idx].weight.data[row, col] *= scaling_factor
        
        logger.info(f"Scaled {len(super_weight_coords)} super weights by {scaling_factor}x")
    
    def get_super_weight_values(
        self,
        super_weight_coords: List[Tuple[int, int, int]],
        down_proj_layers: Dict[int, torch.nn.Module]
    ) -> List[float]:
        """Get current values of super weights.
        
        Args:
            super_weight_coords: List of super weight coordinates
            down_proj_layers: Dictionary of down projection layers
        
        Returns:
            List of super weight values
        """
        values = []
        for layer_idx, row, col in super_weight_coords:
            if layer_idx in down_proj_layers:
                val = down_proj_layers[layer_idx].weight.data[row, col].item()
                values.append(val)
        return values
