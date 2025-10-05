"""Activation analysis for detecting super activations and their patterns."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ActivationAnalyzer:
    """Analyzes activation patterns to identify super activations."""
    
    def __init__(self, model: torch.nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.activation_cache = defaultdict(dict)
        self.hooks = []
        
    def register_hooks(self, target_layers: Dict[int, torch.nn.Module]):
        """Register forward hooks on target layers to capture activations.
        
        Args:
            target_layers: Dictionary mapping layer index to module
        """
        self.clear_hooks()
        
        for layer_idx, module in target_layers.items():
            hook = module.register_forward_hook(
                self._create_hook_fn(layer_idx)
            )
            self.hooks.append(hook)
        
        logger.info(f"Registered hooks on {len(target_layers)} layers")
    
    def _create_hook_fn(self, layer_idx: int):
        """Create hook function for specific layer."""
        def hook_fn(module, input, output):
            # Store input and output activations
            if isinstance(input, tuple):
                input_tensor = input[0].detach()
            else:
                input_tensor = input.detach()
            
            if isinstance(output, tuple):
                output_tensor = output[0].detach()
            else:
                output_tensor = output.detach()
            
            self.activation_cache[layer_idx] = {
                'input': input_tensor,
                'output': output_tensor
            }
        
        return hook_fn
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_cache.clear()
    
    def analyze_down_proj_activations(
        self, 
        num_layers: int
    ) -> Dict[str, np.ndarray]:
        """Analyze down projection activations across all layers.
        
        Returns:
            Dictionary with 'input_max' and 'output_max' arrays [num_layers]
        """
        input_max = np.zeros(num_layers)
        output_max = np.zeros(num_layers)
        input_max_channels = np.zeros(num_layers, dtype=int)
        output_max_channels = np.zeros(num_layers, dtype=int)
        
        for layer_idx in range(num_layers):
            if layer_idx in self.activation_cache:
                cache = self.activation_cache[layer_idx]
                
                # Input activations [batch, seq_len, hidden_dim]
                input_act = cache['input']
                max_val, max_idx = input_act.abs().max(dim=-1)
                input_max[layer_idx] = max_val.max().item()
                input_max_channels[layer_idx] = max_idx[max_val.argmax()].item()
                
                # Output activations
                output_act = cache['output']
                max_val, max_idx = output_act.abs().max(dim=-1)
                output_max[layer_idx] = max_val.max().item()
                output_max_channels[layer_idx] = max_idx[max_val.argmax()].item()
        
        return {
            'input_max': input_max,
            'output_max': output_max,
            'input_max_channels': input_max_channels,
            'output_max_channels': output_max_channels
        }
    
    def detect_activation_spikes(
        self, 
        activation_stats: Dict[str, np.ndarray],
        threshold_multiplier: float = 10.0
    ) -> List[Tuple[int, str]]:
        """Detect layers with activation spikes.
        
        Args:
            activation_stats: Statistics from analyze_down_proj_activations
            threshold_multiplier: Multiplier of mean to consider as spike
        
        Returns:
            List of (layer_idx, 'input'|'output') tuples indicating spikes
        """
        spikes = []
        
        # Check input activations
        input_max = activation_stats['input_max']
        input_mean = input_max.mean()
        input_threshold = input_mean * threshold_multiplier
        
        for layer_idx, val in enumerate(input_max):
            if val > input_threshold:
                spikes.append((layer_idx, 'input'))
                logger.info(f"Input spike detected at layer {layer_idx}: {val:.2f}")
        
        # Check output activations
        output_max = activation_stats['output_max']
        output_mean = output_max.mean()
        output_threshold = output_mean * threshold_multiplier
        
        for layer_idx, val in enumerate(output_max):
            if val > output_threshold:
                spikes.append((layer_idx, 'output'))
                logger.info(f"Output spike detected at layer {layer_idx}: {val:.2f}")
        
        return spikes
    
    def track_super_activation_persistence(
        self,
        super_activation_channel: int,
        start_layer: int
    ) -> Dict[str, List]:
        """Track how super activation persists across layers.
        
        Args:
            super_activation_channel: Channel index of super activation
            start_layer: Layer where super activation first appears
        
        Returns:
            Dictionary with persistence information
        """
        persistence = {
            'magnitudes': [],
            'layers': [],
            'constant_magnitude': True
        }
        
        first_magnitude = None
        
        for layer_idx in sorted(self.activation_cache.keys()):
            if layer_idx < start_layer:
                continue
            
            output_act = self.activation_cache[layer_idx]['output']
            # Get magnitude at super activation channel
            channel_act = output_act[..., super_activation_channel]
            magnitude = channel_act.abs().max().item()
            
            persistence['magnitudes'].append(magnitude)
            persistence['layers'].append(layer_idx)
            
            if first_magnitude is None:
                first_magnitude = magnitude
            elif abs(magnitude - first_magnitude) / first_magnitude > 0.1:
                persistence['constant_magnitude'] = False
        
        return persistence
    
    def get_activation_statistics(
        self, 
        layer_idx: int, 
        activation_type: str = 'output'
    ) -> Dict[str, float]:
        """Get statistical measures of activations for a layer.
        
        Args:
            layer_idx: Layer index
            activation_type: 'input' or 'output'
        
        Returns:
            Dictionary with statistics
        """
        if layer_idx not in self.activation_cache:
            return {}
        
        act = self.activation_cache[layer_idx][activation_type]
        
        return {
            'mean': act.mean().item(),
            'std': act.std().item(),
            'max': act.max().item(),
            'min': act.min().item(),
            'max_abs': act.abs().max().item(),
            'num_outliers_3std': (act.abs() > (act.mean() + 3 * act.std())).sum().item()
        }
    
    def compute_token_wise_activations(
        self,
        layer_idx: int,
        token_positions: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """Get activations for specific token positions.
        
        Args:
            layer_idx: Layer index
            token_positions: List of token positions (None for all)
        
        Returns:
            Dictionary mapping token position to activation tensor
        """
        if layer_idx not in self.activation_cache:
            return {}
        
        output_act = self.activation_cache[layer_idx]['output']
        batch_size, seq_len, hidden_dim = output_act.shape
        
        token_activations = {}
        positions = token_positions if token_positions is not None else range(seq_len)
        
        for pos in positions:
            if pos < seq_len:
                token_activations[pos] = output_act[:, pos, :]
        
        return token_activations
