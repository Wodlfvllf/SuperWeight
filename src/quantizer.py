"""Quantization with super outlier preservation."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

from .utils import clip_outliers, compute_z_score

logger = logging.getLogger(__name__)


class SuperOutlierQuantizer:
    """Quantization methods that preserve super weights and activations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.quant_config = config['quantization']
        
    def quantize_weight_tensor(
        self,
        weight: torch.Tensor,
        num_bits: int = 4,
        block_size: Tuple[int, int] = (128, 128),
        super_weight_coords: Optional[List[Tuple[int, int]]] = None,
        z_clip: float = 3.0
    ) -> Tuple[torch.Tensor, Dict]:
        """Quantize weight tensor with super weight preservation.
        
        Algorithm (Equation 2 from paper):
        1. CLIP: Clip outliers using z-score threshold
        2. Q: Quantize clipped weights
        3. Q^-1: Dequantize
        4. RESTORE: Restore super weights in FP16
        
        Args:
            weight: Weight tensor [out_features, in_features]
            num_bits: Number of quantization bits
            block_size: Block size for quantization
            super_weight_coords: List of (row, col) for super weights
            z_clip: Z-score threshold for clipping
        
        Returns:
            Quantized and dequantized weight, metadata dictionary
        """
        original_dtype = weight.dtype
        original_device = weight.device
        
        # Store original super weight values
        super_weight_values = {}
        if super_weight_coords:
            for row, col in super_weight_coords:
                super_weight_values[(row, col)] = weight[row, col].clone()
        
        # Step 1: CLIP outliers (including super weights temporarily)
        clipped_weight = clip_outliers(weight, z_threshold=z_clip)
        
        # Step 2 & 3: Quantize and dequantize
        quant_weight, quant_metadata = self._block_quantize_dequantize(
            clipped_weight, num_bits, block_size
        )
        
        # Step 4: RESTORE super weights in original precision
        if super_weight_coords:
            for row, col in super_weight_coords:
                quant_weight[row, col] = super_weight_values[(row, col)]
        
        metadata = {
            'num_bits': num_bits,
            'block_size': block_size,
            'z_clip': z_clip,
            'super_weights_restored': len(super_weight_coords) if super_weight_coords else 0,
            **quant_metadata
        }
        
        return quant_weight.to(dtype=original_dtype, device=original_device), metadata
    
    def _block_quantize_dequantize(
        self,
        tensor: torch.Tensor,
        num_bits: int,
        block_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Dict]:
        """Perform block-wise round-to-nearest quantization.
        
        Args:
            tensor: Input tensor
            num_bits: Number of bits
            block_size: (rows, cols) block size
        
        Returns:
            Dequantized tensor and metadata
        """
        out_features, in_features = tensor.shape
        block_rows, block_cols = block_size
        
        # Calculate quantization levels
        n_levels = 2 ** num_bits
        
        # Pad tensor to be divisible by block size
        pad_rows = (block_rows - out_features % block_rows) % block_rows
        pad_cols = (block_cols - in_features % block_cols) % block_cols
        
        if pad_rows > 0 or pad_cols > 0:
            tensor_padded = torch.nn.functional.pad(
                tensor, (0, pad_cols, 0, pad_rows), value=0
            )
        else:
            tensor_padded = tensor
        
        quant_tensor = torch.zeros_like(tensor_padded)
        
        # Block-wise quantization
        for i in range(0, tensor_padded.shape[0], block_rows):
            for j in range(0, tensor_padded.shape[1], block_cols):
                block = tensor_padded[i:i+block_rows, j:j+block_cols]
                
                # Asymmetric quantization
                min_val = block.min()
                max_val = block.max()
                
                scale = (max_val - min_val) / (n_levels - 1)
                
                # Quantize: Q(X) = Round((X - MIN(X)) / δ)
                if scale > 0:
                    quant_block = torch.round((block - min_val) / scale)
                    quant_block = torch.clamp(quant_block, 0, n_levels - 1)
                    
                    # Dequantize: Q^-1(X) = X * δ + MIN(X)
                    dequant_block = quant_block * scale + min_val
                else:
                    dequant_block = block
                
                quant_tensor[i:i+block_rows, j:j+block_cols] = dequant_block
        
        # Remove padding
        quant_tensor = quant_tensor[:out_features, :in_features]
        
        metadata = {
            'padded_shape': tensor_padded.shape,
            'original_shape': tensor.shape,
            'num_blocks': (tensor_padded.shape[0] // block_rows) * 
                         (tensor_padded.shape[1] // block_cols)
        }
        
        return quant_tensor, metadata
    
    def quantize_activation_tensor(
        self,
        activation: torch.Tensor,
        num_bits: int = 8,
        super_activation_coords: Optional[List[int]] = None,
        per_token: bool = True
    ) -> torch.Tensor:
        """Quantize activation tensor with super activation preservation.
        
        Algorithm (Equation 1 from paper):
        1. REPLACE: Replace super activation with median
        2. Q: Quantize activations
        3. Q^-1: Dequantize
        4. RESTORE: Restore super activation in FP16
        
        Args:
            activation: Activation tensor [batch, seq_len, hidden_dim]
            num_bits: Number of bits
            super_activation_coords: List of channel indices for super activations
            per_token: Whether to use per-token quantization
        
        Returns:
            Quantized activation tensor
        """
        original_shape = activation.shape
        original_dtype = activation.dtype
        
        # Store original super activation values
        super_activation_values = {}
        if super_activation_coords:
            for channel_idx in super_activation_coords:
                # Store entire channel across all tokens
                super_activation_values[channel_idx] = activation[..., channel_idx].clone()
        
        # Step 1: REPLACE super activations with median
        if super_activation_coords:
            for channel_idx in super_activation_coords:
                median_val = activation[..., channel_idx].median()
                activation[..., channel_idx] = median_val
        
        # Step 2 & 3: Quantize and dequantize
        if per_token:
            quant_activation = self._per_token_quantize_dequantize(activation, num_bits)
        else:
            quant_activation = self._per_tensor_quantize_dequantize(activation, num_bits)
        
        # Step 4: RESTORE super activations in original precision
        if super_activation_coords:
            for channel_idx in super_activation_coords:
                quant_activation[..., channel_idx] = super_activation_values[channel_idx]
        
        return quant_activation.to(dtype=original_dtype)
    
    def _per_token_quantize_dequantize(
        self,
        tensor: torch.Tensor,
        num_bits: int
    ) -> torch.Tensor:
        """Per-token symmetric quantization."""
        # tensor shape: [batch, seq_len, hidden_dim]
        n_levels = 2 ** num_bits
        
        # Compute per-token scales
        max_vals = tensor.abs().max(dim=-1, keepdim=True)[0]
        scale = max_vals / (n_levels / 2 - 1)
        
        # Quantize
        quant_tensor = torch.round(tensor / (scale + 1e-8))
        quant_tensor = torch.clamp(quant_tensor, -(n_levels // 2), (n_levels // 2) - 1)
        
        # Dequantize
        dequant_tensor = quant_tensor * scale
        
        return dequant_tensor
    
    def _per_tensor_quantize_dequantize(
        self,
        tensor: torch.Tensor,
        num_bits: int
    ) -> torch.Tensor:
        """Per-tensor symmetric quantization."""
        n_levels = 2 ** num_bits
        
        # Global scale
        max_val = tensor.abs().max()
        scale = max_val / (n_levels / 2 - 1)
        
        # Quantize
        quant_tensor = torch.round(tensor / (scale + 1e-8))
        quant_tensor = torch.clamp(quant_tensor, -(n_levels // 2), (n_levels // 2) - 1)
        
        # Dequantize
        dequant_tensor = quant_tensor * scale
        
        return dequant_tensor
    
    def quantize_model_weights(
        self,
        model: nn.Module,
        super_weight_index: List[Tuple[int, int, int]],
        num_bits: int = 4,
        block_size: Tuple[int, int] = (128, 128)
    ) -> Dict:
        """Quantize all model weights with super weight preservation.
        
        Args:
            model: Model to quantize
            super_weight_index: List of (layer, row, col) for super weights
            num_bits: Quantization bits
            block_size: Block size
        
        Returns:
            Dictionary with quantization statistics
        """
        stats = {
            'layers_quantized': 0,
            'super_weights_preserved': 0,
            'total_params': 0
        }
        
        # Group super weights by layer
        layer_super_weights = {}
        for layer_idx, row, col in super_weight_index:
            if layer_idx not in layer_super_weights:
                layer_super_weights[layer_idx] = []
            layer_super_weights[layer_idx].append((row, col))
        
        # Quantize each down_proj layer
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
        
        if hasattr(base_model, 'layers'):
            for layer_idx, layer in enumerate(base_model.layers):
                if hasattr(layer.mlp, 'down_proj'):
                    down_proj = layer.mlp.down_proj
                    
                    super_coords = layer_super_weights.get(layer_idx, None)
                    
                    quant_weight, metadata = self.quantize_weight_tensor(
                        down_proj.weight.data,
                        num_bits=num_bits,
                        block_size=block_size,
                        super_weight_coords=super_coords
                    )
                    
                    down_proj.weight.data = quant_weight
                    
                    stats['layers_quantized'] += 1
                    stats['super_weights_preserved'] += metadata['super_weights_restored']
                    stats['total_params'] += down_proj.weight.numel()
        
        logger.info(f"Quantized {stats['layers_quantized']} layers, "
                   f"preserved {stats['super_weights_preserved']} super weights")
        
        return stats
