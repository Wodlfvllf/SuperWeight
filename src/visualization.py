"""Visualization utilities for super weight analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

sns.set_style("whitegrid")


class SuperWeightVisualizer:
    """Visualization tools for super weight detection and analysis."""
    
    def __init__(self, output_dir: str = "./results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_activation_distribution(
        self,
        activation_stats: Dict[str, np.ndarray],
        model_name: str,
        save: bool = True
    ):
        """Plot maximum activation values across layers (Figure 3 from paper).
        
        Args:
            activation_stats: Dictionary with 'input_max' and 'output_max'
            model_name: Name of model for title
            save: Whether to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        num_layers = len(activation_stats['input_max'])
        layers = np.arange(num_layers)
        
        # Plot input activations
        ax1.plot(layers, activation_stats['input_max'], 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Layer Number', fontsize=12)
        ax1.set_ylabel('Max Activation Value', fontsize=12)
        ax1.set_title(f'{model_name} Max down_proj Input', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot output activations
        ax2.plot(layers, activation_stats['output_max'], 'o-', linewidth=2, markersize=6)
        ax2.set_xlabel('Layer Number', fontsize=12)
        ax2.set_ylabel('Max Activation Value', fontsize=12)
        ax2.set_title(f'{model_name} Max down_proj Output', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name}_activation_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        plt.show()
    
    def plot_super_weight_sensitivity(
        self,
        scaling_factors: List[float],
        accuracies: Dict[str, List[float]],
        model_name: str,
        save: bool = True
    ):
        """Plot super weight sensitivity analysis (Figure 6 from paper).
        
        Args:
            scaling_factors: List of scaling factors tested
            accuracies: Dictionary mapping task names to accuracy lists
            model_name: Name of model
            save: Whether to save plot
        """
        plt.figure(figsize=(10, 6))
        
        for task_name, acc_list in accuracies.items():
            plt.plot(scaling_factors, acc_list, 'o-', label=task_name, linewidth=2, markersize=6)
        
        # Add baseline line at scaling factor 1.0
        if 1.0 in scaling_factors:
            idx = scaling_factors.index(1.0)
            baseline_acc = list(accuracies.values())[0][idx]
            plt.axhline(y=baseline_acc, color='purple', linestyle='--', 
                       label='Original', linewidth=2, alpha=0.7)
        
        plt.xlabel('SW Scaling Factor', fontsize=12)
        plt.ylabel('Average Zero-Shot Acc.', fontsize=12)
        plt.title(f'{model_name} Super Weight Sensitivity', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name}_sensitivity.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        plt.show()
    
    def plot_quantization_comparison(
        self,
        block_sizes: List[int],
        methods: Dict[str, List[float]],
        model_name: str,
        save: bool = True
    ):
        """Plot quantization quality across block sizes (Figure 7 from paper).
        
        Args:
            block_sizes: List of block sizes tested
            methods: Dictionary mapping method names to accuracy lists
            model_name: Name of model
            save: Whether to save
        """
        plt.figure(figsize=(10, 6))
        
        for method_name, accuracies in methods.items():
            plt.plot(block_sizes, accuracies, 'o-', label=method_name, 
                    linewidth=2, markersize=8)
        
        plt.xlabel('Block size', fontsize=12)
        plt.ylabel('Average Zero-shot Accuracy', fontsize=12)
        plt.title(f'{model_name} Block Scaling', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show block sizes nicely
        plt.xticks(range(len(block_sizes)), 
                  [f'{b}x{b}' for b in block_sizes])
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name}_quantization_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        plt.show()
    
    def plot_token_probability_distribution(
        self,
        token_probs_original: Dict[str, float],
        token_probs_no_sw: Dict[str, float],
        model_name: str,
        save: bool = True
    ):
        """Plot token probability distributions (Figure 5 from paper).
        
        Args:
            token_probs_original: Token probabilities with super weight
            token_probs_no_sw: Token probabilities without super weight
            model_name: Name of model
            save: Whether to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        tokens = list(token_probs_original.keys())
        x = np.arange(len(tokens))
        width = 0.35
        
        probs_orig = [token_probs_original[t] for t in tokens]
        probs_no_sw = [token_probs_no_sw[t] for t in tokens]
        
        ax.bar(x - width/2, probs_orig, width, label='Original', alpha=0.8)
        ax.bar(x + width/2, probs_no_sw, width, label='No SW', alpha=0.8)
        
        ax.set_xlabel('Token Labels', fontsize=12)
        ax.set_ylabel('Probabilities', fontsize=12)
        ax.set_title(f'{model_name} Token Probabilities', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'{model_name}_token_probabilities.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        plt.show()
    
    def plot_weight_heatmap(
        self,
        weight_matrix: np.ndarray,
        super_weight_coords: List[Tuple[int, int]],
        layer_name: str,
        save: bool = True
    ):
        """Plot heatmap of weight matrix highlighting super weights.
        
        Args:
            weight_matrix: Weight matrix to visualize
            super_weight_coords: List of (row, col) for super weights
            layer_name: Name of layer
            save: Whether to save
        """
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(weight_matrix, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Weight Value'})
        
        # Mark super weights
        for row, col in super_weight_coords:
            plt.plot(col + 0.5, row + 0.5, 'g*', markersize=20, 
                    markeredgecolor='black', markeredgewidth=2)
        
        plt.title(f'{layer_name} Weights (★ = Super Weight)', fontsize=14)
        plt.xlabel('Input Dimension', fontsize=12)
        plt.ylabel('Output Dimension', fontsize=12)
        
        if save:
            filepath = os.path.join(self.output_dir, f'{layer_name}_heatmap.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        plt.show()
