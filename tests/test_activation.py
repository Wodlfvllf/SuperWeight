"""Unit tests for ActivationAnalyzer."""

import pytest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activation_analyzer import ActivationAnalyzer


@pytest.fixture
def simple_model():
    return torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512)
    )


@pytest.fixture
def analyzer(simple_model):
    config = {'super_weight_detection': {}}
    return ActivationAnalyzer(simple_model, config)


def test_hook_registration(analyzer, simple_model):
    target_layers = {0: simple_model[0], 2: simple_model[2]}
    analyzer.register_hooks(target_layers)
    
    assert len(analyzer.hooks) == 2
    
    analyzer.clear_hooks()
    assert len(analyzer.hooks) == 0


def test_activation_capture(analyzer, simple_model):
    target_layers = {0: simple_model[0]}
    analyzer.register_hooks(target_layers)
    
    # Forward pass
    x = torch.randn(2, 512)
    _ = simple_model(x)
    
    # Check activations captured
    assert 0 in analyzer.activation_cache
    assert 'input' in analyzer.activation_cache[0]
    assert 'output' in analyzer.activation_cache[0]
    
    analyzer.clear_hooks()


def test_analyze_activations(analyzer, simple_model):
    target_layers = {0: simple_model[0], 2: simple_model[2]}
    analyzer.register_hooks(target_layers)
    
    x = torch.randn(2, 512)
    _ = simple_model(x)
    
    stats = analyzer.analyze_down_proj_activations(num_layers=3)
    
    assert 'input_max' in stats
    assert 'output_max' in stats
    assert len(stats['input_max']) == 3
    
    analyzer.clear_hooks()


def test_spike_detection(analyzer, simple_model):
    # Create artificial spike
    target_layers = {0: simple_model[0]}
    analyzer.register_hooks(target_layers)
    
    x = torch.randn(2, 512)
    x[:, 0] = 1000.0  # Create spike
    _ = simple_model(x)
    
    stats = analyzer.analyze_down_proj_activations(num_layers=1)
    spikes = analyzer.detect_activation_spikes(stats, threshold_multiplier=2.0)
    
    assert len(spikes) > 0
    
    analyzer.clear_hooks()
