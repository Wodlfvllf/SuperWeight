"""Unit tests for SuperWeightDetector."""

import pytest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.super_weight_detector import SuperWeightDetector
from src.model_loader import ModelLoader
from src.utils import load_config


@pytest.fixture
def config():
    return {
        'model': {
            'name': 'gpt2',
            'device_map': 'cpu',
            'torch_dtype': 'float32'
        },
        'super_weight_detection': {
            'prompt': 'The quick brown fox',
            'activation_threshold_multiplier': 10.0
        },
        'evaluation': {
            'tasks': ['lambada']
        }
    }


@pytest.fixture
def model_and_tokenizer(config):
    loader = ModelLoader(config)
    return loader.load_model()


def test_detector_initialization(model_and_tokenizer, config):
    model, tokenizer = model_and_tokenizer
    detector = SuperWeightDetector(model, tokenizer, config)
    assert detector is not None
    assert detector.model == model
    assert detector.tokenizer == tokenizer


def test_super_weight_detection_runs(model_and_tokenizer, config):
    model, tokenizer = model_and_tokenizer
    detector = SuperWeightDetector(model, tokenizer, config)
    
    # Get a small subset of layers for testing
    if hasattr(model, 'transformer'):
        layers = {0: model.transformer.h[0].mlp}
    else:
        pytest.skip("Model structure not compatible")
    
    # Should complete without errors
    coords = detector.detect_super_weights(layers, max_iterations=1)
    assert isinstance(coords, list)


def test_validate_super_weights(model_and_tokenizer, config):
    model, tokenizer = model_and_tokenizer
    detector = SuperWeightDetector(model, tokenizer, config)
    
    fake_coords = [(0, 10, 20)]
    if hasattr(model, 'transformer'):
        layers = {0: model.transformer.h[0].mlp.c_proj}
    else:
        pytest.skip("Model structure not compatible")
    
    results = detector.validate_super_weights(fake_coords, layers)
    assert 'magnitudes' in results
    assert 'relative_magnitudes' in results
