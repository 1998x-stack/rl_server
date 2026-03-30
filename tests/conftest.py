import os
import sys
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model checkpoints."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def small_dqn_net():
    """Small DQN network for fast testing."""
    from algo_envs.dqn_gym_classic import DQNGymClassicNet
    net = DQNGymClassicNet()
    return net
