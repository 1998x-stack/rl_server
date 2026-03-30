"""pytest 共享固件：临时模型目录、小型 DQN 网络等。"""
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_model_dir(tmp_path):
    """提供空临时 ``models`` 子目录，用于检查点测试。"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def small_dqn_net():
    """构造轻量 ``DQNGymClassicNet`` 供单元测试快速执行。"""
    from algo_envs.dqn_gym_classic import DQNGymClassicNet
    net = DQNGymClassicNet()
    return net
