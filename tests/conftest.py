"""pytest 共享固件：临时模型目录、小型 DQN 网络等。"""
import pytest


@pytest.fixture
def temp_model_dir(tmp_path):
    """提供空临时 ``models`` 子目录，用于检查点测试。"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def small_dqn_net():
    """构造轻量 ``DQNGymClassicNet`` 供单元测试快速执行。"""
    from rl_server.algorithms.dqn.network import DQNGymClassicNet
    net = DQNGymClassicNet()
    return net
