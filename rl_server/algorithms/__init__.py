# -*- coding: utf-8 -*-
"""算法注册表与工厂：按 ``env_name`` 惰性加载并构造网络、智能体、梯度计算器。"""
from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate

# 环境名 -> (网络类, 智能体类, 计算器类)
_REGISTRY = {}


def register(env_name, net_cls, agent_cls, calculate_cls):
    """将一组实现类注册到全局表。

    Args:
        env_name: 配置中使用的环境/算法标识字符串。
        net_cls: 继承 ``AlgoBaseNet`` 的网络类。
        agent_cls: 继承 ``AlgoBaseAgent`` 的智能体类。
        calculate_cls: 继承 ``AlgoBaseCalculate`` 的梯度计算器类。
    """
    _REGISTRY[env_name] = (net_cls, agent_cls, calculate_cls)


def create_net(env_name: str) -> AlgoBaseNet:
    """构造未训练的网络实例（惰性加载对应子模块）。

    Args:
        env_name: 如 ``DQNGymClassic``、``MujocoNormal`` 等。

    Returns:
        ``AlgoBaseNet`` 子类实例。

    Raises:
        ValueError: 未知环境名或子模块导入失败。
    """
    if env_name not in _REGISTRY:
        _lazy_load(env_name)
    net_cls, _, _ = _REGISTRY[env_name]
    return net_cls()


def create_agent(env_name: str, net: AlgoBaseNet, is_checker: bool = False) -> AlgoBaseAgent:
    """构造智能体，可选评估模式。

    Args:
        env_name: 注册表中的环境名。
        net: 已创建的网络实例。
        is_checker: 为 ``True`` 时使用检查/评估路径（若算法区分）。

    Returns:
        ``AlgoBaseAgent`` 子类实例。
    """
    if env_name not in _REGISTRY:
        _lazy_load(env_name)
    _, agent_cls, _ = _REGISTRY[env_name]
    return agent_cls(net, is_checker)


def create_calculate(env_name: str, net: AlgoBaseNet) -> AlgoBaseCalculate:
    """构造梯度计算器，供 Trainer 进程使用。

    Args:
        env_name: 注册表中的环境名。
        net: 共享的策略/价值网络。

    Returns:
        ``AlgoBaseCalculate`` 子类实例。
    """
    if env_name not in _REGISTRY:
        _lazy_load(env_name)
    _, _, calc_cls = _REGISTRY[env_name]
    return calc_cls(net)


def _lazy_load(env_name: str):
    """按需导入算法子包并填充 ``_REGISTRY``，避免启动时加载全部依赖。

    Args:
        env_name: 环境名。

    Raises:
        ValueError: 未知 ``env_name`` 或 ``ImportError`` 包装后抛出。
    """
    try:
        if env_name == 'DQNGymClassic':
            from rl_server.algorithms.dqn.network import DQNGymClassicNet
            from rl_server.algorithms.dqn.agent import DQNGymClassicAgent
            from rl_server.algorithms.dqn.calculator import DQNGymClassicCalculate
            register(env_name, DQNGymClassicNet, DQNGymClassicAgent, DQNGymClassicCalculate)
        elif env_name == 'MujocoNormal':
            from rl_server.algorithms.ppo.mujoco_normal import MujocoNormalNet, MujocoNormalAgent, MujocoNormalCalculate
            register(env_name, MujocoNormalNet, MujocoNormalAgent, MujocoNormalCalculate)
        elif env_name == 'MicroRTS':
            from rl_server.algorithms.ppo.microrts import MicroRTSNet, MicroRTSAgent, MicroRTSCalculate
            register(env_name, MicroRTSNet, MicroRTSAgent, MicroRTSCalculate)
        elif env_name == 'MujocoBeta':
            from rl_server.algorithms.ppo.mujoco_beta import MujocoBetaNet, MujocoBetaAgent, MujocoBetaCalculate
            register(env_name, MujocoBetaNet, MujocoBetaAgent, MujocoBetaCalculate)
        elif env_name == 'SACMujocoNormal':
            from rl_server.algorithms.sac.mujoco_normal import MujocoNormalQNet, MujocoNormalQAgent, MujocoNormalQCalculate
            register(env_name, MujocoNormalQNet, MujocoNormalQAgent, MujocoNormalQCalculate)
        elif env_name == 'TD3MujocoNormal':
            from rl_server.algorithms.td3.mujoco_normal import MujocoNormalQNet as TD3Net, MujocoNormalQAgent as TD3Agent, MujocoNormalQCalculate as TD3Calculate
            register(env_name, TD3Net, TD3Agent, TD3Calculate)
        else:
            raise ValueError(f"Unknown algorithm: {env_name}")
    except ImportError as e:
        raise ValueError(f"Failed to load algorithm {env_name}: {e}")
