# -*- coding: utf-8 -*-
"""Algorithm registry and factory functions."""
from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate

# Algorithm registry - maps env_name to (Net, Agent, Calculate) classes
_REGISTRY = {}


def register(env_name, net_cls, agent_cls, calculate_cls):
    _REGISTRY[env_name] = (net_cls, agent_cls, calculate_cls)


def create_net(env_name: str) -> AlgoBaseNet:
    if env_name not in _REGISTRY:
        _lazy_load(env_name)
    net_cls, _, _ = _REGISTRY[env_name]
    return net_cls()


def create_agent(env_name: str, net: AlgoBaseNet, is_checker: bool = False) -> AlgoBaseAgent:
    if env_name not in _REGISTRY:
        _lazy_load(env_name)
    _, agent_cls, _ = _REGISTRY[env_name]
    return agent_cls(net, is_checker)


def create_calculate(env_name: str, net: AlgoBaseNet) -> AlgoBaseCalculate:
    if env_name not in _REGISTRY:
        _lazy_load(env_name)
    _, _, calc_cls = _REGISTRY[env_name]
    return calc_cls(net)


def _lazy_load(env_name: str):
    """Lazy-load algorithm modules to populate registry."""
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
