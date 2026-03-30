# -*- coding: utf-8 -*-
"""遗留全局工厂：从 YAML 加载配置，按名称实例化 ``algo_envs`` 中的 Net/Agent/Calculate。

通过 ``globals()`` 按 ``{env_name}Net`` 等命名查找类；新代码优先使用 ``rl_server.algorithms``。
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

# all for globals()
import algo_envs.algo_base as AlgoBase
from algo_envs.dqn_gym_classic import DQNGymClassicNet,DQNGymClassicAgent,DQNGymClassicCalculate

from algo_envs.ppo_microrts import MicroRTSNet,MicroRTSAgent,MicroRTSCalculate
from algo_envs.ppo_mujoco_normal import MujocoNormalNet,MujocoNormalAgent,MujocoNormalCalculate
from algo_envs.ppo_mujoco_beta import MujocoBetaNet,MujocoBetaAgent,MujocoBetaCalculate
from algo_envs.ppo_mujoco_beta_alpha import MujocoBetaAlphaNet,MujocoBetaAlphaAgent,MujocoBetaAlphaCalculate
from algo_envs.ppo_mujoco_beta_relative import MujocoBetaRelativeNet,MujocoBetaRelativeAgent,MujocoBetaRelativeCalculate


import libs.config_loader as config_loader

# Load configuration from YAML
_CONFIG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'default.yaml')
_OVERRIDE_PATH = os.environ.get('RL_SERVER_CONFIG', None)
_config = config_loader.load_config(_CONFIG_PATH, _OVERRIDE_PATH)


def get_current_env_name() -> str:
    """返回配置中的 ``training.env_name``，默认 ``DQNGymClassic``。"""
    return _config.get('training', {}).get('env_name', 'DQNGymClassic')


def create_net(env_name: str) -> AlgoBase.AlgoBaseNet:
    """按 ``{env_name}Net`` 从已导入符号中实例化网络。"""
    net_name = env_name + 'Net'
    try:
        return globals()[net_name]()
    except:
        print("create_net error. net name is:",net_name)
        exit()

def create_agent(env_name: str, sample_net: AlgoBase.AlgoBaseNet, is_checker: bool = False) -> AlgoBase.AlgoBaseAgent:
    """按 ``{env_name}Agent`` 实例化智能体。"""
    agent_name = env_name + 'Agent'
    try:
        return globals()[agent_name](sample_net, is_checker)
    except:
        if is_checker:
            print("create_check_agent error. agent name is:",agent_name)
        else:
            print("create_sample_agent error. agent name is:",agent_name)
        exit()

def create_calculate(env_name: str, calculate_net: AlgoBase.AlgoBaseNet) -> AlgoBase.AlgoBaseCalculate:
    """按 ``{env_name}Calculate`` 实例化梯度计算器。"""
    calculate_name = env_name + 'Calculate'
    try:
        return globals()[calculate_name](calculate_net)
    except:
        print("create_calculate error. calculate name is:",calculate_name)
        exit()

def get_current_queue_config() -> dict:
    """汇总训练与队列相关超参，供旧版训练脚本使用。"""
    training = _config.get('training', {})
    queues = _config.get('queues', {})
    return {
        'len_grads_queue': queues.get('len_grads_queue', 1000),
        'len_batch_queue': queues.get('len_batch_queue', 1000),
        'len_sample_queue': queues.get('len_sample_queue', 1000),
        'batch_update_grads_server': training.get('batch_update_grads_server', 10),
        'enable_checker': training.get('enable_checker', True),
        'num_trainer': training.get('num_trainers', 1),
        'num_sampler': training.get('num_samplers', 2),
        'num_update_grads': training.get('num_update_grads', 1),
        'version_update_sample_model': training.get('version_update_sample_model', 1),
        'version_update_calculate_model': training.get('version_update_calculate_model', 1),
    }


def _redis_dict_from_config(section: str) -> dict:
    """将某 Redis 小节转为 ``libs.redis_cache`` 所需的 ``ip/port/db/pw`` 字典。"""
    redis_cfg = _config.get('redis', {}).get(section, {})
    return {
        'ip': redis_cfg.get('host', 'localhost'),
        'port': str(redis_cfg.get('port', 6379)),
        'db': str(redis_cfg.get('db', 0)),
        'pw': redis_cfg.get('password', ''),
    }


def get_current_redis_MODEL_CONFIG() -> dict:
    """模型同步用 Redis 连接参数。"""
    return _redis_dict_from_config('model')


def get_current_redis_exps_config() -> dict:
    """经验队列用 Redis 连接参数。"""
    return _redis_dict_from_config('exps')


def get_current_redis_grads_config() -> dict:
    """梯度队列用 Redis 连接参数。"""
    return _redis_dict_from_config('grads')