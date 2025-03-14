# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
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


# 获取环境名称
def get_current_env_name() -> str:
    # return "MujocoNormal"
    # return "MujocoBeta"
    # return "MujocoBetaAlpha"
    # return "MujocoBetaRelative"
    return "DQNGymClassic"
    # return "MicroRTS"

def create_net(env_name: str) -> AlgoBase.AlgoBaseNet:
    net_name = env_name + 'Net'
    try:
        return globals()[net_name]()
    except:
        print("create_net error. net name is:",net_name)
        exit()

def create_agent(env_name: str, sample_net: AlgoBase.AlgoBaseNet, is_checker: bool = False) -> AlgoBase.AlgoBaseAgent:
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
    calculate_name = env_name + 'Calculate'
    try:
        return globals()[calculate_name](calculate_net)
    except:
        print("create_calculate error. calculate name is:",calculate_name)
        exit()

# 获取队列参数
def get_current_queue_config() -> dict:
    return queue_args_dict

# 模型redis配置信息
def get_current_redis_MODEL_CONFIG() -> dict:
    return redis_args_dict_server_linux

# 经验redis配置信息
def get_current_redis_exps_config() -> dict:
    return redis_args_dict_server_linux

# 梯度redis配置信息
def get_current_redis_grads_config() -> dict:
    return redis_args_dict_server_linux


# 队列参数
queue_args_dict = dict()
queue_args_dict['len_grads_queue'] = 1000
queue_args_dict['len_batch_queue'] = 1000
queue_args_dict['len_sample_queue'] = 1000
queue_args_dict['batch_update_grads_server'] = 10 # 梯度服务器更新梯度批次
queue_args_dict['enable_checker'] = True
queue_args_dict['num_trainer'] = 1 # 工人数量
queue_args_dict['num_sampler'] = 2
queue_args_dict['num_update_grads'] = 1 # 梯度更新数量
queue_args_dict['version_update_sample_model'] = 1 # 采样模型更新间隔
queue_args_dict['version_update_calculate_model'] = 1 # 计算模型更新间隔

# redis 参数
redis_args_dict_sxl = dict()
redis_args_dict_sxl['ip'] = '192.168.1.69' # 服务器IP
redis_args_dict_sxl['port'] = '6379' # 服务器端口
redis_args_dict_sxl['db'] = '0' # 服务器db
redis_args_dict_sxl['pw'] = '123456' # 服务器密码

redis_args_dict_jyz = dict()
redis_args_dict_jyz['ip'] = '192.168.1.69' # 服务器IP
redis_args_dict_jyz['port'] = '6379' # 服务器端口
redis_args_dict_jyz['db'] = '0' # 服务器db
redis_args_dict_jyz['pw'] = '123456' # 服务器密码

redis_args_dict_xm = dict()
redis_args_dict_xm['ip'] = '192.168.1.69' # 服务器IP
redis_args_dict_xm['port'] = '6379' # 服务器端口
redis_args_dict_xm['db'] = '0' # 服务器db
redis_args_dict_xm['pw'] = '123456' # 服务器密码

redis_args_dict_server_windows = dict()
redis_args_dict_server_windows['ip'] = '192.168.1.229' # 服务器IP
redis_args_dict_server_windows['port'] = '6379' # 服务器端口
redis_args_dict_server_windows['db'] = '0' # 服务器db
redis_args_dict_server_windows['pw'] = '12345678' # 服务器密码

redis_args_dict_server_linux = dict()
redis_args_dict_server_linux['ip'] = '192.168.12.158' # 服务器IP
redis_args_dict_server_linux['port'] = '6379' # 服务器端口
redis_args_dict_server_linux['db'] = '0' # 服务器db
redis_args_dict_server_linux['pw'] = '12345678' # 服务器密码

redis_args_dict_grad = dict()
redis_args_dict_grad['ip'] = '192.168.1.229' # 服务器IP
redis_args_dict_grad['port'] = '6379' # 服务器端口
redis_args_dict_grad['db'] = '0' # 服务器db
redis_args_dict_grad['pw'] = '12345678' # 服务器密码