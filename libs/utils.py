# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import random
import torch
import torch.multiprocessing as mp

import numpy as np
from tensorboardX import SummaryWriter
import traceback, time, glob
from typing import Optional, Dict
    
def get_model_state_path(prefix: str, version: Optional[str] = None) -> Optional[str]:
    """
    获取模型文件路径
    Args:
        prefix: 模型前缀 (e.g. 'local')
        version: 指定版本号 (None表示最新版本)
    Returns:
        完整文件路径或None
    """
    # 构造标准化路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    model_dir = os.path.normpath(os.path.join(base_dir, prefix))
    
    # 版本匹配逻辑
    pattern = f"{prefix}_{version}_*.td" if version else f"{prefix}_*.td"
    candidates = glob.glob(os.path.join(model_dir, pattern))
    
    # 过滤精确版本匹配（防止版本号部分匹配）
    if version:
        candidates = [f for f in candidates 
                     if os.path.basename(f).startswith(f"{prefix}_{version}_")]
    
    if not candidates:
        return None
    
    # 按修改时间排序
    return max(candidates, key=os.path.getmtime, default=None)

def get_model_from_file(
    model: torch.nn.Module,
    prefix: str,
    version: Optional[str] = None,
    map_location: Optional[str] = None,
    strict: bool = True
) -> Optional[Dict]:
    """
    加载模型状态字典
    Args:
        model: 待加载的模型实例
        prefix: 模型前缀
        version: 指定版本号
        map_location: 设备映射 (e.g. 'cuda:0')
        strict: 是否严格匹配模型结构
    Returns:
        包含元信息的字典或None
    """
    #从models文件夹加载模型 prefix 为前缀 local、sample、grad  filename: prefix_version_y_m_d_h_m_s.td(torch.state_dict)
    #路径为 models/prefix/prefix_version_y_m_d_h_m_s.td
    file_path = get_model_state_path(prefix, version)
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        checkpoint = torch.load(file_path, map_location=map_location)
        if 'state_dict' not in checkpoint or 'version' not in checkpoint:
            raise ValueError("Invalid checkpoint format")
        
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        return {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    
    except (IOError, RuntimeError, ValueError) as e:
        print(f"Load model failed: {str(e)}")
        return None

def save_model_to_file(
    model: torch.nn.Module,
    prefix: str,
    version: str,
    metadata: Optional[Dict] = {},
    timestamp_format: str = "%Y%m%d%H%M%S",
    max_versions: int = 5
) -> Optional[str]:
    """
    保存模型状态
    Args:
        model: 待保存的模型
        prefix: 模型前缀
        version: 版本标识符
        metadata: 附加元信息
        timestamp_format: 时间戳格式 (None禁用时间戳)
        max_versions: 最大保留版本数
    Returns:
        保存的文件路径或None
    """
    # 构造保存目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    save_dir = os.path.normpath(os.path.join(base_dir, prefix))
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名
    timestamp = time.strftime(timestamp_format) if timestamp_format else ""
    filename = f"{prefix}_{version}_{timestamp}.td" if timestamp else f"{prefix}_{version}.td"
    save_path = os.path.join(save_dir, filename)

    # 构建检查点
    checkpoint = {
        'state_dict': model.state_dict(),
        'version': version,
        'timestamp': time.time(),
    }
    if metadata:
        checkpoint.update(metadata)

    try:
        torch.save(checkpoint, save_path)
        # 版本清理
        if max_versions > 0:
            existing = sorted(glob.glob(os.path.join(save_dir, f"{prefix}_{version}_*.td")),
                            key=os.path.getmtime, 
                            reverse=True)
            for old_file in existing[max_versions-1:]:
                os.remove(old_file)
        return save_path
    except (IOError, RuntimeError) as e:
        print(f"Save model failed: {str(e)}")
        return None
    
# start_time = time.time()
#运行退出
def exit_run(): 
    # if time.time()-start_time > 4*60*60: 
        # return True
    path = os.path.abspath(os.path.dirname(__file__) + '/' + '../') 
    if os.path.exists(path+"/exit.cmd"):
        return True
    else:
        return False
    
#杀掉进程
def kill_process(pid):
    if os.name == 'nt':  # Windows
        cmd = 'taskkill /pid ' + str(pid) + ' /f'  # 强制终止
    elif os.name == 'posix':  # Linux/Unix
        cmd = 'kill ' + str(pid)  # 默认发送SIGTERM(15)
    else:
        print('Undefined os.name')
        return
    
    try:
        os.system(cmd)
        print(pid, 'killed')
    except Exception as e:
        print(e)

#设置随机种子      
def setup_seed(seed = None):
    if seed is None:
        seed = 1970010101

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True