# -*- coding: utf-8 -*-
"""通用工具：信号、模型读写、随机种子、多进程与模仿学习样本导出。

含遗留 ``exit.cmd`` 退出检测与进程终止封装。
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import random
import torch
import torch.multiprocessing as mp # 计算密集型，而非IO密集型，GIL

import numpy as np
from tensorboardX import SummaryWriter
import traceback, time, glob
from typing import Optional, Dict
import signal
import threading

_shutdown_event = threading.Event()


def setup_signal_handlers():
    """注册 SIGTERM/SIGINT，将内部 ``_shutdown_event`` 置位。"""
    def _handler(signum, frame):
        _shutdown_event.set()
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    
def get_model_state_path(prefix: str, version: Optional[str] = None) -> Optional[str]:
    """在 ``models/{prefix}/`` 下按前缀与可选版本匹配 ``.td`` 文件，返回最近修改的一条。

    Args:
        prefix: 模型前缀目录名。
        version: 若指定则文件名需以 ``{prefix}_{version}_`` 开头；``None`` 表示取最新。

    Returns:
        匹配到的绝对路径；无匹配则为 ``None``。
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
)-> Optional[int]:
    """从磁盘加载 ``state_dict`` 到 ``model``。

    Args:
        model: 待加载模块。
        prefix: 与保存时一致的前缀。
        version: 可选版本；``None`` 取最新。
        map_location: ``torch.load`` 设备映射。
        strict: 是否严格匹配键。

    Returns:
        检查点中的版本号；失败为 ``None``。
    """
    # 从models文件夹加载模型 prefix 为前缀 local、sample、grad  filename: prefix_version_y_m_d_h_m_s.td(torch.state_dict)
    # 路径为 models/prefix/prefix_version_y_m_d_h_m_s.td
    file_path = get_model_state_path(prefix, version)
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        checkpoint = torch.load(file_path, map_location=map_location)
        if 'state_dict' not in checkpoint or 'version' not in checkpoint:
            raise ValueError("Invalid checkpoint format")
        
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        return {k: v for k, v in checkpoint.items() if k != 'state_dict'}['version']
    
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
    """将 ``state_dict`` 写入 ``models/{prefix}/`` 并可选裁剪旧文件。

    Args:
        model: 要保存的模块。
        prefix: 目录与文件名前缀。
        version: 版本字符串。
        metadata: 合并进检查点的额外字段。
        timestamp_format: 时间戳 ``strftime`` 格式；空字符串则文件名不含时间。
        max_versions: 同一 ``prefix_version`` 下保留的带时间戳文件数量。

    Returns:
        保存路径或 ``None``。
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
    
def exit_run() -> bool:
    """是否应退出：内部 shutdown 事件或项目根目录存在 ``exit.cmd``（兼容旧逻辑）。"""
    if _shutdown_event.is_set():
        return True
    # Legacy fallback: check for exit.cmd file
    path = os.path.abspath(os.path.dirname(__file__) + '/' + '../')
    if os.path.exists(path + "/exit.cmd"):
        return True
    return False
    
def kill_process(pid):
    """按 PID 终止进程（Windows 用 taskkill，POSIX 用 kill）。"""
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

def setup_seed(seed = None):
    """固定 PyTorch/NumPy/Python 随机种子及 cudnn 确定性选项。"""
    if seed is None:
        seed = 1970010101
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 硬件与框架优化设置
    torch.backends.cudnn.deterministic = True  # 禁用非确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭自动优化
    os.environ['PYTHONHASHSEED'] = str(seed)   # 防止哈希随机化，固定 ​Python 解释器的哈希随机化种子，确保字典遍历、集合操作的顺序一致性
    
def setup_mp():
    """多进程 ``spawn`` 并限制 OpenMP/MKL 线程数。"""
    # 设置多进程模式
    mp.set_start_method('spawn')  # 强制使用安全的进程创建方式
    os.environ["OMP_NUM_THREADS"] = "1"  # 避免 OpenMP 线程冲突
    os.environ["MKL_NUM_THREADS"] = "1"  # 避免 MKL 数学库线程冲突
    print(f"[DEBUG] 当前启动方法: {mp.get_start_method()}")
    print(f"[DEBUG] OMP线程数: {os.environ.get('OMP_NUM_THREADS')}")
    

def save_samples_for_IL(sample_queue: mp.Queue):
    """从队列耗尽样本并保存为 ``obs.npy`` / ``act.npy``（模仿学习数据导出）。"""
    obs_array = []
    act_array = []
    while not sample_queue.empty():
        samples_info = sample_queue.get()
        exp_lists = samples_info['exps']
        for exp in exp_lists:
            observation = np.expand_dims(exp[0], axis=0)
            action = np.expand_dims(exp[1], axis=0)
            obs_array.append(observation)
            act_array.append(action)
            
    obs_array = np.concatenate(obs_array, axis=0)
    act_array = np.concatenate(act_array, axis=0)
    np.save('obs.npy', obs_array)
    np.save('act.npy', act_array)