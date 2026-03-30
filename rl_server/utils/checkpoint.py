# -*- coding: utf-8 -*-
"""模型检查点：按前缀与版本查找、加载与保存 PyTorch 状态字典。"""
import os
import time
import glob
import pickle
import torch
from typing import Optional


def get_model_state_path(prefix: str, version: Optional[str] = None, base_dir: str = 'models') -> Optional[str]:
    """在指定目录下按前缀（及可选版本）查找最新的 ``.td`` 检查点文件。

    Args:
        prefix: 模型名前缀，对应子目录 ``base_dir/prefix``。
        version: 若给定，仅匹配文件名以 ``{prefix}_{version}_`` 开头的文件。
        base_dir: 模型根目录。

    Returns:
        最近修改时间最新的匹配文件路径；若无匹配则返回 ``None``。
    """
    model_dir = os.path.normpath(os.path.join(base_dir, prefix))
    pattern = f"{prefix}_{version}_*.td" if version else f"{prefix}_*.td"
    candidates = glob.glob(os.path.join(model_dir, pattern))
    if version:
        candidates = [f for f in candidates if os.path.basename(f).startswith(f"{prefix}_{version}_")]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime, default=None)


def load_model(
    model: torch.nn.Module,
    prefix: str,
    version: Optional[str] = None,
    base_dir: str = 'models',
    map_location: Optional[str] = None,
) -> Optional[int]:
    """从磁盘加载检查点并写入 ``model``。

    Args:
        model: 待加载权重的模块。
        prefix: 与保存时一致的前缀。
        version: 可选版本字符串；``None`` 表示取该前缀下最新文件。
        base_dir: 模型根目录。
        map_location: 传给 ``torch.load`` 的设备映射（如 ``'cpu'``）。

    Returns:
        检查点中的整数版本号；文件不存在或格式非法时返回 ``None``。
    """
    file_path = get_model_state_path(prefix, version, base_dir)
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        checkpoint = torch.load(file_path, map_location=map_location)
        if 'state_dict' not in checkpoint or 'version' not in checkpoint:
            raise ValueError("Invalid checkpoint format")
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint['version']
    except (IOError, RuntimeError, ValueError, pickle.UnpicklingError) as e:
        print(f"Load model failed: {str(e)}")
        return None


def save_model(
    model: torch.nn.Module,
    prefix: str,
    version: str,
    base_dir: str = 'models',
    max_versions: int = 5,
) -> Optional[str]:
    """将模型状态原子写入磁盘，并可限制保留的历史文件数量。

    Args:
        model: 要保存的模块。
        prefix: 与目录名、文件名前缀一致。
        version: 版本字符串，写入检查点元数据。
        base_dir: 模型根目录。
        max_versions: 保留的 ``.td`` 文件数量上限；超过则删除最旧文件。``0`` 表示不限制。

    Returns:
        成功时为最终 ``.td`` 路径；失败时为 ``None``。
    """
    save_dir = os.path.normpath(os.path.join(base_dir, prefix))
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"{prefix}_{version}_{timestamp}.td"
    save_path = os.path.join(save_dir, filename)
    tmp_path = save_path + '.tmp'
    checkpoint = {
        'state_dict': model.state_dict(),
        'version': version,
        'timestamp': time.time(),
    }
    try:
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, save_path)
        if max_versions > 0:
            existing = sorted(
                glob.glob(os.path.join(save_dir, f"{prefix}_*.td")),
                key=os.path.getmtime, reverse=True
            )
            for old_file in existing[max_versions:]:
                os.remove(old_file)
        return save_path
    except (IOError, RuntimeError) as e:
        print(f"Save model failed: {str(e)}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None
