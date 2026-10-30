# -*- coding: utf-8 -*-
"""模型检查点：按前缀与版本查找、加载与保存 PyTorch 状态字典。"""
import os
import time
import glob
import pickle
import torch
from typing import Optional

from rl_server.utils.logging import Log


def _default_model_dir() -> str:
    """模型根目录：``RL_SERVER_MODEL_DIR`` 优先，否则当前工作目录下的 ``models``。"""
    return os.environ.get('RL_SERVER_MODEL_DIR', 'models')

_checkpoint_log: Optional[Log] = None


def _log() -> Log:
    """惰性创建检查点专用日志（只在真正出错/写日志时才建文件）。"""
    global _checkpoint_log
    if _checkpoint_log is None:
        _checkpoint_log = Log('checkpoint')
    return _checkpoint_log


def get_model_state_path(prefix: str, version: Optional[str] = None,
                         base_dir: Optional[str] = None) -> Optional[str]:
    """在指定目录下按前缀（及可选版本）查找最新的 ``.td`` 检查点文件。

    Args:
        prefix: 模型名前缀，对应子目录 ``base_dir/prefix``。
        version: 若给定，仅匹配文件名以 ``{prefix}_{version}_`` 开头的文件。
        base_dir: 模型根目录。

    Returns:
        最近修改时间最新的匹配文件路径；若无匹配则返回 ``None``。
    """
    base_dir = base_dir or _default_model_dir()
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
    base_dir: Optional[str] = None,
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
        checkpoint = torch.load(file_path, map_location=map_location or 'cpu', weights_only=True)
        if 'state_dict' not in checkpoint or 'version' not in checkpoint:
            raise ValueError("Invalid checkpoint format")
        try:
            model.load_state_dict(checkpoint['state_dict'])
            model._checkpoint_missing_keys = []
        except RuntimeError as mismatch:
            # 网络结构新增参数（例如 SAC 的 log_alpha）后，旧检查点会缺键。
            # 这里降级为非严格加载并明确记录，而不是让整个训练起不来。
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            missing = list(getattr(result, 'missing_keys', []) or [])
            unexpected = list(getattr(result, 'unexpected_keys', []) or [])
            model._checkpoint_missing_keys = missing
            _log().warning(
                f"checkpoint loaded partially path={file_path} version={checkpoint['version']} "
                f"missing_keys={missing} unexpected_keys={unexpected} "
                f"reason={type(mismatch).__name__}"
            )
        return checkpoint['version']
    except (IOError, RuntimeError, ValueError, pickle.UnpicklingError) as e:
        # 以前只 print 到 stdout：save/load 失败在日志文件里完全不可见。
        _log().error(f"checkpoint load failed path={file_path} error={type(e).__name__}:{e}")
        return None


def save_model(
    model: torch.nn.Module,
    prefix: str,
    version: str,
    base_dir: Optional[str] = None,
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
    base_dir = base_dir or _default_model_dir()
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
        _log().error(f"checkpoint save failed prefix={prefix} version={version} error={type(e).__name__}:{e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None
