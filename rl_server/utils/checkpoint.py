# -*- coding: utf-8 -*-
"""Model checkpoint management."""
import os
import time
import glob
import torch
from typing import Optional, Dict


def get_model_state_path(prefix: str, version: Optional[str] = None, base_dir: str = 'models') -> Optional[str]:
    model_dir = os.path.normpath(os.path.join(base_dir, prefix))
    pattern = f"{prefix}_{version}_*.td" if version else f"{prefix}_*.td"
    candidates = glob.glob(os.path.join(model_dir, pattern))
    if version:
        candidates = [f for f in candidates if os.path.basename(f).startswith(f"{prefix}_{version}_")]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime, default=None)


def load_model(model: torch.nn.Module, prefix: str, version: Optional[str] = None,
               base_dir: str = 'models', map_location: Optional[str] = None) -> Optional[int]:
    file_path = get_model_state_path(prefix, version, base_dir)
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        checkpoint = torch.load(file_path, map_location=map_location)
        if 'state_dict' not in checkpoint or 'version' not in checkpoint:
            raise ValueError("Invalid checkpoint format")
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint['version']
    except (IOError, RuntimeError, ValueError) as e:
        print(f"Load model failed: {str(e)}")
        return None


def save_model(model: torch.nn.Module, prefix: str, version: str,
               base_dir: str = 'models', max_versions: int = 5) -> Optional[str]:
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
