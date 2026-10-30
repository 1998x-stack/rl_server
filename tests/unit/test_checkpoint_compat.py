"""Checkpoint compatibility: structural additions must not brick a resume."""
import torch

from rl_server.utils import checkpoint
from rl_server.utils.logging import Log


def test_load_model_falls_back_to_non_strict_when_keys_are_missing(tmp_path, monkeypatch):
    log = Log("unit_checkpoint_compat", base_dir=str(tmp_path))
    monkeypatch.setattr(checkpoint, "_checkpoint_log", log)

    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    old_state = {k: v.clone() for k, v in model.state_dict().items()}
    # 模拟"老检查点"：少了一个后来新增的参数（如 SAC 的 log_alpha）。
    model.register_parameter("log_alpha", torch.nn.Parameter(torch.zeros(())))
    base_dir = tmp_path / "models"
    save_dir = base_dir / "compat_net"
    save_dir.mkdir(parents=True)
    torch.save({"version": 11, "state_dict": old_state},
               save_dir / "compat_net_11_1.td")

    loaded = checkpoint.load_model(model, "compat_net", "11", base_dir=str(base_dir))
    log.close()

    assert loaded == 11
    content = (tmp_path / "unit_checkpoint_compat" / "unit_checkpoint_compat.log").read_text()
    assert "checkpoint loaded partially" in content
    assert "log_alpha" in content
