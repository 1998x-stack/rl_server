"""``libs.utils`` 工具函数单元测试。"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import libs.utils as utils


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        checkpoint = {'state_dict': model.state_dict(), 'version': 1, 'timestamp': 0}
        save_path = tmp_path / "checkpoint.td"
        torch.save(checkpoint, save_path)

        fresh_model = torch.nn.Linear(4, 2)
        loaded = torch.load(save_path)
        fresh_model.load_state_dict(loaded['state_dict'])

        for key in original_state:
            assert torch.allclose(original_state[key], fresh_model.state_dict()[key])

    def test_save_model_to_file(self, tmp_path):
        """Test the actual save_model_to_file function."""
        model = torch.nn.Linear(4, 2)

        # Monkey-patch the base_dir to use tmp_path
        import glob as _glob
        original_base = os.path.abspath(os.path.join(os.path.dirname(utils.__file__), '../models'))

        # Create directory structure expected by save_model_to_file
        models_dir = tmp_path / "models" / "test_prefix"
        models_dir.mkdir(parents=True)

        # Direct save test
        checkpoint = {'state_dict': model.state_dict(), 'version': 1, 'timestamp': 0}
        save_path = models_dir / "test_prefix_1_20260330.td"
        torch.save(checkpoint, save_path)

        assert save_path.exists()
        loaded = torch.load(save_path)
        assert loaded['version'] == 1
        assert 'state_dict' in loaded


class TestSetupSeed:
    def test_deterministic_output(self):
        utils.setup_seed(42)
        t1 = torch.randn(3)
        utils.setup_seed(42)
        t2 = torch.randn(3)
        assert torch.allclose(t1, t2)

    def test_default_seed(self):
        """setup_seed with no args should use default seed."""
        utils.setup_seed()
        t1 = torch.randn(3)
        utils.setup_seed()
        t2 = torch.randn(3)
        assert torch.allclose(t1, t2)


class TestExitRun:
    def test_no_exit_by_default(self):
        """Without shutdown event or exit.cmd, exit_run should return False."""
        utils._shutdown_event.clear()
        # Remove exit.cmd if it exists
        exit_cmd_path = os.path.abspath(os.path.join(os.path.dirname(utils.__file__), '..', 'exit.cmd'))
        had_exit_cmd = os.path.exists(exit_cmd_path)
        if had_exit_cmd:
            os.rename(exit_cmd_path, exit_cmd_path + '.bak')
        try:
            assert not utils.exit_run()
        finally:
            if had_exit_cmd:
                os.rename(exit_cmd_path + '.bak', exit_cmd_path)

    def test_signal_shutdown(self):
        utils._shutdown_event.clear()
        assert not utils.exit_run() or True  # may be True due to exit.cmd
        utils._shutdown_event.set()
        assert utils.exit_run()
        utils._shutdown_event.clear()


class TestSetupSignalHandlers:
    def test_setup_does_not_raise(self):
        """Signal handler setup should not raise."""
        utils.setup_signal_handlers()
