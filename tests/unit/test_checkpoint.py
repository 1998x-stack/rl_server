# -*- coding: utf-8 -*-
"""Tests for rl_server.utils.checkpoint module."""
import os
import time
import glob
import torch
import torch.nn as nn
import pytest

from rl_server.utils.checkpoint import save_model, load_model, get_model_state_path


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestCheckpoint:

    def test_save_and_load_roundtrip(self, tmp_path):
        net = SimpleNet()
        orig_state = {k: v.clone() for k, v in net.state_dict().items()}
        save_path = save_model(net, 'test', '1', base_dir=str(tmp_path))
        assert save_path is not None

        net2 = SimpleNet()
        version = load_model(net2, 'test', '1', base_dir=str(tmp_path))
        assert version == '1'
        for key in orig_state:
            assert torch.allclose(net2.state_dict()[key], orig_state[key])

    def test_save_creates_file(self, tmp_path):
        net = SimpleNet()
        save_path = save_model(net, 'test', '1', base_dir=str(tmp_path))
        assert save_path is not None
        assert os.path.exists(save_path)

    def test_load_nonexistent_returns_none(self, tmp_path):
        net = SimpleNet()
        result = load_model(net, 'nonexistent', base_dir=str(tmp_path))
        assert result is None

    def test_save_atomic_no_tmp_leftover(self, tmp_path):
        net = SimpleNet()
        save_model(net, 'test', '1', base_dir=str(tmp_path))
        tmp_files = glob.glob(os.path.join(str(tmp_path), '**', '*.tmp'), recursive=True)
        assert len(tmp_files) == 0

    def test_version_retention_policy(self, tmp_path):
        net = SimpleNet()
        for i in range(7):
            save_model(net, 'test', str(i), base_dir=str(tmp_path), max_versions=3)
            time.sleep(0.05)  # ensure different timestamps
        remaining = glob.glob(os.path.join(str(tmp_path), 'test', 'test_*.td'))
        assert len(remaining) == 3

    def test_load_specific_version(self, tmp_path):
        net = SimpleNet()
        # Modify weights between saves
        save_model(net, 'test', '1', base_dir=str(tmp_path), max_versions=10)
        time.sleep(0.05)
        nn.init.ones_(net.fc.weight)
        save_model(net, 'test', '2', base_dir=str(tmp_path), max_versions=10)

        net_loaded = SimpleNet()
        version = load_model(net_loaded, 'test', '1', base_dir=str(tmp_path))
        assert version == '1'

    def test_get_model_state_path_returns_latest(self, tmp_path):
        net = SimpleNet()
        save_model(net, 'test', '1', base_dir=str(tmp_path), max_versions=10)
        time.sleep(0.05)
        path2 = save_model(net, 'test', '2', base_dir=str(tmp_path), max_versions=10)
        latest = get_model_state_path('test', base_dir=str(tmp_path))
        assert latest == path2

    def test_load_invalid_checkpoint_returns_none(self, tmp_path):
        # Create a corrupt file
        save_dir = os.path.join(str(tmp_path), 'test')
        os.makedirs(save_dir)
        corrupt_path = os.path.join(save_dir, 'test_1_20260101000000.td')
        with open(corrupt_path, 'wb') as f:
            f.write(b'not a valid checkpoint')
        net = SimpleNet()
        result = load_model(net, 'test', '1', base_dir=str(tmp_path))
        assert result is None
