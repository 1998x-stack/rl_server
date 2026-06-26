# -*- coding: utf-8 -*-
"""Tests for heartbeat utilities in rl_server.utils.process."""
import os
import time

from rl_server.utils.process import write_heartbeat, cleanup_heartbeat, HEARTBEAT_DIR


class TestHeartbeat:

    def test_write_creates_file(self):
        write_heartbeat('test_worker', 0)
        path = os.path.join(HEARTBEAT_DIR, f"test_worker_0_{os.getpid()}")
        assert os.path.exists(path)
        # cleanup
        os.remove(path)

    def test_write_content_is_timestamp(self):
        write_heartbeat('test_worker', 1)
        path = os.path.join(HEARTBEAT_DIR, f"test_worker_1_{os.getpid()}")
        with open(path) as f:
            content = f.read()
        ts = float(content)
        # Should be recent (within last 5 seconds)
        assert time.time() - ts < 5
        os.remove(path)

    def test_cleanup_removes_file(self):
        write_heartbeat('test_worker', 2)
        path = os.path.join(HEARTBEAT_DIR, f"test_worker_2_{os.getpid()}")
        assert os.path.exists(path)
        cleanup_heartbeat('test_worker', 2)
        assert not os.path.exists(path)

    def test_cleanup_nonexistent_is_noop(self):
        # Should not raise
        cleanup_heartbeat('nonexistent_worker', 999)

    def test_different_workers_different_files(self):
        write_heartbeat('sampler', 0)
        write_heartbeat('trainer', 0)
        path_s = os.path.join(HEARTBEAT_DIR, f"sampler_0_{os.getpid()}")
        path_t = os.path.join(HEARTBEAT_DIR, f"trainer_0_{os.getpid()}")
        assert os.path.exists(path_s)
        assert os.path.exists(path_t)
        assert path_s != path_t
        os.remove(path_s)
        os.remove(path_t)
