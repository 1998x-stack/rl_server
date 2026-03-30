import os
import sys
import pytest
import tempfile
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from libs.config_loader import load_config, interpolate_env_vars


class TestInterpolateEnvVars:
    def test_simple_substitution(self):
        os.environ['TEST_HOST'] = 'myhost'
        assert interpolate_env_vars('${TEST_HOST}') == 'myhost'
        del os.environ['TEST_HOST']

    def test_default_value(self):
        result = interpolate_env_vars('${NONEXISTENT_VAR:default_val}')
        assert result == 'default_val'

    def test_no_substitution(self):
        assert interpolate_env_vars('plain_string') == 'plain_string'

    def test_empty_default(self):
        result = interpolate_env_vars('${NONEXISTENT_VAR:}')
        assert result == ''

    def test_integer_conversion(self):
        os.environ['TEST_PORT'] = '6379'
        result = interpolate_env_vars('${TEST_PORT:6379}')
        assert result == '6379'
        del os.environ['TEST_PORT']


class TestLoadConfig:
    def test_load_yaml_file(self):
        config_data = {
            'redis': {'host': 'localhost', 'port': 6379},
            'training': {'env_name': 'CartPole'}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config = load_config(f.name)
            assert config['redis']['host'] == 'localhost'
            assert config['training']['env_name'] == 'CartPole'
        os.unlink(f.name)

    def test_env_var_interpolation_in_yaml(self):
        os.environ['TEST_REDIS_HOST'] = 'redis.prod.internal'
        config_data = {'redis': {'host': '${TEST_REDIS_HOST:localhost}'}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config = load_config(f.name)
            assert config['redis']['host'] == 'redis.prod.internal'
        del os.environ['TEST_REDIS_HOST']
        os.unlink(f.name)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path.yaml')

    def test_merge_configs(self):
        base = {'redis': {'host': 'localhost', 'port': 6379}, 'training': {'lr': 0.001}}
        override = {'redis': {'host': 'prod-host'}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as bf:
            yaml.dump(base, bf)
            bf.flush()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as of:
            yaml.dump(override, of)
            of.flush()
        config = load_config(bf.name, of.name)
        assert config['redis']['host'] == 'prod-host'
        assert config['redis']['port'] == 6379
        assert config['training']['lr'] == 0.001
        os.unlink(bf.name)
        os.unlink(of.name)
