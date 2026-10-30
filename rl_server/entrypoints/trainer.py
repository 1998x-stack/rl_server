import collections
import argparse
import os
import time

from rl_server.algorithms import create_calculate, create_net
from rl_server.config.loader import load_config
from rl_server.transport.redis_cache import RedisCache
from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_signal_handlers, should_exit


def _connection_config(section):
    return {
        'ip': section.get('host', 'localhost'),
        'port': str(section.get('port', 6379)),
        'db': str(section.get('db', 0)),
        'pw': section.get('password', ''),
        'namespace': section.get('namespace', ''),
    }


class RedisTrainer:
    def __init__(self, env_name, redis_config, log):
        self.env_name = env_name
        self.log = log
        self.model = create_net(env_name)
        self.calculator = create_calculate(env_name, self.model)
        self.model_cache = RedisCache(log, _connection_config(redis_config['model']))
        self.exps_cache = RedisCache(log, _connection_config(redis_config['exps']))
        self.grads_cache = RedisCache(log, _connection_config(redis_config['grads']))
        self.model_version = None

    def run_once(self):
        published_version = self.model_cache.get_train_version()
        if published_version is None:
            return False
        if published_version != self.model_version:
            loaded_version = self.model_cache.get_train_snapshot(self.model)
            if loaded_version is None:
                return False
            self.model_version = loaded_version

        experiences, sample_version = self.exps_cache.pop_exps()
        if experiences is None:
            return False
        gradients, _ = self.calculator.generate_grads(
            experiences, {'TRAIN_VERSION': self.model_version}
        )
        for gradient in gradients:
            if not self.grads_cache.push_grads(
                gradient, self.model_version, sample_version
            ):
                raise RuntimeError('Unable to publish gradients')
        return True

    def run(self):
        self.log.log_info(
            f"redis trainer running env={self.env_name} pid={os.getpid()}"
        )
        started = time.time()
        stats = collections.Counter()
        while not should_exit():
            try:
                if self.model_cache.get_exit_flag() == 1:
                    self.log.log_info("redis trainer exit flag received")
                    break
                if self.run_once():
                    stats['gradients'] += 1
                    elapsed = max(time.time() - started, 1e-6)
                    self.log.info_throttled(
                        "redis_trainer_progress",
                        f"redis trainer progress version={self.model_version} "
                        f"batches={stats['gradients']} "
                        f"batches_per_sec={stats['gradients'] / elapsed:.2f}",
                        interval=30,
                    )
                else:
                    time.sleep(0.1)
            except Exception:
                # 以前 run_once 出异常会直接冒泡退出，“trainer 静默消失”就是这么来的。
                stats['errors'] += 1
                self.log.exception("redis trainer loop")
                time.sleep(1)

        elapsed = max(time.time() - started, 1e-6)
        self.log.log_info(
            f"redis trainer exited batches={stats['gradients']} errors={stats['errors']} "
            f"elapsed={elapsed:.0f}s"
        )
        self.log.summary({'version': self.model_version, 'batches': stats['gradients']})


def main():
    parser = argparse.ArgumentParser(description='RL Server Redis Trainer')
    parser.add_argument('--config', default=None)
    parser.add_argument('--override', default=None)
    parser.add_argument('--env-name', default=None)
    args = parser.parse_args()

    setup_seed()
    setup_signal_handlers()
    config_path = args.config or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'default.yaml'
    )
    config = load_config(config_path, args.override)
    env_name = args.env_name or config.get('training', {}).get('env_name', 'DQNGymClassic')
    log = Log.from_config(config, 'redis_trainer')
    log.log_info(f"start env={env_name} pid={os.getpid()}")
    trainer = RedisTrainer(env_name, config['redis'], log)
    trainer.run()


if __name__ == '__main__':
    main()
