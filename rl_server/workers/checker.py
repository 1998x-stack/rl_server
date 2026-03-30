# -*- coding: utf-8 -*-
"""
Checker worker for model evaluation.
Based on check_main/checker.py with updated imports.
"""
import time
import torch.nn as nn
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed
from rl_server.algorithms import create_net, create_agent


class CheckerWorker:
    """
    Checker worker that runs in a separate process.
    Periodically evaluates the model in an environment and logs metrics.
    """

    def __init__(self, model_dict, share_model: nn.Module, env_name, log: Log):
        self.model_dict = model_dict
        self.share_model = share_model
        self.self_version = model_dict['TRAIN_VERSION']
        self.env_name = env_name
        self.log = log
        self.process = None
        self.comment = "checker"

    def process_function(self):
        setup_seed()
        check_net = create_net(self.env_name)
        check_agent = create_agent(self.env_name, check_net, is_checker=True)
        writer = SummaryWriter(comment=self.comment)
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                check_net.load_state_dict(self.share_model.state_dict())
                check_net.train(False)
                self.self_version = self.model_dict['TRAIN_VERSION']
                infos = check_agent.check_single_env()
                if isinstance(infos, dict):
                    for (key, value) in infos.items():
                        writer.add_scalar(key, value, self.self_version)
                time.sleep(0)
            except Exception:
                self.log.log_exception()
                continue
        writer.close()

        self.log.log_info(f'exit checker processid {self.process.pid}')

    def run(self, comment: str = None):
        if comment is not None:
            self.comment = comment
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start checker processid {self.process.pid}')

    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except Exception:
            self.log.log_exception()
