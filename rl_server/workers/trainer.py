# -*- coding: utf-8 -*-
"""
Trainer worker for computing gradients from experience.
Based on train_main_local/trainer.py with updated imports.
"""
import time
import queue
from typing import Dict

import torch.nn as nn
import torch.multiprocessing as mp

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed
from rl_server.algorithms import create_calculate


class TrainerWorker:
    """
    Trainer worker that runs in a separate process.
    Consumes samples from sample_queue, computes gradients, and pushes to grads_queue.
    """

    def __init__(self, idx: int, model_dict: Dict, share_model: nn.Module,
                 sample_queue: mp.Queue, grads_queue: mp.Queue, env_name: str, log: Log):
        self.log = log
        self.process = None
        self.trainer_id = idx
        self.env_name = env_name

        self.model_dict = model_dict
        self.share_model = share_model

        self.sample_queue = sample_queue
        self.grads_queue = grads_queue

    def process_function(self):
        setup_seed()
        calculate = create_calculate(self.env_name, self.share_model)
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                samples_info = self.sample_queue.get()
                grads_list, train_version = calculate.generate_grads(samples_info['exps'], self.model_dict)

                for grads in grads_list:
                    grads_info = dict()
                    grads_info['grads'] = grads
                    grads_info['grads_version'] = train_version
                    grads_info['sample_version'] = samples_info['sample_version']
                    self.grads_queue.put(grads_info)

                time.sleep(0)

            except queue.Full:
                time.sleep(1)
                continue
            except queue.Empty:
                continue
            except Exception:
                self.log.log_exception()
                continue
        self.log.log_info(f'exit trainer processid {self.process.pid} trainerid {self.trainer_id}')

    def run(self):
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start trainer processid {self.process.pid} trainerid {self.trainer_id}')

    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except Exception:
            self.log.log_exception()
