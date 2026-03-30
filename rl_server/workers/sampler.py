# -*- coding: utf-8 -*-
"""
Sampler worker for collecting environment experience.
Based on train_main_local/sampler.py with updated imports.
"""
import time
import queue
import torch.nn as nn
import torch.multiprocessing as mp

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed
from rl_server.algorithms import create_agent


class SamplerWorker:
    """
    Sampler worker that runs in a separate process.
    Collects experience from environments and pushes to sample_queue.
    """

    def __init__(self, idx, model_dict, share_model: nn.Module, sample_queue, env_name, log: Log):
        self.sampler_id = idx
        self.model_dict = model_dict
        self.share_model = share_model
        self.env_name = env_name
        self.log = log
        self.sample_queue = sample_queue
        self.process = None

    def process_function(self):
        setup_seed()
        sample_agent = create_agent(self.env_name, self.share_model)

        while True:
            if self.model_dict['is_exit']:
                break

            try:
                exps_list = sample_agent.sample_multi_envs(self.model_dict)

                if exps_list is not None:
                    for exps in exps_list:
                        exps_info = dict()
                        exps_info['sample_version'] = self.model_dict['TRAIN_VERSION']
                        exps_info['exps'] = exps
                        self.sample_queue.put(exps_info)
                else:
                    self.log.log_info("sample_agent.sample_multi_envs return None")

                time.sleep(0)

            except queue.Full:
                time.sleep(5)
                continue
            except Exception:
                self.log.log_exception()
                continue

        try:
            del sample_agent
        except Exception:
            self.log.log_exception()

        self.log.log_info(f'exit sampler processid {self.process.pid} samplerid {self.sampler_id}')

    def run(self):
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start sampler processid {self.process.pid} samplerid {self.sampler_id}')

    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except Exception:
            self.log.log_exception()
