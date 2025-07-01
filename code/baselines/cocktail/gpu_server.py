import copy
import ray
import torch
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from baselines.cocktail.model_queue import BertQueue, BertQueue2
from utils.logger_setup import setup_logger

import time

# ======================================================================================================================
# Single GPU server
# ======================================================================================================================
@ray.remote(num_gpus=1)
class GpuServer:

    def __init__(self, server_id, models, warmup_sample, max_bs=1000):
        """
        GpuServer manages queues and models
        :param server_id:
        :param models:
        :param max_bs: Maximum batch size
        """
        # Needed for ExLlama: Initialize torch cuda state.
        torch.cuda._lazy_init()

        self.logger = setup_logger()

        # Model.
        self.model = None
        self.model_dict = models

        # Initialize model queue.
        if max_bs == 0:
            self.queue = BertQueue()
        else:
            self.queue = BertQueue2(max_bs=max_bs)

        # State.
        self.server_id = server_id
        self.cur_batch_size = 1
        self.mark_for_shutdown = False

        # Warmup sample. TODO: Bert specific code below.
        self.warmup_sample = (warmup_sample[0].to("cuda"), warmup_sample[1].to("cuda"))
        self.warmup_sample_id = [i for i in range(warmup_sample[0].shape[0])]



    def load_model(self, model_name):
        """
        Load a model.
        :param model_name:
        :return:
        """
        start = time.time()
        self.logger.debug("loading model")
        self.model = copy.deepcopy(self.model_dict[model_name])
        self.model.load()
        self.warmup()
        self.logger.debug(f"done loading model, {time.time() - start}")

        return 0


    def warmup(self):
        self.model.forward(self.warmup_sample, self.warmup_sample_id)
        return 0


    def shutdown(self):
        # Mark for shut down but only shut down after last check queue has been
        # called to empty the queue.
        self.mark_for_shutdown = True


    def alive(self):
        """
        Check if actor alive.
        :return:
        """
        return 1


    def add_to_queue(self, sample_ids, samples):
        self.queue.add(sample_ids, samples)


    def check_queue(self):
        self.logger.debug(f"Server {self.server_id} queue: {self.queue.size()}")

        # 0. If server shut down, don't do anything.
        if self.model is None:
            return None, None

        # 1. Check if queue full enough for inference.
        if self.queue.size() < self.cur_batch_size:
            return None, None

        # 2. Predict and get certainties.
        samples, sample_ids = self.queue.get(-1) #queue.get(self.cur_batch_sizes[queue_id])
        _, preds = self.model.forward(samples, sample_ids)

        self.logger.debug(f"Server {self.server_id} done with inference: {sample_ids}")

        if self.mark_for_shutdown:
            self.model = None
            self.mark_for_shutdown = False

        return sample_ids, preds


    def close(self):
        return 0
