import asyncio
import math
import time

from collections import defaultdict

import numpy as np
import ray
from utils.logger_setup import setup_logger


@ray.remote
class Producer:

    def __init__(self, gpu_servers, packing_factor, log_file_prefix="log"):
        """

        :param gpu_servers:
        :param workload_predictions: list with prediction for each interval
        :param log_file_prefix:
        :param planning_interval:
        """
        # Set up logging.
        self.logger = setup_logger()
        self.log_file = open(f"{log_file_prefix}_producer.txt", "w+")
        self.cost_file = open(f"{log_file_prefix}_cost.txt", "w+")

        # Get asyncio loop.
        self.loop = asyncio.get_running_loop()

        # State.
        self.model_counts = []
        self.gpu_servers = gpu_servers
        self.model_on_server = [None] * len(gpu_servers)
        self.num_model_replicas = defaultdict(int)
        self.server_load = np.ones(len(gpu_servers))
        self.id_counter = 0

        self.packing_factor = packing_factor


    def bootstrap(self, load_models):
        # Load models.
        for m in load_models:
            self.add_replicas(m, 1, blocking=True)

        self.log_gpus_used()

        return 0


    def alive(self):
        """
        Check if actor alive.
        :return:
        """
        return 1


    def close(self):
        """
        FLush and close log file
        :return:
        """
        self.log_gpus_used()
        self.log_file.close()
        self.cost_file.close()


    def ensemble_selection(self):
        # How does cocktail get their accuracy online?
        # TODO
        raise NotImplementedError


    def remove_replicas(self, model, n_remove):
        server_idx = 0
        num_servers = len(self.gpu_servers)
        while n_remove > 0 and server_idx < num_servers:
            if self.model_on_server[server_idx] == model:
                self.logger.debug(f"Shut down server {server_idx}")
                self.model_on_server[server_idx] = None
                self.num_model_replicas[model] -= 1
                n_remove -= 1
                self.gpu_servers[server_idx].shutdown.remote()

            server_idx += 1

        assert n_remove == 0


    async def spin_up(self, server_id, model):
        self.logger.debug(f"Spin up server {server_id}")
        await self.gpu_servers[server_id].load_model.remote(model)
        self.model_on_server[server_id] = model
        self.num_model_replicas[model] += 1

        # Make sure this server doesn't need to "catch up" to the others
        self.server_load[server_id] = np.max(self.server_load)


    def spin_up_blocking(self, server_id, model):
        ray.get(self.gpu_servers[server_id].load_model.remote(model))
        self.model_on_server[server_id] = model
        self.num_model_replicas[model] += 1

        # Make sure this server doesn't need to "catch up" to the others
        self.server_load[server_id] = np.max(self.server_load)


    def add_replicas(self, model, n_add, blocking=False):
        server_idx = 0
        num_servers = len(self.gpu_servers)
        while n_add > 0 and server_idx < num_servers:
            if self.model_on_server[server_idx] is None:
                self.model_on_server[server_idx] = "starting"
                if not blocking:
                    self.loop.create_task(self.spin_up(server_idx, model))
                else:
                    self.spin_up_blocking(server_idx, model)

                n_add -= 1

            server_idx += 1

        if n_add > 0:
            self.logger.error(f"Scaling failed for model {model}: Requested {n_add} more instances")


    def log_gpus_used(self):
        # Log how many GPUs are in use.
        gpus_used = sum([1 for m in self.model_on_server if m is not None])
        self.cost_file.write(f"{time.time()},{gpus_used}\n")
        self.logger.debug(f"Scaled to {gpus_used} instances")


    def autoscale(self, desired_scale):
        """
        Scale running model replicas.
        :param scale_factor:
        :return:
        """
        # Get running models and scale them linearly.
        # NOTE: Since we always use the same SLOs / workload, we don't need the
        #  importance sampling and can scale uniformly.
        for m in self.num_model_replicas:
            cur_replicas = self.num_model_replicas[m]
            # next_replicas = round(cur_replicas*scale_factor)
            # TODO:
            if cur_replicas > 0:
                next_replicas = desired_scale
            else:
                next_replicas = 0

            # Add/remove replicas
            if cur_replicas > next_replicas:
                self.remove_replicas(m, cur_replicas-next_replicas)
            elif cur_replicas < next_replicas:
                self.add_replicas(m, next_replicas-cur_replicas)

        self.log_gpus_used()


    def replan(self, predicted_workload):
        """
        We trigger the replan from the workload runner: Since we give Cocktail the
        ground-truth workload forecast, we need to perfectly sync it with the runner.
        :param predicted_workload:
        :return:
        """
        self.logger.debug(f"In replan. Current: {self.model_on_server}, predicted workload: {predicted_workload}")

        # Reset server load statistics.
        self.server_load = np.ones(len(self.gpu_servers))

        # TODO: Get scaling factor. We need to know queries per planning interval?
        # NOTE: Since we always use the same SLOs / workload, we don't need the
        #  importance sampling and can scale uniformly.
        desired_scale = math.ceil(predicted_workload / self.packing_factor)

        # 1. Update ensemble.
        # TODO

        # 2. Autoscale.
        self.autoscale(desired_scale)


    def infer(self, sample):
        """
        Assign id to sample, load balance and send to GPU server.
        :param sample: any; the sample (e.g. text)
        :return: int; sample id
        """
        # 1. Assign id and log arrival time.
        arrival_time = time.time()

        # TODO: This is BERT specific
        num_samples = sample[0].shape[0]

        sample_ids = []
        for _ in range(num_samples):
            sample_id = self.id_counter
            sample_ids.append(sample_id)
            self.log_file.write(f"{arrival_time},{sample_id}\n")
            self.id_counter += 1

        # 2. Load balancing: Determine GPU server.
        _, server_id = min([(self.server_load[i], i) for i, m in enumerate(self.model_on_server) if m is not None and m != "starting"])
        self.server_load[server_id] += 1

        # 3. Add to queue of GPU server.
        self.gpu_servers[server_id].add_to_queue.remote(sample_ids=sample_ids, samples=sample)
        self.logger.debug(f"Sent to server {server_id}: {sample_ids}")

        return sample_ids
