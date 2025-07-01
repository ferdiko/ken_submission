import asyncio
import math
import time

import numpy as np
import ray
from utils.logger_setup import setup_logger


# QPS_POLLING_INTERVAL = 0.02

@ray.remote
class Producer:

    def __init__(self, gpu_servers, gear_plan, log_file_prefix="log", qps_polling_interval=0.1):
        """
        The producer dispatches sample to the GPU servers. This involves load balancing
        and checking if the requested model if loaded on a server.
        :param gpu_servers:
        :param gear_plan:
        :param log_file_prefix:
        """
        self.logger = setup_logger()
        self.log_file = open(f"{log_file_prefix}_producer.txt", "w+")

        # Sample routing.
        self.gpu_servers = gpu_servers
        self.id_counter = 0

        # Load balancing state. TODO: Read from gear plan
        self.gear_plan = gear_plan
        cur_gear = gear_plan.gears[-1]
        self.last_poll = time.time()
        self.query_counter = 0
        self.target_load = np.zeros(len(gpu_servers))
        self.cur_load = np.ones(len(gpu_servers))

        # Start polling for gear switches.
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.poll_gear_switch(qps_polling_interval))



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
        self.log_file.close()


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

        self.query_counter += num_samples

        # 2. Load balancing: Determine GPU server.
        norm_cur_load = self.cur_load / np.sum(self.cur_load)
        server_id = np.argmax(self.target_load - norm_cur_load)
        self.cur_load[server_id] += 1

        # 3. Add to queue 0 of GPU server.
        # self.logger.debug(f"Sent to GPU server: {sample_ids} ({time.time()-arrival_time})")
        self.gpu_servers[server_id].add_to_queue.remote(sample_ids=sample_ids, samples=sample, queue_idx=0)

        return sample_ids


    async def poll_gear_switch(self, qps_polling_interval):
        """
        Periodically measure QPS and switch gears.
        :return:
        """
        while True:
            # Measure QPS
            cur_time = time.time()
            measurement_interval =  cur_time - self.last_poll
            qps = math.ceil(self.query_counter / measurement_interval)
            self.logger.debug(f"Measured QPS: {qps}")

            self.last_poll = cur_time
            self.query_counter = 0

            # Switch gear in GPU servers.
            for server in self.gpu_servers:
                server.switch_gear.remote(qps=qps)

            # Switch gear in producer.
            #new_gear = self.gear_plan.gears[qps]
            #entry_model = new_gear.models[0]
            #self.target_load = new_gear.load_assignment[entry_model]
            #self.cur_load = np.zeros(len(new_gear.models))

            await asyncio.sleep(qps_polling_interval)
