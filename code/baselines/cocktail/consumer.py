import time

import ray
import asyncio

from utils.logger_setup import setup_logger


POLLING_INTERVAL = 0.05 #0.05

@ray.remote
class Consumer:

    def __init__(self, gpu_servers, log_file_prefix="log"):
        self.gpu_servers = gpu_servers

        # Set up logging
        self.logger = setup_logger()
        self.log_file = open(f"{log_file_prefix}_consumer.txt", "w+")

        # get asyncio loop (Ray created the loop)
        self.loop = asyncio.get_event_loop()

        # Start polling the gpu queues for triggering inference.
        num_servers = len(gpu_servers)
        for server_idx in range(num_servers):
            self.loop.create_task(self.poll_queue(server_idx))


    def alive(self):
        """
        Check if actor alive.
        :return:
        """
        return 1


    def close(self):
        self.log_file.close()


    async def poll_queue(self, server_idx):
        gpu_server = self.gpu_servers[server_idx]

        while True:
            # Poll queue.
            done_ids, done_preds = await gpu_server.check_queue.remote()

            # Write finished sample ids to file.
            if done_ids is not None:
                current_time = time.time()

                for sample_id, pred in zip(done_ids, done_preds):
                    self.log_file.write(f"{current_time},{sample_id},{pred}\n")

            # Sleep.
            await asyncio.sleep(POLLING_INTERVAL)
