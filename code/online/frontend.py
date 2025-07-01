import copy
import signal
import sys
import ray

from online.consumer import Consumer
from online.producer import Producer
from online.gpu_server import GpuServer, MultiGpuServer
from utils.logger_setup import setup_logger


class Frontend:

    def __init__(self, gear_plan, model_dict, num_gpus=-1, log_prefix="run", qps_fac=1, max_bs=1000, max_queue_len=8, qps_polling_interval=0.1):
        """
        Frontend to Cascade Serve.
        :param gear_plan:
        :param model_dict:
        :param log_prefix: Prefix to log files of producer and consumer (logs entry and exit times of samples).
        """
        # Get logger.
        self.logger = setup_logger()
        self.logger.info("Starting frontend.")

        # Spawn GPU servers.
        self.gpu_servers = []
        placement = gear_plan.model_placement
        #for server_id, models in enumerate(placement):

        for server_id in range(num_gpus):
            self.logger.debug(f"launch server {server_id}")

            models = placement[0]

            #models = ["llama_03b_repl_0", "llama_03b_repl_1", "llama_13b_repl_0", "llama_13b_repl_1", "llama_70b"]
            #models = ["llama_03b_repl_0", "llama_03b_repl_1", "llama_07b", "llama_13b", "llama_70b"]
            #models = ["llama_13b_repl_0", "llama_13b_repl_1"]

            # Create with all modes placed on the server.
            server_model_dict = {}
            for m in models:
                server_model_dict[m] = copy.deepcopy(model_dict[m])

            # Spawn GPU server process.
            self.gpu_servers.append(GpuServer.remote(server_id=server_id,
                                                          gear_plan=gear_plan,
                                                          models=server_model_dict,
                                                          qps_fac=qps_fac,
                                                          max_bs=max_bs,
                                                          max_queue_len=max_queue_len))

        ray.get([s.alive.remote() for s in self.gpu_servers])

        # Spawn producer and consumer.
        self.producer = Producer.remote(self.gpu_servers, gear_plan, log_file_prefix=log_prefix, qps_polling_interval=qps_polling_interval)
        self.consumer = Consumer.remote(self.gpu_servers, log_file_prefix=log_prefix)
        ray.get(self.producer.alive.remote())
        ray.get(self.consumer.alive.remote())

        # Shut down on Ctrl+C.
        signal.signal(signal.SIGINT, self.shutdown)


    def warmup(self, samples, num_runs=2):
        """
        Warmup all servers.
        :param samples:
        :param num_runs: Number of times models should perform inference on samples
        :return:
        """
        self.logger.debug("Warming up ...")
        warmup_futures = []

        for s in self.gpu_servers:
            warmup_futures.append(s.warmup.remote(samples, num_runs))

        ray.get(warmup_futures)


    def infer(self, sample):
        return self.producer.infer.remote(sample)


    def set_qps(self, qps):
        """
        For debugging.
        """
        self.gpu_servers[0].set_qps_debug.remote(qps)


    def shutdown(self, signum=None, frame=None):
        print('Shutting down ...')
        ray.get(self.producer.close.remote())
        ray.get(self.consumer.close.remote())
        for s in self.gpu_servers:
            ray.get(s.close.remote())
        ray.shutdown()
        sys.exit(0)
