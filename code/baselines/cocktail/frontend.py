import copy
import signal
import sys
import ray

from baselines.cocktail.consumer import Consumer
from baselines.cocktail.gpu_server import GpuServer
from baselines.cocktail.producer import Producer

from utils.logger_setup import setup_logger


class CocktailFrontend:

    def __init__(self, model_dict, packing_factor, warmup_sample, max_gpus=-1, log_prefix="run", max_bs=1000):
        """
        Frontend to Cascade Serve.
        :param gear_plan:
        :param model_dict:
        :param log_prefix: Prefix to log files of producer and consumer (logs entry and exit times of samples).
        """
        # Get logger.
        self.logger = setup_logger()
        self.logger.info("Starting Cocktail frontend.")

        # Spawn GPU servers.
        self.gpu_servers = []
        for server_id in range(max_gpus):
            self.logger.debug(f"Launch server {server_id}.")

            # Create with all modes placed on the server.
            server_model_dict = copy.deepcopy(model_dict)

            # Spawn GPU server process.
            self.gpu_servers.append(GpuServer.remote(server_id=server_id,
                                                     models=server_model_dict,
                                                     max_bs=max_bs,
                                                     warmup_sample=warmup_sample))

        ray.get([s.alive.remote() for s in self.gpu_servers])

        # Spawn producer and consumer.
        self.producer = Producer.remote(self.gpu_servers, packing_factor, log_prefix)
        self.consumer = Consumer.remote(self.gpu_servers, log_prefix)
        ray.get(self.producer.alive.remote())
        ray.get(self.consumer.alive.remote())

        # Shut down on Ctrl+C.
        signal.signal(signal.SIGINT, self.shutdown)


    def bootstrap(self, load_models):
        """
        Warmup all servers.
        :param samples:
        :param num_runs: Number of times models should perform inference on samples
        :return:
        """
        self.logger.debug("Bootstrapping ...")
        ray.get(self.producer.bootstrap.remote(load_models))


    def replan(self, predicted_workload):
        self.producer.replan.remote(predicted_workload)


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
