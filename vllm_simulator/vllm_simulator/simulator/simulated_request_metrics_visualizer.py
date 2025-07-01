from dataclasses import asdict
import json
from typing import List

from vllm_simulator.simulator.simulated_request import SimulatedRequest

from utils import configure_logging

# logger = configure_logging(__name__)
logger = configure_logging(__name__, log_to_file=True, log_file_path="outputs/logs/vllm_simulator.log")


class SimulatedRequestMetricsVisualizer:
    '''Enables visualization of the filled in RequestMetrics after a simulation run.'''

    def __init__(self, sorted_requests: List[SimulatedRequest]):
        '''Requests should be sorted from earliest to latest arrived for most coherent visualization.'''
        self.requests: List[SimulatedRequest] = sorted_requests

    def log_metrics(self, only_finished_requests:bool=False) -> None:
        '''Logs RequestMetrics of all requests.
        
        Set only_finished_requests=True if only metrics of finished requests are desired.'''
        finished_request_str = " of finished requests only" if only_finished_requests else ""
        logger.info(f"Logging request metrics{finished_request_str}.")
        for request in self.requests:
            if only_finished_requests and not request.finished():
                continue
            logger.info(request.get_metrics())

    def dump_metrics_json(self, metrics_file_path: str, only_finished_requests:bool=False) -> None:
        logger.info(f"Dumping RequestMetrics objects to JSON at {metrics_file_path}.")
        
        # Convert requests into JSON serializable object.
        data = []
        for request in self.requests:
            if only_finished_requests and not request.finished():
                continue
            logger.info(request.get_metrics())
        
            # Parse request into a dict and append it to data.
            data.append(request.get_metrics_dict())

        with open(metrics_file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        logger.info(f"Finished dumping RequestMetrics objects JSON to {metrics_file_path}.")

    def to_request_dict(self) -> dict:
        """Used in MetricsVerifier to extract relevant timestamps."""
        request_dict = {}
        for request in self.requests:
            logger.info(request.name)
            request_dict[request.name] = request.get_metrics_dict()
        return request_dict
