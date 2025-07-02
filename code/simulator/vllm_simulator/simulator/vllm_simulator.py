from collections import deque
from typing import List, Optional

from vllm_simulator.profiler.vllm_complete_profile import VllmCompleteProfile
from vllm_simulator.simulator.cache_simulator import CacheSimulatorConfig
from vllm_simulator.simulator.simulated_request import SimulatedRequest
from vllm_simulator.simulator.simulated_request_metrics_visualizer import SimulatedRequestMetricsVisualizer
from vllm_simulator.simulator.request_scheduler import RequestScheduler
from utils.simulator_helpers import validate_simulated_request_input

from utils import configure_logging

logger = configure_logging(__name__, log_to_file=True, log_file_path="outputs/logs/vllm_simulator.log")


class VllmSimulator:
    def __init__(self, vllm_complete_profile: VllmCompleteProfile) -> None:
        '''Initializes a VllmSimulator instance with its VllmProfile.'''
        self.vllm_complete_profile = vllm_complete_profile
        self.max_batch_tokens = vllm_complete_profile.max_num_batched_tokens

        self.cache_simulator_config = CacheSimulatorConfig(
            self.vllm_complete_profile.block_size,
            self.vllm_complete_profile.num_gpu_blocks,
            self.max_batch_tokens,
            self.vllm_complete_profile.num_watermark_blocks)

        logger.info("Initialized " + str(self))

    def simulate_requests(self, in_tokens: List[int], out_tokens: List[int], queries_per_iter: List[int], query_ids: Optional[List[str]]=None)-> SimulatedRequestMetricsVisualizer:
        '''Returns forward pass runtimes of all requests and logs breakdown of when requests start and finish.
        
        Uses discrete events to move time forward.'''
        validate_simulated_request_input(in_tokens, out_tokens, queries_per_iter)

        requests = []
        dummy_arrival_timestamp = 0
        for i in range(len(in_tokens)):
            num_in, num_out = in_tokens[i], out_tokens[i]
            query_id = "" if query_ids is None else query_ids[i]
            requests.append(SimulatedRequest(num_in, num_out, dummy_arrival_timestamp, query_id))

        return self.simulate_requests_per_iter(requests, queries_per_iter)

    def simulate_requests_per_iter(self, requests: List[SimulatedRequest], queries_per_iter: List[int]) -> SimulatedRequestMetricsVisualizer:
        '''Outputs an event token verification log for TokenVerifier.
        
        Replicates simulate_requests exactly.'''
        logger.info("simulate_requests for requests=" + str(requests))
        # Initialize scheduler. Scheduler starts at arbitrary time (not useful for token verification)
        request_scheduler = RequestScheduler(requests, self.cache_simulator_config)

        forward_pass_index = 0

        # While requests are not done being processed, iteratively determine tokens in each forward pass and its runtime.
        while True:
            # If all requests have finished being processed, simulation is finished.
            if request_scheduler.all_requests_finished():
                logger.info(f"All requests finished.")
                break

            logger.debug(f"++++++++++++++++++ FORWARD PASS {forward_pass_index} BEGIN ++++++++++++++++++")
            # Add additional queries as per the input schedule if needed.
            if forward_pass_index < len(queries_per_iter):
                for _ in range(queries_per_iter[forward_pass_index]):
                    request_scheduler.move_unstarted_to_waiting_request()

            # Then, execute the next forward pass.
            self._next_forward_pass(request_scheduler, log_token_verification=True)
            logger.debug(f"++++++++++++++++++ FORWARD PASS {forward_pass_index} END ++++++++++++++++++")
            forward_pass_index += 1

        # Ensure state of scheduler and CacheSimulator instances are correct.
        request_scheduler.check_end_conditions()

        # Log time that last request finished.
        sim_end_time = request_scheduler.get_curr_time()
        logger.info(f"Simulation ends at {sim_end_time} ms.")

        # Return visualizer of request metrics.
        return request_scheduler.get_simulated_request_metrics_visualizer()
    
    def _next_forward_pass(self, request_scheduler: RequestScheduler, log_token_verification: bool=False):
        '''Executes next forward pass, using initialized RequestScheduler and runtime tracking.'''
        # Determine number of tokens in next forward pass.
        tokens_in_forward_pass, includes_chunked_prefill = request_scheduler.get_tokens_in_next_forward_pass(log_token_verification)
        logger.info(f"tokens_in_forward_pass={tokens_in_forward_pass}")

        # Determine runtime of this forward pass.
        use_cuda_graph_profile = not includes_chunked_prefill
        forward_pass_runtime = self.vllm_complete_profile.predict_forward_pass_runtime(tokens_in_forward_pass, use_cuda_graph_profile)
        logger.info(f"forward_pass_runtime={forward_pass_runtime}")

        # Finish this forward pass.
        request_scheduler.finish_forward_pass(forward_pass_runtime)

    def __str__(self):
        return "VllmSimulator{max_batch_tokens=%d}" % (self.max_batch_tokens)
    