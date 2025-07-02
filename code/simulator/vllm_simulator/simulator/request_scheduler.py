from collections import deque
from dataclasses import dataclass
import json
from typing import Deque, List, Optional, Tuple

from vllm_simulator.simulator.cache_simulator import CacheSimulator, CacheSimulatorConfig
from vllm_simulator.simulator.simulated_request import SimulatedRequest
from vllm_simulator.simulator.simulated_request_metrics_visualizer import SimulatedRequestMetricsVisualizer

from utils import configure_logging

# logger = configure_logging(__name__)
logger = configure_logging(__name__, log_to_file=True, log_file_path="outputs/logs/vllm_simulator.log")
# HACK: This is using a logger/handler defined in vllm/vllm/logger.py. Should use one defined in vllm/utils.logging_setup.py.
token_verification_logger = configure_logging("sim_token_verification_logger")


@dataclass
class SchedulerOutputs:
    '''Parent class for scheduling outputs from running or waiting queues.'''
    # Requests that have contributed tokens to this forward pass.
    scheduled_requests: List[SimulatedRequest]
    # Total number of batched tokens.
    num_batched_tokens: int
    # Tokens per request included in this forward pass. Request identified by vllm name.
    request_to_tokens: dict[str, int]

    @classmethod
    def create_empty(cls):
        '''Returns an empty instance that is ready to be filled in with scheduling specifics.'''
        raise NotImplementedError("Subclass must implement this method!")
    
    def get_total_batch_tokens(self) -> int:
        raise NotImplementedError("Subclass must implement this method!")
    
    def maybe_includes_chunked_prefill(self, request_is_chunked_prefill):
        raise NotImplementedError("Subclass must implement this method!")

    def __str__(self) -> str:
        raise NotImplementedError("Subclass must implement this method!") 


@dataclass
class SchedulerRunningOutputs(SchedulerOutputs):
    '''Requests that are scheduled from the running queue.'''
    # Requests that have been preempted.
    preempted_requests: List[SimulatedRequest]
    # Whether this scheduling run includes chunked prefill. Indicates whether CUDA graph profile is used or not.
    includes_chunked_prefill: bool=False

    @classmethod
    def create_empty(cls):
        '''Returns an empty instance that is ready to be filled in with scheduling specifics.'''
        return SchedulerRunningOutputs(
            scheduled_requests=[],
            preempted_requests=[],
            num_batched_tokens=0,
            request_to_tokens={})
    
    def get_total_batch_tokens(self) -> int:
        return self.num_batched_tokens
    
    def __str__(self) -> str:
        return "SchedulerRunningOutputs{scheduled=%d, preempted=%d, num_batched_tokens=%d}" % (len(self.scheduled_requests), len(self.preempted_requests), self.num_batched_tokens)

    def maybe_includes_chunked_prefill(self, request_is_chunked_prefill):
        # Sets it to true if a scheduled request has chunked prefill.
        if request_is_chunked_prefill:
            self.includes_chunked_prefill = True


@dataclass
class SchedulerWaitingOutputs(SchedulerOutputs):
    '''Requests that are scheduled from the running or waiting queues.'''
    # Total number of running scheduled tokens. 
    num_running_batched_tokens: int
    # Whether this scheduling run includes chunked prefill. Indicates whether CUDA graph profile is used or not.
    includes_chunked_prefill: bool=False

    @classmethod
    def create_empty(cls, running_scheduled_tokens):
        '''Returns an empty instance that is ready to be filled in with scheduling specifics.'''
        return SchedulerWaitingOutputs(
            scheduled_requests=[],
            num_batched_tokens=0,
            num_running_batched_tokens=running_scheduled_tokens,
            request_to_tokens={})
    
    def get_waiting_batch_tokens(self) -> int:
        return self.num_batched_tokens

    def get_total_batch_tokens(self) -> int:
        return self.num_running_batched_tokens + self.num_batched_tokens

    def __str__(self) -> str:
        return "SchedulerWaitingOutputs{scheduled=%d, num_batched_tokens=%d}" % (len(self.scheduled_requests), self.num_batched_tokens)

    def maybe_includes_chunked_prefill(self, request_is_chunked_prefill):
        # Sets it to true if a scheduled request has chunked prefill.
        if request_is_chunked_prefill:
            self.includes_chunked_prefill = True


class RequestScheduler:
    '''Uses discrete events to track time and return which requests should start.'''
    def __init__(self, requests: List[SimulatedRequest], cache_simulator_config: CacheSimulatorConfig):
        '''Initializes RequestScheduler with the given requests.''' 
        # Sort requests with earliest arrived at the left head.
        requests.sort(key=lambda request: request.get_arrival_timestamp())

        # All requests with earliest arrived at the left head. Used for metrics visualization.
        self.all_requests = requests
        # Requests that have not yet been acknowledged by the scheduler. Initially stores all requests.
        # From highest (earliest arrived) to lowest priority.
        self.unstarted_requests: Deque[SimulatedRequest] = deque([request for request in requests])
        # Requests that are waiting to start prefill from highest to lowest priority.
        self.waiting_requests: Deque[SimulatedRequest] = deque()
        # Requests that are in prefill or decode phase from highest to lowest priority.
        self.running_requests: Deque[SimulatedRequest] = deque()
        # Set current time to the start time of the first request.
        self.curr_time = self.unstarted_requests[0].get_arrival_timestamp()
        # Tracks cache capacity.
        self.cache_simulator = CacheSimulator.from_config(cache_simulator_config)
        
        logger.info(f"Initialized RequestScheduler(start_time={self.curr_time}, num_request_groups={len(self.unstarted_requests)})")

    def get_tokens_in_next_forward_pass(self, log_token_verification: bool=False) -> Tuple[int, bool]:
        '''Returns the number of tokens in the next forward pass.
        
        Schedules requests according to the following priority: 1) running requests in 
        a) decode or b) prefill phases, 2) waiting requests that have not started prefill.
        
        Maintains state of queues and request tokens progress, but does not update time or request statuses.'''
        # logger.info("called get_tokens_in_next_forward_pass")
        # logger.info(self.get_queue_str())
        self.log_queue_lengths()

        if not log_token_verification:
            # Check if the starting times of any unstarted requests have passed (i.e. it is now a waiting request).
            self._queue_unstarted_requests()
            # logger.info("After queue_unstarted")
            # logger.info(self.get_queue_str())

        # Schedule running requests.
        running_scheduled: SchedulerRunningOutputs = self._schedule_running()
        # logger.info("After _schedule_running")
        logger.info(f"running_scheduled: {str(running_scheduled)}")
        # logger.info(f"{[request.name for request in running_scheduled.scheduled_requests]}")
        # logger.info(self.get_queue_str())
        # logger.info("After _schedule_running -> running.extendleft")
        # self.running_requests = deque(running_scheduled.scheduled_requests)
        # # self.running_requests = deque(running_scheduled.scheduled_requests) + self.running_requests
        # logger.info(self.get_queue_str())
        # self.log_queue_lengths()

        # Schedule waiting requests if possible (ie no preempted requests and batch + cache have capacity).
        waiting_scheduled = self._schedule_waiting(running_scheduled)

        # Update queues and event_tokens tracker.
        self.running_requests = deque(running_scheduled.scheduled_requests) + deque(waiting_scheduled.scheduled_requests)
        self.waiting_requests = deque(running_scheduled.preempted_requests) + self.waiting_requests
        event_tokens = running_scheduled.request_to_tokens | waiting_scheduled.request_to_tokens
        
        # Calculate total tokens in batch.
        total_batched_tokens = running_scheduled.get_total_batch_tokens() + waiting_scheduled.get_waiting_batch_tokens()

        # Log event_tokens for token verification.
        event_tokens["total_batched_tokens"] = total_batched_tokens
        if log_token_verification:
            token_verification_logger.info(json.dumps(event_tokens))

        # Note: At this point, cache state may be incorrect since FINISHED requests have
        # not been removed. This clean up is done in finish_forward_pass.

        # Return total number of tokens in this forward pass.
        includes_chunked_prefill = running_scheduled.includes_chunked_prefill or waiting_scheduled.includes_chunked_prefill
        logger.info(f"Forward pass with {total_batched_tokens} tokens; includes_chunked_prefill={includes_chunked_prefill}.")
        return total_batched_tokens, includes_chunked_prefill

    def _queue_unstarted_requests(self):
        '''Moves requests whose arrival timestamp is after curr_time into the waiting queue.'''
        while self.unstarted_requests:
            request = self.unstarted_requests[0]
            if request.get_arrival_timestamp() > self.curr_time:
                # Found the first request who has not yet arrived.
                break
            else:
                # Request has arrived. Move it to waiting queue.
                self.waiting_requests.append(request)
                request.update_request_status(self.curr_time)
                self.unstarted_requests.popleft()

    def _schedule_running(self) -> SchedulerRunningOutputs:
        '''Schedule requests that are running. 
        
        Running queue can include decode and chunked prefill requests.'''
        output = SchedulerRunningOutputs.create_empty()

        while self.running_requests:
            request = self.running_requests[0]

            # Invariant: all requests in running_requests are not yet finished (i.e. still have tokens remaining).
            assert request.prefill() or request.decoding(), "Request in running_requests must have PREFILL or DECODING status."

            if output.get_total_batch_tokens() == self.cache_simulator.max_batch_tokens:
                # No budget.
                break

            # Remove request from running queue and continue with processing this request.
            self.running_requests.popleft()

            # Vllm uses simple heuristic in BlockManagerV1 by checking if at least 1 free gpu block exists.
            while not self._can_append_slots(request):
                cont_loop = True
                # Determine which running requests to preempt.
                if self.running_requests:
                    # Preempt the most recently scheduled request.
                    victim_request = self.running_requests.pop()
                    logger.info("victim from right")
                else:
                    # No other sequence to preempt. Must preempt current sequence group.
                    victim_request = request
                    cont_loop = False
                    logger.info("victim is self")

                # Preempt the victim request.
                logger.info(f"PREEMPT Request {victim_request.name}")
                self.cache_simulator.preempt(victim_request)
                victim_request.reset()
                output.preempted_requests.append(victim_request)

                if not cont_loop:
                    # No other running_requests.
                    break
            else:
                # If request is chunked prefill, set flag as True in output.
                output.maybe_includes_chunked_prefill(request.prefill())
                # Consume request and update output.
                num_request_tokens_consumed = self._consume_request_for_forward_pass(request, output)
                # Track tokens consumed.
                output.request_to_tokens[request.name] = num_request_tokens_consumed

        return output

    def _schedule_waiting(self, running_scheduled: SchedulerRunningOutputs) -> SchedulerWaitingOutputs:
        '''Schedule requests that have not begun their prefill stage if 1) no requests preempted in _schedule_running,
        2) batch has capacity, and 3) cache has capacity.

        Preempted requests will be in this queue as well. Assumes recompute (instead of swapping).'''
        output = SchedulerWaitingOutputs.create_empty(running_scheduled.get_total_batch_tokens())

        # If requests were preempted in _schedule_running, early return.
        if running_scheduled.preempted_requests:
            return output

        # Else, try to schedule waiting requests.
        while self.waiting_requests:
            request = self.waiting_requests[0]

            # Invariant: all requests in self.queued_requests are in WAITING status.
            assert request.waiting(), "Request in queued_requests must have WAITING status."

            if output.get_total_batch_tokens() == self.cache_simulator.max_batch_tokens:
                # Early exit if batch is full.
                break

            if not self._has_prefill_cache_capacity(request):
                # Early exit if cache is full.
                break

            # Else, continue scheduling this request. Remove from waiting queue.
            self.waiting_requests.popleft()

            # Consume request and update output.
            num_request_tokens_consumed = self._consume_request_for_forward_pass(request, output)
            # Track tokens consumed.
            output.request_to_tokens[request.name] = num_request_tokens_consumed
            # Request was in prefill this stage, set flag as True in output.
            output.maybe_includes_chunked_prefill(True)

        return output

    def _consume_request_for_forward_pass(self, request: SimulatedRequest, output: SchedulerOutputs) -> int:
        '''Consumes request tokens to update cache state and current SchedulerOutputs object.
        
        Returns the number of request tokens consumed'''
        # Update the number of tokens in this forward pass. Also updates request state.
        num_request_tokens_consumed = self.cache_simulator.get_num_consumed_tokens_from_request(
            output.get_total_batch_tokens(), self.curr_time, request)
        output.num_batched_tokens += num_request_tokens_consumed

        # Add request to output.
        output.scheduled_requests.append(request)

        return num_request_tokens_consumed

    def _can_append_slots(self, request: SimulatedRequest) -> bool:
        '''Returns true if there is at least 1 free GPU block available.
        
        Replicates vllm simple heuristic: https://github.com/plantbud/vllm/blob/main/vllm/core/block_manager_v1.py#L378-L388'''
        assert request.prefill() or request.decoding(), "Request must be PREFILL or DECODING "
        return self.cache_simulator.num_total_blocks_available() > 0

    def _has_prefill_cache_capacity(self, request: SimulatedRequest) -> bool:
        '''Returns True if cache has capacity for this request.'''
        assert request.waiting(), "Request must be PREFILL if checking cache capacity"
        # Checks if GPU blocks are available to be earmarked.
        return self.cache_simulator.has_prefill_capacity(request)

    def finish_forward_pass(self, forward_pass_runtime: Optional[int]) -> None:
        '''Executes forward pass and manages state of requests and cache.'''
        # Update request_scheduler curr_time.
        self.update_time(forward_pass_runtime)
        # Update status and metrics of requests using request_scheduler.
        self.update_all_requests_status()
        # Clean cache of finished requests (before next forward pass begins).
        self.cache_simulator.clean_cache()
        # Simulates any decode tokens. These tokens are not included in the batch of the current forward pass.
        self.update_decoding_requests()

        logger.info("forward pass finished with following state:")
        self.log_queue_lengths()
        logger.info("forward pass finished with following running requests:")
        for request in self.running_requests:
            logger.info(str(request))

        self.visualize_cache()

        self._check_request_invariants()

    def update_time(self, time_to_increment: Optional[int]) -> None:
        '''Updates current time of the RequestScheduler.'''
        if time_to_increment is not None:
            self.curr_time += time_to_increment
            logger.info(f"RequestScheduler time updated to {self.curr_time}")

    def update_decoding_requests(self) -> None:
        '''Simulates any decode tokens for active requests.'''
        logger.info("Appending decode tokens.")
        for request in self.running_requests:
            if request.decoding():
                request.append_decode_token()
            
    def get_curr_time(self) -> float:
        '''Returns current time of scheduler.'''
        return self.curr_time

    def any_request_is_decoding(self) -> bool:
        '''Returns True if there is at least one request in the decode phase.'''
        for request in self.running_requests:
            if request.decoding():
                return False
            
        return True

    def all_requests_finished(self) -> bool:
        '''Returns True if all requests in this schedule have finished.'''
        if self.unstarted_requests or self.waiting_requests or self.running_requests:
            return False
        
        for request in self.all_requests:
            if not request.finished():
                return False

        return True
    
    def update_all_requests_status(self) -> None:
        '''Loops through each request in scheduled_requests and updates its status, metrics.
        
        Called after forward pass runtime has been predicted and the requests need to track runtime metrics.'''
        new_running_requests = deque()

        for request in self.running_requests:
            request.update_request_status(self.curr_time)
            if not request.finished():
                new_running_requests.append(request)

        self.running_requests = new_running_requests

    ###################################################
    ### Checking invariants to help with debugging. ###
    ###################################################
    def check_end_conditions(self) -> None:
        # Check capacity in queues.
        assert not self.running_requests, "Running requests queue must be empty."
        assert not self.waiting_requests, "Waiting requests queue must be empty."
        
        # Check request statuses.
        assert self.all_requests_finished(), "All requests must have finished status."

        # Assert cache is in correct state.
        self.cache_simulator.check_end_conditions()

    def _check_request_invariants(self):
        '''Throws an error if any request status does not match the list they are in.'''
        for request in self.unstarted_requests:
            assert request.unstarted(), "Request in unstarted_requests must be in UNSTARTED phase."

        for request in self.waiting_requests:
            assert request.waiting(), "Request in waiting_requests must be in WAITING phase."

        for request in self.running_requests:
            assert request.prefill() or request.decoding(), "Request in active_requests must be in PREFILL or DECODING phase."

    def log_queue_lengths(self):
        logger.info(f"QUEUE LENGTHS\n\twaiting: {len(self.waiting_requests)},\n\trunning: {len(self.running_requests)}")

    def get_queue_str(self) -> str:
        return f"\n\tunstarted: {self.get_unstarted_queue_names()}\n\twaiting: {self.get_waiting_queue_names()}\n\trunning: {self.get_running_queue_names()}"

    def get_unstarted_queue_names(self) -> List[str]:
        out = []
        for request in self.unstarted_requests:
            out.append(request.name)

        return out
    
    def get_waiting_queue_names(self) -> List[str]:
        out = []
        for request in self.waiting_requests:
            out.append(request.name)

        return out
    
    def get_running_queue_names(self) -> List[str]:
        out = []
        for request in self.running_requests:
            out.append(request.name)

        return out

    ########################################################################
    ### Short circuit functions for token verification queue management. ###
    ########################################################################
    def move_unstarted_to_waiting_request(self):
        """Moves next unstarted request to waiting queue and sets arrival_timestamp to current scheduler time."""
        request = self.unstarted_requests[0]
        request.set_arrival_time(self.curr_time)
        # Move it to waiting queue.
        self.waiting_requests.append(request)
        request.update_request_status(self.curr_time)
        self.unstarted_requests.popleft()

    ######################
    ### Visualization. ###
    ######################
    def get_simulated_request_metrics_visualizer(self) -> SimulatedRequestMetricsVisualizer:
        '''Returns a visualizer object to easily log or print the request metrics.'''
        return SimulatedRequestMetricsVisualizer(self.all_requests)

    def visualize_cache(self):
        logger.info(f"visualize_cache")
        logger.info(f"num_free_gpu: {self.cache_simulator._num_gpu_blocks_available()}")
        logger.info(f"num_free_watermark: {self.cache_simulator._num_watermark_blocks_available()}")
        self.cache_simulator.visualize_block_tables()
