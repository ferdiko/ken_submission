from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, List
import uuid

from utils import configure_logging

# logger = configure_logging(__name__)
logger = configure_logging(__name__, log_to_file=True, log_file_path="outputs/logs/vllm_simulator.log")


class RequestStatus(Enum):
    UNSTARTED = "UNSTARTED"
    WAITING = "WAITING"
    PREFILL = "PREFILL" # Indicates the request requires prefill phases.
    DECODE = "DECODE" # Indicates the request requires decode phases.
    FINISHED = "FINISHED"


@dataclass
class RequestMetrics:
    '''Metrics associated with a SimulatedRequest.

    Simplified version of RequestMetrics from vllm repo.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first acknowledged by the request_scheduler in RequestScheduler.get_new_requests.
        first_token_time: The time when the first token was generated.
        finish_time: The time when the request was finished.
        ttft: first_token_time - arrival_time
        tgt: finish_time - arrival_time
    '''
    name: str = ""
    arrival_time: Optional[float] = None
    first_scheduled_time: Optional[float] = None
    first_prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finished_time: Optional[float] = None
    # Calculated LLM metrics.
    # Time the request spent in vllm's queue.
    time_in_queue: Optional[float] = None
    ttft: Optional[float] = None
    tgt: Optional[float] = None
    # Tracks last time from forward_passes.
    most_recent_time: Optional[float] = None
    # Time span for each forward pass.
    forward_passes: List[float] = field(default_factory=list)

    def reset(self) -> None:
        '''Resets all tracking metrics that are updated as request moves through its forward passes.

        first_scheduled_time is not reset since this is the time that it is moved into the waiting queue.
        
        Called if request is evicted from cache in CacheSimulator.'''
        self.finished_time = None

    def track_forward_pass(self, curr_time: float):
        '''Append a new forward pass duration.'''
        self.forward_passes.append(curr_time - self.most_recent_time)
        self.most_recent_time = curr_time

    def __str__(self):
        return "Metrics{\n\ttotal_time=%s,\n\tarrival_time=%s,\n\tfirst_scheduled_time=%s,\n\tfirst_token_time=%s,\n\tfinished_time=%s}" % (self.total_time(), self.arrival_time, self.first_scheduled_time, self.first_token_time, self.finished_time)

    def get_dict(self):
        '''Used to JSON serialize the metric for file dumping in SimulatedRequestMetricsVisualizer.'''
        return {
            "name": self.name,
            "arrival_time": self.arrival_time,
            "first_scheduled_time": self.first_scheduled_time,
            "first_token_time": self.first_token_time,
            "finished_time": self.finished_time,
            "ttft": self.ttft,
            "tgt": self.tgt,
            "forward_passes": self.forward_passes,
        }


class SimulatedRequest:
    '''Defines an input request to be  simulated.'''
    # request_id: int # UUID to uniquely identify this request.
    # status: RequestStatus # Details what stage the request is in.
    # arrival_timestamp: int # Timestamp the request is received at.
    # in_tokens: int # Total number of tokens in sequence.
    # in_tokens_remaining: int # Number of tokens from input prompt not yet ingested by LLM.
    # out_tokens: int # Number of output tokens that need to be generated.
    # out_tokens_remaining: int # Number of output tokens that have yet to be generated.

    def __init__(self, in_tokens: int, out_tokens: int, arrival_timestamp: int, name: str=""):
        '''Initializes SimulatedRequest instance.'''
        self.in_tokens = in_tokens
        self.in_tokens_remaining = in_tokens
        self.out_tokens = out_tokens
        self.out_tokens_remaining = out_tokens
        self.name = name

        self.request_id = uuid.uuid4()
        self.status = RequestStatus.UNSTARTED

        # Initialize metrics tracking for this request.
        self.metrics = RequestMetrics(name=name, arrival_time=arrival_timestamp)

        logger.info(f"Request {self.request_id} with {self.in_tokens} in_tokens and {self.out_tokens} out_tokens initialized.")

    def consume(self, remaining_batch_tokens: int, curr_time: float) -> int:
        '''Returns the number of tokens from this request to include in the next forward pass.
        
        Greedily includes as many tokens as possible from this request'''
        # Transition request from waiting to prefill phase. Otherwise, status remains unchanged.
        if self.waiting():
            self.update_request_status(curr_time)
        assert not self.unstarted() and not self.waiting() and not self.finished(), f"Request {self.request_id} must be in prefill or decode phase."

        # Token consumption is greedy.
        if self.prefill():
            request_tokens = min(self.in_tokens_remaining, remaining_batch_tokens)
            self.in_tokens_remaining -= request_tokens
            # logger.debug(f"prefill consumed tokens: {request_tokens}")
        elif self.decoding():
            # Note: out_tokens_remaining is decremented in append_decode_tokens from the previous forward pass.
            # In the decoding phase, one token is generated in the last forward pass.
            request_tokens = 1

        # logger.info(f"Request-{self.name}:{self.request_id}: {request_tokens} tokens in batch")
        return request_tokens

    def update_request_status(self, timestamp: float):
        '''Checks in_tokens_remaining and out_tokens_remaining to update status.
        
        Also updates any request metrics with timestamp.'''
        # Request is done being processed.
        if self.finished():
            return
        
        # Request has been acknowledged by the scheduler. This happens in RequestScheduler._queue_unstarted_requests.
        if self.unstarted():
            self.status = RequestStatus.WAITING
            # logger.info(f"Request {self.name}:{self.request_id} updated to status {self.status} at {timestamp} ms.")
            return
        
        # Request has been acknowledged in RequestScheduler._schedule_waiting.
        if self.waiting():
            self.status = RequestStatus.PREFILL
            self.maybe_set_first_scheduled_time(timestamp)
            # logger.info(f"Request {self.name}:{self.request_id} updated to status {self.status}.")
            return
        
        # Track the forward pass duration.
        self.metrics.track_forward_pass(timestamp)

        # Track the first_token_time. Must try before finished_time is set.
        self.maybe_set_first_token_time(timestamp)

        new_status = self.status
        if self.out_tokens_remaining == 0:
            assert self.in_tokens_remaining == 0, f"Request {self.request_id} began decode phase without finishing prefill phase."
            self.maybe_set_finished_time(timestamp)
            # logger.info(f"finished_time is {timestamp}")
            new_status = RequestStatus.FINISHED
        elif self.in_tokens_remaining == 0:
            new_status = RequestStatus.DECODE

        # Set the status and log the change.
        if new_status != self.status:
            self.status = new_status
            # logger.info(f"Request {self.name}:{self.request_id} updated to status {new_status}.")

    def append_decode_token(self) -> None:
        """Generates a decode token.

        This happens at the end of every forward pass in which prefill is finished and decode is not yet completed.
        """
        self.out_tokens_remaining -= 1
        assert self.out_tokens_remaining >= 0, "There must be a non-negative number of out_tokens_remaing."

    def get_num_prefill_tokens(self) -> int:
        '''Dynamically calculates prefill tokens as this may change if request has been preempted.'''
        return self.in_tokens_remaining

    def get_total_tokens(self) -> int:
        '''Returns total number of tokens that this request would consume in the KV cache.'''
        return self.in_tokens + self.out_tokens

    def reset(self) -> None:
        '''Resets request's status and tracking metrics.
        
        Only called if request is preempted (i.e. evicted from cache in CacheSimulator).'''
        if self.in_tokens_remaining == 0:
            # A preempted request's prefill becomes the original prefill AND any decode tokens generated in past forward passes.
            self.in_tokens_remaining = self.in_tokens + (self.out_tokens - self.out_tokens_remaining)
            # self.out_tokens_remaining remains unchanged.
        else: 
            self.in_tokens_remaining = self.in_tokens
            self.out_tokens_remaining = self.out_tokens
        self.status = RequestStatus.WAITING

        # Reset RequestMetrics.
        self.metrics.reset()

    ############################
    # Methods to check status. #
    ############################
    def unstarted(self):
        '''Returns True if SimulatedRequest has not started processing.'''
        return self.status == RequestStatus.UNSTARTED
    
    def waiting(self):
        '''Returns True if SimulatedRequest is waiting to be consumed.'''
        return self.status == RequestStatus.WAITING
    
    def prefill(self):
        '''Returns True if SimulatedRequest is in prefill phase.'''
        return self.status == RequestStatus.PREFILL
    
    def decoding(self):
        '''Returns True if SimulatedRequest is in decode phase.'''
        return self.status == RequestStatus.DECODE 

    def finished(self):
        '''Returns True if SimulatedRequest has finished processing.'''
        return self.status == RequestStatus.FINISHED

    ################################
    # Method to visualize request. #
    ################################
    def __str__(self):
        return "SimulatedRequest{name=%s, request_id=%s, in_tokens=%d, in_tokens_remaining=%d, out_tokens=%d, out_tokens_remaining=%d, status=%s}" % (self.name, self.request_id, self.in_tokens, self.in_tokens_remaining, self.out_tokens, self.out_tokens_remaining, self.status)

    def get_metrics(self):
        out = ""
        metrics_dict = self.get_metrics_dict()
        for key, val in metrics_dict.items():
            out += f"\n\t{key}: {val}"
        return out
    
    def get_arrival_timestamp(self) -> float:
        return self.metrics.arrival_time
    
    def get_first_token_time(self) -> float:
        return self.metrics.first_token_time

    def get_finished_time(self) -> float:
        return self.metrics.finished_time
    
    def get_metrics_dict(self):
        '''Return JSON serializable dictionary with request ID and all metrics.'''
        res = {"request_id": str(self.request_id), "in_tokens": self.in_tokens, "out_tokens": self.out_tokens} | asdict(self.metrics)
        return res
        
    def __eq__(self, other):
        # UUID must be the same object (i.e. it was initialized for the same request).
        return isinstance(other, self.__class__) and self.request_id == other.request_id

    def __hash__(self):
        # Use the request_id for hashing since it uniquely identifies the request.
        return hash(self.request_id)
    
    ##################################
    # Method to set request metrics. #
    ##################################
    def set_arrival_time(self, time: float):
        self.metrics.arrival_time = time
        self.metrics.most_recent_time = time

    def maybe_set_first_scheduled_time(self, time: float):
        if self.metrics.first_scheduled_time is None:
            self.metrics.first_scheduled_time = time
            self.metrics.time_in_queue = self.metrics.first_scheduled_time - self.metrics.arrival_time
            # TODO: Should this be set somewhere else?
            self.metrics.first_prefill_start_time = time

    def maybe_set_first_token_time(self, time: float):
        if self.out_tokens_remaining == self.out_tokens - 1 and self.metrics.first_token_time is None:
            self.metrics.first_token_time = time

    def maybe_set_finished_time(self, time: float):
        if self.out_tokens_remaining == 0 and self.metrics.finished_time is None:
            self.metrics.finished_time = time
            self.metrics.ttft = self.metrics.first_token_time - self.metrics.arrival_time
            self.metrics.tgt = self.metrics.finished_time - self.metrics.arrival_time
