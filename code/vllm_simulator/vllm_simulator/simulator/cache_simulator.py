from collections import deque
from dataclasses import dataclass

from utils.gpu_block_helpers import tokens_to_blocks, CacheCapacityError
from vllm_simulator.simulator.simulated_request import SimulatedRequest
from vllm_simulator.simulator.request_gpu_block_tracker import RequestGpuBlockTracker


from utils import configure_logging

# logger = configure_logging(__name__)
logger = configure_logging(__name__, log_to_file=True, log_file_path="outputs/logs/vllm_simulator.log")


@dataclass
class CacheSimulatorConfig:
    '''Contains all required variables for initializing a CacheSimulator.'''
    block_size: int
    num_gpu_blocks: int
    max_batch_tokens: int
    num_watermark_blocks: int


class CacheSimulator:
    '''Simulates the capacity of a KV cache and updates the available capacity as requests are processed.'''
    def __init__(self, block_size: int, total_gpu_blocks: int, max_batch_tokens: int, num_watermark_blocks: int):
        '''Initializes CacheSimulator instance.'''
        # Characterization variables.
        self.block_size = block_size # Number of tokens in a block.
        self.num_gpu_blocks = total_gpu_blocks - num_watermark_blocks # Number of GPU blocks in the cache that can be used for prefill or decode.
        self.max_batch_tokens = max_batch_tokens # Maximum number of tokens that can be in one batch.
        self.num_watermark_blocks = num_watermark_blocks # Number of GPU blocks reserved for decode tokens only.

        # Tracks cache capacity
        self.request_gpu_block_trackers: dict[SimulatedRequest, RequestGpuBlockTracker] = {}

        # Log instance information.
        logger.info("Initialized " + str(self))

    @classmethod
    def from_config(cls, config: CacheSimulatorConfig):
        '''Factory method to initialize a CacheSimulator.'''
        return cls(config.block_size, config.num_gpu_blocks, config.max_batch_tokens, config.num_watermark_blocks)

    def has_prefill_capacity(self, request: SimulatedRequest) -> bool:
        '''Returns true if cache has capacity to fit all prefill tokens of the request.'''
        assert request.waiting(), f"Request being considered to start prefill must be in waiting status, but it is in {request.status}."
        needed_gpu_blocks = tokens_to_blocks(request.get_num_prefill_tokens(), self.block_size)
        # Note: vllm does not decrement watermark_blocks ever.
        return self.num_total_blocks_available() - needed_gpu_blocks >= self.num_watermark_blocks

    # # TODO: deprecate. Replaced with BlockManagerV1 simple heuristic of 1 free GPU block available.
    # def has_decode_capacity(self, request: SimulatedRequest) -> bool:
    #     '''Returns true if an additional decode token from the request can fit in KV cache.'''
    #     assert request.decoding(), f"Request being considered for decoding must be in decoding status, but it is in {request.status}."
    #     request_gpu_block_tracker = self._get_or_create_request_gpu_block_tracker(request)

    #     # Return true if occupied GPU blocks have additional space or there is a GPU/watermark block available.
    #     return (
    #         request_gpu_block_tracker.has_token_space() or
    #         self._num_gpu_blocks_available() > 0 or
    #         self._num_watermark_blocks_available() > 0)

    def num_total_blocks_available(self) -> int:
        '''Returns the number of available GPU blocks in this cache, including watermark.'''
        return self._num_gpu_blocks_available() + self._num_watermark_blocks_available()

    def _num_gpu_blocks_available(self):
        '''Returns the number of available GPU blocks in this cache, excluding watermark.'''
        num_gpu_blocks_used = sum(request_gpu_block_tracker.num_gpu_blocks for request_gpu_block_tracker in self.request_gpu_block_trackers.values())
        return self.num_gpu_blocks - num_gpu_blocks_used

    def _num_watermark_blocks_available(self):
        '''Returns the number of available watermark GPU blocks in this cache.'''
        num_watermark_gpu_blocks_used = sum(request_gpu_block_tracker.num_watermark_blocks for request_gpu_block_tracker in self.request_gpu_block_trackers.values())
        return self.num_watermark_blocks - num_watermark_gpu_blocks_used

    def get_num_consumed_tokens_from_request(self, curr_batch_tokens: int, curr_time: float, request: SimulatedRequest) -> int:
        '''Returns the number of consumed tokens in this forward pass from the given request.'''
        self.mark_gpu_blocks_used(request)
        
        remaining_batch_tokens = self.max_batch_tokens - curr_batch_tokens
        curr_request_tokens = request.consume(remaining_batch_tokens, curr_time)

        # Return the number of KV cache tokens from this request.
        return curr_request_tokens

    def preempt(self, request: SimulatedRequest) -> None:
        '''Frees request tokens in cache.'''
        self._mark_gpu_blocks_available(request)
        logger.info("Preempted Request %s\n\t%s" % (request.request_id, str(self)))

    def clean_cache(self):
        '''Removes tokens of finished requests from cache.'''
        finished_requests = []
        for request in self.request_gpu_block_trackers.keys():
            if request.finished():
                finished_requests.append(request)
        
        for request in finished_requests:
            # Remove tokens of finished request from the cache.
            self._mark_gpu_blocks_available(request)
    
    def mark_gpu_blocks_used(self, request: SimulatedRequest) -> None:
        '''Updates cache state to mark blocks as used and associate it with given request.'''
        if request.waiting():
            # Prefill blocks are reserved when request first starts prefill.
            self._mark_prefill_gpu_blocks_used(request)
        elif request.decoding():
            self._mark_decode_gpu_blocks_used(request)

    def _mark_prefill_gpu_blocks_used(self, request: SimulatedRequest) -> None:
        '''Updates cache state to mark blocks needed for prefill tokens as used and associates it with given request.
        
        When this is called, guaranteed that 1) cache has space for prefill tokens, 2) request in waiting phase.'''
        assert request.waiting(), f"Request {request.request_id} must be in waiting status, but it is {request.status}"

        # Blocks are marked as used by following call to create tracker.
        self._get_or_create_request_gpu_block_tracker(request)

        assert self._num_gpu_blocks_available() >= 0, "Cache must have >= 0 GPU blocks available at all times."

    def _mark_decode_gpu_blocks_used(self, request: SimulatedRequest) -> None:
        '''Updates cache state to mark blocks as used and associates it with given request.
        
        When this is called, guaranteed that 1) cache has space for 1 token, 2) request in decode phase.'''
        assert request.decoding(), f"Request {request.request_id}  must be in waiting status, but it is {request.status}"

        request_gpu_block_tracker = self._get_or_create_request_gpu_block_tracker(request)
        if request_gpu_block_tracker.has_token_space():
            # Try to use already marked GPU blocks if possible.
            request_gpu_block_tracker.add_token()
        else:
            # Else, use KV cache blocks. Watermark will be used if 1) no other normal KV cache blocks are available,
            #   2) all token space is used in the request's GPU blocks.
            use_watermark_if_no_token_space = self._num_gpu_blocks_available() == 0
            request_gpu_block_tracker.add_token(use_watermark_if_no_token_space)

        assert self._num_gpu_blocks_available() >= 0 or self._num_watermark_blocks_available() >= 0, "Cache must have >= 0 GPU blocks available at all times."

    def _mark_gpu_blocks_available(self, request: SimulatedRequest) -> None:
        '''Updates cache state to mark blocks associated with given request as available.'''
        del self.request_gpu_block_trackers[request]

        assert self._num_gpu_blocks_available() <= self.num_gpu_blocks, "Cache must have <= total GPU blocks available at all times."
        assert self._num_watermark_blocks_available() <= self.num_watermark_blocks, "Cache must have <= total watermark blocks available at all times."

    def _get_or_create_request_gpu_block_tracker(self, request: SimulatedRequest) -> RequestGpuBlockTracker:
        '''Retrieves the RequestGpuBlockTracker for this request. Creates one if it does not exist.'''
        if request not in self.request_gpu_block_trackers:
            self.request_gpu_block_trackers[request] = RequestGpuBlockTracker(request, self.block_size)

        return self.request_gpu_block_trackers[request]

    def __str__(self):
        return "CacheSimulator{block_size=%s, num_gpu_blocks=%d, _num_gpu_blocks_available=%d, num_watermark_blocks=%d, _num_watermark_blocks_available=%d}" % (self.block_size, self.num_gpu_blocks, self._num_gpu_blocks_available(), self.num_watermark_blocks, self._num_watermark_blocks_available())
    
    ###################################################
    ### Checking invariants to help with debugging. ###
    ###################################################

    def check_end_conditions(self) -> None:
        '''Throws an error if state is incorrect at end of simulation'''
        assert self.num_gpu_blocks == self._num_gpu_blocks_available(), "After all requests are finished, there must be the original number of GPU blocks available."
        assert self.num_watermark_blocks == self._num_watermark_blocks_available(), "After all requests are finished, there must be the original number of watermark blocks available."

    ##########################
    # Visualization methods. #
    ##########################
    def visualize_block_tables(self):
        logger.info("visualizing block_tables")
        for request, gpu_block_tracker in self.request_gpu_block_trackers.items():
            logger.info(f"\tRequest-{request.name}: {gpu_block_tracker.get_total_blocks()} = {gpu_block_tracker.num_gpu_blocks} + {gpu_block_tracker.num_watermark_blocks}")
