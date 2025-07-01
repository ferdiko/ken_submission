from typing import List

from vllm_simulator.simulator.simulated_request import SimulatedRequest
from utils.gpu_block_helpers import tokens_to_blocks, blocks_to_tokens

from utils import configure_logging

logger = configure_logging(__name__)

class RequestGpuBlockTracker:
    def __init__(self, request: SimulatedRequest, block_size: int):
        '''Initializing a RequestGpuBlockTracker instance '''
        self.num_gpu_blocks = tokens_to_blocks(request.get_num_prefill_tokens(), block_size) # Number of GPU blocks earmarked for prefill tokens.
        self.num_tokens = request.get_num_prefill_tokens() # Number of tokens that are contained in the marked GPU blocks.
        self.num_watermark_blocks = 0 # Number of watermark blocks used by this request.
        self.block_size = block_size

        self.request_id = request.request_id # Added for debug print
        self.request_name = request.name

        logger.info(f"Created Request-{str(request.request_id)} " + str(self))

    def has_token_space(self, num_tokens: int=1) -> bool:
        '''Returns true if num_tokens can fit in the GPU blocks that are earmarked for this request.'''
        total_tokens_marked_from_blocks = blocks_to_tokens(self.num_gpu_blocks + self.num_watermark_blocks, self.block_size)
        available_tokens_marked_from_blocks = total_tokens_marked_from_blocks - self.num_tokens
        return available_tokens_marked_from_blocks >= num_tokens

    def add_token(self, use_watermark_if_no_token_space:bool=False) -> None:
        '''Adds token by filling in pre-existing GPU blocks or borrowing another.
        
        When this is called, guaranteed that there is available space for another token within the pre-existing blocks or a free GPU block to be used.'''
        if not self.has_token_space():
            # If no more token space available, take a new GPU block.
            if use_watermark_if_no_token_space:
                # Watermark blocks can only be used for decode requests.
                self.num_watermark_blocks += 1
            else:
                self.num_gpu_blocks += 1
        self.num_tokens += 1

        logger.info(f"Request-{self.request_name}:{str(self.request_id)} " + str(self))

    def __str__(self):
        return "RequestGpuBlockTracker{\n\tnum_gpu_blocks=%s, watermark_blocks=%d, num_tokens=%s, blocksTokensShouldTake=%s}" % (self.num_gpu_blocks, self.num_watermark_blocks, self.num_tokens, tokens_to_blocks(self.num_tokens, self.block_size))

    ##########################
    # Visualization methods. #
    ##########################
    def get_total_blocks(self):
        return self.num_gpu_blocks + self.num_watermark_blocks
