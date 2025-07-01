import math


class CacheCapacityError(Exception):
    '''Error used if cache is not large enough to contain a request.'''
    # Inherits all methods from Exception class.
    pass


def tokens_to_blocks(num_tokens: int, block_size: int) -> int:
    '''Returns minimum whole number of blocks that will fit tokens.'''
    return math.ceil(num_tokens / block_size)


def blocks_to_tokens(num_blocks: int, block_size: int) -> int:
    '''Returns number of tokens that num_blocks GPU blocks can store.'''
    return num_blocks * block_size
