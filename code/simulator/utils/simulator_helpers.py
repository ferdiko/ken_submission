import logging
import os
from typing import List


def validate_simulated_request_input(in_tokens: List[int], out_tokens: List[int], queries_per_iter: List[int]):
    """Validate the number of queries defined and that {in, out}_tokens > 0."""
    assert len(in_tokens) == len(out_tokens), f"The number of defined requests must be the same, but len(in_tokens)={len(in_tokens)}, len(out_tokens)={len(out_tokens)}."
    assert len(in_tokens) == sum(queries_per_iter), f"The total queries over all iterations must be equal to the number of defined requests={len(in_tokens)}."

    for i, x in enumerate(in_tokens):
        assert x > 0, f"Request {i} must have in_tokens > 0."

    for i, x in enumerate(out_tokens):
        assert x > 0, f"Request {i} must have out_tokens > 0."


def get_logger_records(logger_name="steps_logger"):
    '''Retrieves the records from a logger with a JSONListHandler.'''
    logger = logging.getLogger(logger_name)
    # Find the JSONListHandler in the logger's handlers
    json_handler = None
    print(f"there are this many handlers: {len(logger.handlers)}")
    return logger.handlers[0].records


def maybe_create_dir(file_path: str):
    '''Create directory if it does not exist.'''
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
