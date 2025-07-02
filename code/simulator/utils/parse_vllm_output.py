'''
Helper functions to parse the trace and token_log output from a vLLM run.
'''
import json

from vllm_simulator.profiler.vllm_profile import TokensToRuntime 

MAX_TOKENS = 10e4 # TODO: This is arbitrary, but I don't think the GPUs right now can handle this many tokens regardless.
STEPS_LOG_FILE_PATH = "outputs/steps.log"


def parse_cache_log(logger, log_file_path="outputs/cache.log"):
    '''Reads in the cache.log and parses out the KV cache attributes.'''
    logger.info(f"parse_cache_log reading from {log_file_path}")
    with open(log_file_path, 'r') as f:
        block_size_info = f.readline().strip().split(":")
        assert block_size_info and block_size_info[0] == "block_size", f"First line of cache_log is {block_size_info}"
        block_size = int(block_size_info[1])

        num_gpu_blocks_info = f.readline().strip().split(":")
        assert num_gpu_blocks_info and num_gpu_blocks_info[0] == "num_gpu_blocks", f"Second line of cache_log is {num_gpu_blocks_info}"
        num_gpu_blocks = int(num_gpu_blocks_info[1])

        num_watermark_blocks_info = f.readline().strip().split(":")
        assert num_watermark_blocks_info and num_watermark_blocks_info[0] == "num_watermark_blocks", f"Third line of cache_log is {watermark_blocks_info}"
        num_watermark_blocks = int(num_watermark_blocks_info[1])

    return block_size, num_gpu_blocks, num_watermark_blocks
    

def parse_steps_log(logger, log_file_path="outputs/steps.log"):
    with open(log_file_path, "r") as file:
        steps_info = json.load(file)

    assert steps_info, f"steps_info must have nonzero length."
    return steps_info

def linear_interpolate_tokens_to_event_duration(logger, data, input_token_upper_limit):
    '''Initializes intervals using data to construct intervals using linear interpolation.'''
    logger.info("_linear_interpolate_tokens_to_event_duration started.")
    intervals = {} # Stores (token_interval_start, token_interval_end) to lambda num_tokens: runtime

    assert len(data) > 0, "Ensure that at least 1 run of vLLM has been characterized."

    # Sort the data and create intervals for each pair of adjacent data points.
    sorted_data = [(num_tokens, runtime) for num_tokens, runtime in data.items()]
    sorted_data.sort()

    # Start the first interval at 0 tokens.
    interval_start = 0
    interval_start_runtime = None
    i = 0
    while i < len(sorted_data):
        # Set interval end to current datapoint.
        interval_end, curr_datapoint_runtime = sorted_data[i]

        # If it is the final interval, modify it such that it extends to MAX_TOKENS.
        interval = (interval_start, interval_end) if i < len(sorted_data) - 1 else (interval_start, input_token_upper_limit)

        # Determine a lambda function that relates tokens to runtime. Must pass in the line specifications
        # to avoid the lambda functions modifying each other in the loop.
        if i == 0:
            # Since no data exists before this point, it will be a constant function.
            tokens_to_runtime = TokensToRuntime(0, curr_datapoint_runtime)
            # tokens_to_runtime_line = {"slope": 0, "intercept": curr_datapoint_runtime}
            logger.info(f"Interval {interval} with function y = {curr_datapoint_runtime}")
        else:
            # Calculate slope and y-intercept of line.
            slope = (curr_datapoint_runtime - interval_start_runtime) / (interval_end - interval_start)
            intercept = curr_datapoint_runtime - slope * interval_end # Use the current datapoint to calculate b.
            tokens_to_runtime = TokensToRuntime(slope, intercept)
            # tokens_to_runtime_line = {"slope": slope, "intercept": intercept}
            logger.info(f"Interval {interval} with function y = {slope} * x + {intercept}")

        # Record interval to lambda function.
        intervals[interval] = tokens_to_runtime

        # Update loop variables
        i += 1
        interval_start = interval_end
        interval_start_runtime = curr_datapoint_runtime

    logger.info(f"_linear_interpolate_tokens_to_event_duration finished.")
    return intervals


################################################
# For profiling tokenizer/embedding overheads. #
################################################
def parse_tokenizer_log_file(logger, log_file_path="outputs/profiling_data/tokenizer_metrics.json"):
    with open(log_file_path, 'r') as f:
        request_metrics = json.load(f)
    return request_metrics
