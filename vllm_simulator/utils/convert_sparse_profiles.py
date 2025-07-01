"""Converts Ferdi's sparse profiling data to VLLM profiles."""

from typing import Optional, Tuple
import os

from vllm_simulator.profiler.vllm_profile import VllmProfile
from vllm_simulator.profiler.vllm_complete_profile import VllmCompleteProfile
import utils.parse_vllm_output as parse_vllm_output
from utils import configure_logging

logger = configure_logging(__name__)

# TODO: Run the following if imports are not working.
# export PYTHONPATH=~/code/vllm_simulator:$PYTHONPATH

# Constants for the given profiles.
BLOCK_SIZE=16
NUM_WATERMARK_BLOCKS=3
MAX_MODEL_LEN=8192 # Taken from vllm specs.

# Note: 
# 1. A100, H100 each have 80 GB of RAM.
# 2. Token size:
#   8b: 0.125 MB per token
#   70b: 0.313 MB per token
# 3. So, kv cache capacity = (80GB*tp_size - model weight size)/(token_size * 16) - water_mark
# TODO: Models and their sizes in UNITS?
MODEL_HARDWARE_TO_KV_CACHE_CAPACITY = {
    # "Llama-3B-fp8_4-A100": ,
    "Llama-8B-fp8_4": 31997 * 4 * 2,
    "Llama-8B-bf16_1": 31997,
    "Llama-8B-bf16_4": 31997 * 4,
    "Llama-70B-bf16_4": 35942,
    "Llama-70B-fp8_4": 35942 * 2,
}

def convert_sparse_profile_to_vllm_complete_profile(
        sparse_profile_file_path: str,
        model_id: str,
        num_gpus: int,
        gpu: str,
        input_token_upper_limit: Optional[int] = None,
) -> VllmCompleteProfile:
    num_gpu_blocks = calculate_num_gpu_blocks(num_gpus, model_id)
    max_num_batched_tokens, intervals = csv_to_intervals(sparse_profile_file_path, input_token_upper_limit)
    # TODO: How to get cuda and no cuda profiles?
    cuda_profile = VllmProfile(
        intervals=intervals,
        uses_cuda=True,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=num_gpu_blocks,
        num_watermark_blocks=NUM_WATERMARK_BLOCKS,
        model=model_id,
        model_name=model_id,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_MODEL_LEN,
        input_token_upper_limit=input_token_upper_limit,
    )
    no_cuda_profile = VllmProfile(
        intervals=intervals,
        uses_cuda=False,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=num_gpu_blocks,
        num_watermark_blocks=NUM_WATERMARK_BLOCKS,
        model=model_id,
        model_name=model_id,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_MODEL_LEN,
        input_token_upper_limit=input_token_upper_limit,
    )
    # Currently cuda and no_cuda_profiles are the same due to lack of data.
    vllm_complete_profile = VllmCompleteProfile(
        cuda_profile=cuda_profile,
        no_cuda_profile=no_cuda_profile,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=num_gpu_blocks,
        num_watermark_blocks=NUM_WATERMARK_BLOCKS,
        model=model_id,
        model_name=model_id,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_MODEL_LEN,
        input_token_upper_limit=input_token_upper_limit,
    )
    # Write vllm_complete_profile to file.
    vllm_complete_profile.dump(f"outputs/profiles/autoscaling/{model_id}_{num_gpus}-{gpu}_complete_profile.json")
    return vllm_complete_profile


def calculate_num_gpu_blocks(num_gpus: int, model_id: str) -> int:
    '''Theoretical calculation of number of GPU blocks.'''
    model_hardware_id = f"{model_id}_{num_gpus}"
    return MODEL_HARDWARE_TO_KV_CACHE_CAPACITY[model_hardware_id]


def csv_to_intervals(sparse_profile_file_path: str, input_token_upper_limit: Optional[int]=None) -> Tuple[int, dict]:
    '''Converts sparse profile CSV to intervals.'''
    # Read data from csv.
    data = {} # Maps num_tokens to runtime in seconds
    with open(sparse_profile_file_path, 'r') as file:
        for line in file:
            # Ignore the first line.
            if "batch_size" in line:
                continue
            # Parse the num_tokens in the batch to runtime.
            num_tokens, runtime_ms = line.strip().split(",")
            runtime_s = float(runtime_ms) / 1000
            data[int(num_tokens)] = runtime_s
            # Update input_token_upper_limit if necessary.
            if input_token_upper_limit is None or int(num_tokens) > input_token_upper_limit:
                input_token_upper_limit = int(num_tokens)
    return input_token_upper_limit, parse_vllm_output.linear_interpolate_tokens_to_event_duration(logger, data, input_token_upper_limit)

if __name__ == "__main__":
    sparse_profile_dir = "outputs/sparse_profiling_data"

    file_names = os.listdir(sparse_profile_dir)
    
    for profile_file_path in file_names:
        # Parse model and hardware specs from file name.
        profile_specs = profile_file_path.split("_")
        gpu = profile_specs[0].capitalize()
        if profile_specs[1] == "llama3b":
                # TODO: Support 3B model.
                continue
                model = "Llama-3B"
        elif profile_specs[1] == "llama8b":
                model = "Llama-8B"
        elif profile_specs[1] == "llama70b":
                model = "Llama-70B"
        model_id = f"{model}-{profile_specs[2]}"
        num_gpus = int(profile_specs[3][-5])

        # Converts csv to vllm profile and dumps it.
        convert_sparse_profile_to_vllm_complete_profile(
            sparse_profile_file_path=f"{sparse_profile_dir}/{profile_file_path}",
            model_id=model_id,
            num_gpus=num_gpus,
            gpu=gpu,
        )
