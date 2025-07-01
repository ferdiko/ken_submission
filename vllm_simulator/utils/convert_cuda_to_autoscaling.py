"""Converts VLLM profiles with cuda and no cuda times to only cuda times. For autoscaling."""

from typing import Optional, Tuple
import os

from vllm_simulator.profiler.vllm_profile import VllmProfile
from vllm_simulator.profiler.vllm_complete_profile import VllmCompleteProfile
import utils.parse_vllm_output as parse_vllm_output
from utils import configure_logging

logger = configure_logging(__name__)


def convert_cuda_to_autoscaling(profile_file_path: str, output_profile_file_path: str) -> None:
    """Converts VLLM profile with cuda and no cuda times to only cuda times."""
    # Load the VLLM profile.
    vllm_complete_profile = VllmCompleteProfile.load(profile_file_path)

    cuda_profile = vllm_complete_profile.cuda_profile

    # Duplicate the cuda_profile for the no_cuda_profile.
    no_cuda_profile = VllmProfile(
        intervals=cuda_profile.intervals,
        uses_cuda=True,
        block_size=cuda_profile.block_size,
        num_gpu_blocks=cuda_profile.num_gpu_blocks,
        num_watermark_blocks=cuda_profile.num_watermark_blocks,
        model=cuda_profile.model,
        model_name=cuda_profile.model_name,
        max_num_batched_tokens=cuda_profile.max_num_batched_tokens,
        max_model_len=cuda_profile.max_model_len,
        max_num_seqs=cuda_profile.max_num_seqs,
        input_token_upper_limit=cuda_profile.input_token_upper_limit,
    )

    # Create a new VLLM complete profile with only cuda times.
    new_vllm_complete_profile = VllmCompleteProfile(
        cuda_profile=cuda_profile,
        no_cuda_profile=no_cuda_profile,
        block_size=vllm_complete_profile.block_size,
        num_gpu_blocks=vllm_complete_profile.num_gpu_blocks,
        num_watermark_blocks=vllm_complete_profile.num_watermark_blocks,
        model=vllm_complete_profile.model,
        model_name=vllm_complete_profile.model_name,
        max_num_batched_tokens=vllm_complete_profile.max_num_batched_tokens,
        max_model_len=vllm_complete_profile.max_model_len,
        max_num_seqs=vllm_complete_profile.max_num_seqs,
        input_token_upper_limit=vllm_complete_profile.input_token_upper_limit,
    )

    # Dump the new VLLM complete profile.
    new_vllm_complete_profile.dump(output_profile_file_path)


if __name__ == "__main__":
    profile_dir = "outputs/profiles"

    file_names = os.listdir(profile_dir)
    
    for profile_file_path in file_names:
        if not profile_file_path.endswith(".json"):
              continue
        
        # Create autoscaling profile without no cuda times and dump it.
        convert_cuda_to_autoscaling(
             profile_file_path=f"{profile_dir}/{profile_file_path}",
             output_profile_file_path=f"outputs/profiles/autoscaling/{profile_file_path}")