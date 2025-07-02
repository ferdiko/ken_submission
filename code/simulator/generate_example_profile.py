from utils import configure_logging
# Comment this out if no GPUs available or VllmProfiler unneeded (it takes a long time to import).
from vllm_simulator.profiler.vllm_profiler import VllmProfiler

# Initialize logging.
logger = configure_logging()

# model = "neuralmagic/Meta-Llama-3-70B-Instruct-FP8"
# model_name = "llama-70b_L40S"
# model = "neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
# model_name = "llama-8b_L40S"
# model = "neuralmagic/Llama-3.2-1B-Instruct-FP8"
# model_name = "llama-1b_L40S"
model = "neuralmagic/Llama-3.2-3B-Instruct-FP8"
model_name = "llama-3b_L40S"

vllm_profiler = VllmProfiler(
    model=model,
    model_name=model_name,
    max_tokens=1, # Limiting max number of output tokens for profiling.
    max_num_batched_tokens=1024,
    max_model_len=2048,
    max_num_seqs=1003)
        
vllm_profiler.dump_complete_profile(f"outputs/profiles/{model_name}_complete_profile.json")
# vllm_profiler.dump_vllm_profile(f"outputs/profiles/llama-8b_L40S.json")
