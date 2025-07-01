from utils import configure_logging
from vllm_simulator.profiler.vllm_complete_profile import VllmCompleteProfile
from vllm_simulator.simulator.vllm_simulator import VllmSimulator

# Initialize logging.
logger = configure_logging()

# Load a profile from json.
vllm_complete_profile = VllmCompleteProfile.load("outputs/profiles/8b_complete_profile.json")

# # Or, generate a profile.
# vllm_profiler = VllmProfiler(
#     model=model,
#     model_name=model_name,
#     max_tokens=1, # Limiting max number of output tokens for profiling.
#     max_num_batched_tokens=1024,
#     max_model_len=2048,
#     max_num_seqs=1003)

# vllm_complete_profile = vllm_profiler.get_vllm_complete_profile()

# vllm_profiler.dump_complete_profile(f"outputs/profiles/8b_complete_profile.json")

# Initialize simulator with profile.
simulator = VllmSimulator(vllm_complete_profile)

# Simulate a workload with 1 large request.
request_metrics_visualizer = simulator.simulate_requests(
    in_tokens=[1000],
    out_tokens=[200],
    queries_per_iter=[1])

# Log the simulator request metrics to terminal.
request_metrics_visualizer.log_metrics()