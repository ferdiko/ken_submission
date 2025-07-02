from vllm_simulator.profiler.vllm_profiler import VllmProfiler

# Run profiler and save steps_coordinates_{cuda, no_cuda} into correct directory.
# VllmProfiler(
#     model_name="llama8b_fp8",
#     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
#     max_tokens=1,
#     max_num_batched_tokens=512,
#     max_model_len=8192,
#     max_num_seqs=512)

# VllmProfiler(
#     model_name="llama8b_fp8",
#     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
#     max_tokens=1,
#     max_num_batched_tokens=1024,
#     max_model_len=8192,
#     max_num_seqs=1024)

# VllmProfiler(
#     model_name="llama70b_fp8",
#     model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
#     max_tokens=1,
#     max_num_batched_tokens=512,
#     max_model_len=6700,
#     max_num_seqs=512)

VllmProfiler(
    model_name="llama70b_fp8",
    model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
    max_tokens=1,
    max_num_batched_tokens=1024,
    max_model_len=6700,
    max_num_seqs=1024)
