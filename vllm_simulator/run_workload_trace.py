from typing import List
import random

from vllm import LLM, SamplingParams, RequestOutput, SequenceGroup
import utils.generate_vllm_input as generate_vllm_input

from torch.profiler import ProfilerActivity, profile


def _generate_vllm_prompts(llm, in_tokens, out_tokens) -> List[SequenceGroup]:
    """Generates random vllm prompts."""
    # Use tokenizer to generate prompts
    tokenizer = llm.get_tokenizer()
    prompts, prompt_token_ids = generate_vllm_input.generate_random_prompts(tokenizer, in_tokens)
    
    # Generate requests as sequence groups.
    sampling_params = []
    for out in out_tokens:
        sampling_params.append(SamplingParams(best_of=1,
                                        temperature=0.0,
                                        top_p=1,
                                        top_k=-1,
                                        # use_beam_search=False,
                                        max_tokens=out,
                                        presence_penalty=0,
                                        frequency_penalty=0,
                                        min_tokens=out))

    seq_groups = llm.produce_request(prompts, sampling_params, prompt_token_ids=prompt_token_ids)

    # Combine sequence groups and number of out_tokens.
    for s, out in zip(seq_groups, out_tokens):
        s.seqs[0].out_tokens = out

    return seq_groups

# Clear vllm.log
VLLM_LOG_FILE_PATH = "outputs/logs/vllm.log"
with open(VLLM_LOG_FILE_PATH, 'w'):
    pass

llm = LLM(
    model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
    # model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    quantization="fp8",
    tensor_parallel_size=2,
    # enforce_eager=False,
    # Extra arguments for EngineArgs as input to the LLMEngine go here.
    max_num_batched_tokens=512, # Must be >= max_model_len, max_num_seqs. Essentially no limit, allows modification of batch size. Defaults to 8192 if enable_chunked_prefill==False.
    max_model_len=6700,
    max_num_seqs=512,
    enable_chunked_prefill=True, # Toggling chunked_prefill.
)

# # Use a 5 to 1 ratio of in to out tokens. Randomly add large and small requests with 1:1 ratio.
# num_requests = 20
# queries_per_iter = [num_requests]
# in_tokens = []
# out_tokens = []
# for _ in range(num_requests):
#     x = random.randint(500, int((max_model_len // 1.2) - 100))  # Generate a random in_token. Upper limit ensures in + out < max_model_len (even with randmoness of prompt generation).
#     in_tokens.append(x)
#     out_tokens.append(x // 5)

# Send a warmup.
queries_per_iter = [1]
in_tokens = [4]
out_tokens = [2]
seq_groups = _generate_vllm_prompts(llm, in_tokens, out_tokens)
# Determine the vllm_in_tokens for the vllm tokenizer generated prompts.
vllm_in_tokens = []
for seq_group in seq_groups:
    # Note: Each seq_group is a request and only holds one sequence for our purposes. Thus,
    #   only need to inspect the first sequence in the seq_group.
    seq = seq_group.seqs[0]
    vllm_in_tokens.append(seq.get_prompt_len())
llm.run_workload_trace(queries_per_iter=queries_per_iter,
                        requests=seq_groups,
                        log_token_verification=True)
# Warmup finished.

# 5 large simulataneous requests
num_requests = 5
queries_per_iter = [1 for _ in range(num_requests)]
in_tokens = []
out_tokens = []
for _ in range(num_requests):
    in_tokens.append(1000)
    out_tokens.append(200)

print("Start generating prompts.")
seq_groups = _generate_vllm_prompts(llm, in_tokens, out_tokens)
print("Prompts generated.")

# Determine the vllm_in_tokens for the vllm tokenizer generated prompts.
vllm_in_tokens = []
for seq_group in seq_groups:
    # Note: Each seq_group is a request and only holds one sequence for our purposes. Thus,
    #   only need to inspect the first sequence in the seq_group.
    seq = seq_group.seqs[0]
    vllm_in_tokens.append(seq.get_prompt_len())

# Run workload. This outputs tokens per event log, query_arrival_timestamps, and query_ids
print("Start running workload trace.")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    _, query_arrival_timestamps, query_ids = llm.run_workload_trace(queries_per_iter=queries_per_iter,
                                                                            requests=seq_groups,
                                                                            log_token_verification=True)
print("Finished running workload trace.")

prof.export_chrome_trace(f"llama_70b_512_5_seq_1000_in_no_warmup.json")
