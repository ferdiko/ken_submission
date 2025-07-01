from vllm import LLM, SamplingParams
from runner_utils import *
import time
import asyncio
from transformers import AutoTokenizer
import numpy as np
import psutil
import argparse
from collections import defaultdict


# def run_vicuna(llm_small, llm_large, cert_thresh, sampling_params, num_requests=-1):
#     # Create prompts.
#     prompts = []
#     benchmark_tasks = get_mt_bench_prompts(vicuna=True)
#     for q in benchmark_tasks:
#         # TODO: Maybe add system prompt? For multi turn the format is important
#         prompts.append(f"Prompt: {q[0]}\nResponse: ")

#     if num_requests > 0:
#         prompts = prompts[:num_requests]

#     # Add prompts to system.
#     loop.run_until_complete(asyncio.gather(
#                                 llm_small.add_new_requests((prompts, sampling_params)),
#                                 llm_large.add_new_requests((prompts, sampling_params)),
#                             ))

#     # Dict of request token sequences.
#     generated_tokens = {}
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

#     for i, p in enumerate(prompts):
#         generated_tokens[i] = tokenizer.encode(p)

#     # Do forward steps.
#     num_steps = len(prompts)*1000
#     tokens_generated_total = 0
#     tokens_generated_large_model = 0

#     start = time.time()
#     for _ in range(num_steps):
#         _, iter_tokens, iter_certainties = llm_small.pause_and_step_sync([])

#         for k in iter_tokens.keys():
#             if iter_certainties[k] < cert_thresh:
#                 tokens_generated_large_model += 1

#                 other_requests = [ki for ki in generated_tokens.keys() if ki != k]

#                 llm_large.add_tokens_to_request_sync((k, generated_tokens[k]))

#                 iter_tokens_large = {}
#                 while len(iter_tokens_large) < 1:
#                     # Run prefill until a token is generated.
#                     _, iter_tokens_large, _ = llm_large.pause_and_step_sync(other_requests)
                
#                 generated_tokens[k].append(iter_tokens_large[k])

#                 llm_large.resume_requests_sync(other_requests)
#                 llm_small.add_tokens_to_request_sync((k, generated_tokens[k])) # I think this should work since last token is next one to be propagated anyways and we only ever add one.
#             else:
#                 generated_tokens[k].append(iter_tokens[k])

#             tokens_generated_total += 1
        
#         # TODO: Free sequences if other model in cascade freed them!!
#         assert False

#     print(f"COST: {tokens_generated_large_model}/{tokens_generated_total}")
#     print("TIME:", time.time() - start)

#     # Decode tokens into text.
#     tasks = []
#     for k in generated_tokens.keys():
#         response = tokenizer.decode(generated_tokens[k])
#         tasks.append(response)

#     return tasks


def run_mt_bench(llm, sampling_params, num_requests=-1):
    # Get requests.
    benchmark_tasks = get_mt_bench_prompts(vicuna=False)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    if num_requests > 0:
        benchmark_tasks = benchmark_tasks[:num_requests]

    max_turns = max([len(t) for t in benchmark_tasks])

    # Experiment stats and hyperparams.
    tokens_generated_total = 0
    start = time.time()

    # Do i-th turn for all requests.
    req_id = 0
    system_prompt = "You are a helpful assistant and need to answer user questions based on \
the previous chat history. In the chat history, the user's requests are preceeded by 'Task:' \
and the responses are preceeded by 'Response:'\n\n"
    all_requests = [system_prompt for _ in benchmark_tasks]

    # Keep track of generated tokens.
    # generated_tokens = defaultdict(list)

    # outputs = []

    max_turns = 2
    for i in range(max_turns):
        # Prepare inputs for turn.
        sampling_params.max_tokens += 1000

        for j, t in enumerate(benchmark_tasks):
            if len(t) <= i:
                continue

            all_requests[j] += f"Task: {t[i]}\nResponse: "

            # generated_tokens[j] = tokenizer.encode(all_requests[j])

            # print("\n"*5)
            # print("INPUT:\n", all_requests[j])

        outputs, _ = llm.generate(all_requests, sampling_params)
        # llm.add_new_requests(all_requests, sampling_params)

        # for _ in range(num_steps):
        #     _, iter_tokens, _ = llm.step()

        # for k in iter_tokens.keys():
        #     generated_tokens[j].append(iter_tokens[k])

        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            all_requests[j] += generated_text + "\n"
        
    # print(f"COST: {tokens_generated_large_model}/{tokens_generated_total}")
    print("TIME:", time.time() - start)

    return all_requests



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run simple cascade model with specified parameters.")
    parser.add_argument("--model", type=str, required=True, help="The name of the first model (e.g., '3b').")

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    model = args.model

    # t = num turns
    # i = initial token budge
    # a = additional token budget per iter
    # pp = presence penalty
    out_file_path = f"mt_bench_runs_async/vllm_{model}_t2_i1000_a1000_pp0.txt"

    if not os.path.exists(out_file_path):

        print("Running:", out_file_path, "\n")

        # Create a sampling params object.
        sampling_params = SamplingParams(best_of=1,
                                        temperature=0.0,
                                        top_p=1,
                                        top_k=-1,
                                        max_tokens=1000,
                                        presence_penalty=0,
                                        frequency_penalty=0,
                                        min_tokens=0)

        models = {
            "1b": "meta-llama/Llama-3.2-1B-Instruct",
            "3b": "meta-llama/Llama-3.2-3B-Instruct",
            "8b": "meta-llama/Llama-3.1-8B-Instruct",
            "70b": "meta-llama/Llama-3.1-70B-Instruct"
        }

        # Create LLMs.
        llm = LLM(
                #   model="meta-llama/Llama-3.2-1B-Instruct", 
                model=models[model], #"meta-llama/Llama-3.1-70B-Instruct", 
                enforce_eager=False,
                max_num_batched_tokens=1024,
                max_num_seqs=512,
                enable_chunked_prefill=True,
                tensor_parallel_size=4,
                )


        # Run workload.
        outputs = run_mt_bench(
                            llm=llm,
                            sampling_params=sampling_params,
                            num_requests=80)

        # Print results / write to file.
        for o in outputs:
            print("\n===========================\n")
            print(f"{o}\n")

        write_responses_to_file(outputs, out_file_path)

