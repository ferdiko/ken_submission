from vllm import LLM, SamplingParams
from runner_utils import *
import time
import asyncio
from transformers import AutoTokenizer
import numpy as np
import psutil
import argparse
import os
import numpy as np
import signal
import traceback
from collections import defaultdict
import threading
import json


def shutdown_vllm():
    """
    vLLM doesn't shut down properly. This is a hack to free up all memory
    """
    time.sleep(1)
    processes = [p for p in psutil.process_iter(['pid', 'name']) if p.info['name'] == 'pt_main_thread']
    processes_sorted = sorted(processes, key=lambda p: p.info['pid'], reverse=True)
    
    # Kill each process
    for process in processes_sorted:
        pid = process.info['pid']
        try:
            os.kill(pid, signal.SIGTERM)  # or signal.SIGKILL for a forced kill
        except Exception as e:
            print(f"Could not kill process with PID {pid}: {e}")


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


def run_mt_bench(llm_small, llm_large, cert_thresh, sampling_params, casc_thresh, num_requests=-1):

    final_cert_dict = {}

    # Get requests.
    benchmark_tasks = get_mt_bench_prompts(vicuna=False)
    
    max_turns = max([len(t) for t in benchmark_tasks])

    if num_requests > 0:
        benchmark_tasks *= num_requests
        benchmark_tasks = benchmark_tasks[:num_requests]

    # Dict of request token sequences.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Init data structures keeping track of request generations.
    # HACK: Different rounds have different request_id's that need to be mapped to same request.
    generated_tokens_small = {}
    generated_tokens_large = {}

    # Data structure keeping track of request certainties.
    request_certainties = defaultdict(list)

    system_prompt = "Answer user questions based on the previous chat history. In the chat history, \
the user's requests are preceeded by 'Task:' and the responses are preceeded by 'Response:'. Just fill \
in the last response field. Be concise.\n\n"

    # Experiment stats and hyperparams.
    tokens_generated_total = 0
    tokens_generated_large_model = 0
    certainty_horizon = 1
    num_steps = 1000 #len(benchmark_tasks)*1000
    # num_steps = 100
    large_steps = 1 # run large model every N steps
    start = time.time()

    # Init data structures that hold responses. 
    # HACK: Could be easier since, in the end, we just step through the stuff normally.
    prompts = [None for _ in range(len(benchmark_tasks)*max_turns)]
    generated_tokens_small = [None for _ in range(len(benchmark_tasks)*max_turns)] # TODO: total queries with qps
    generated_tokens_large = [None for _ in range(len(benchmark_tasks)*max_turns)] # TODO: total queries with qps
    final_responses = []

    # large_to_small_id = {} # Do this with pointers.

    for k, task in enumerate(benchmark_tasks):
        k_prompt = system_prompt
        # k_tokens = tokenizer.encode(k_prompt)

        prompts[k] = k_prompt
        # generated_tokens[k] = k_tokens


    # Do i-th turn for all requests.
    max_turns = 2 # TODO: If you change that, see bug below (in prepare for next turn)
    req_id_small = 0
    req_id_large = 0
    first_turn_id = 0 # first (small) request id used for this run.
    for turn_no in range(max_turns):
        print("\n", "="*20, f"Turn {turn_no}", "\n")
        # sampling_params.max_tokens += 300

        # 1. Add prompt for next turn.
        add_prompts = []
        for task in benchmark_tasks:
            if turn_no < len(task):
                prompts[req_id_small] += f"Task: {task[turn_no]}\nResponse: "
                add_prompts.append(prompts[req_id_small])
                generated_tokens_small[req_id_small] = tokenizer.encode(prompts[req_id_small])
                req_id_small += 1

        # 2. Add prompts to system.
        llm_small.add_new_requests_sync((add_prompts, sampling_params))

        # 3. Process requests (do forward steps).
        for step_no in range(num_steps):

            # print("step", step_no)

            # A. Do step with small LLM.
            _, iter_tokens, iter_certainties = llm_small.step_sync()
            tokens_generated_total += len(iter_tokens)

            # B. Check if we should cascade or record certainties for later check.
            for k in iter_certainties.keys():
                request_certainties[k].append(iter_certainties[k])
                generated_tokens_small[k].append(iter_tokens[k])

                # Check if sample is cascaded.
                if len(request_certainties[k]) == certainty_horizon: # and np.mean(request_certainties[k]) < cert_thresh:
                    # print("cascade", k)

                    final_cert_dict[k] = np.prod(request_certainties[k])

            #         generated_tokens_large[req_id_large] = tokenizer.encode(prompts[k])
            #         generated_tokens_small[k] = generated_tokens_large[req_id_large] # Pointer to actual tokens that will be used.

            #         llm_large.add_new_requests_sync(([prompts[k]], sampling_params)) # TODO: Make async
            #         llm_small.free_requests_sync([k]) # TODO: Make async
            #         req_id_large += 1

            # # C. Make step with large model.
            # if step_no % large_steps == 0:
            #     _, iter_tokens, _ = llm_large.step_sync()

            #     for k in iter_tokens.keys():
            #         generated_tokens_large[k].append(iter_tokens[k])

            #     tokens_generated_large_model += len(iter_tokens)


        # 4. Prepare for next round: Remove EOS tokens and add new line character.
        end_of_sequence_token = 128009
        for i, generated_tokens in enumerate(generated_tokens_small[first_turn_id:req_id_small]):
            assert generated_tokens is not None
            final_responses.append(tokenizer.decode(generated_tokens[1:]))

            if turn_no + 1 < len(benchmark_tasks[i]):
                # Prepare next round's prompt. Remove Start-Of-Seq token.
                prompts[req_id_small + i] = tokenizer.decode(generated_tokens[1:]) + "\n\n" # new line can be part of tokens.
        
        first_turn_id = req_id_small


    with open("mt_bench_single_model_samples/certs_70b_norm_prod_h1.json", "w") as f:
        json.dump(final_cert_dict, f, indent=2)

    assert False

    print(f"COST: {tokens_generated_large_model}/{tokens_generated_total}")
    print("TIME:", time.time() - start)

    invocs_string = f"{tokens_generated_large_model}/{tokens_generated_total}"
    return final_responses, invocs_string



# def main(loop, model1, model2, cert_thresh):
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run simple cascade model with specified parameters.")
    parser.add_argument("--model1", type=str, required=True, help="The name of the first model (e.g., '3b').")
    parser.add_argument("--model2", type=str, required=True, help="The name of the second model (e.g., '70b').")
    parser.add_argument("--cert-thresh", type=float, required=True, help="The certification threshold (e.g., 3).")

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    model1 = args.model1
    model2 = args.model2
    cert_thresh = args.cert_thresh

    # model1 = "8b"
    # model2 = "70b"
    # cert_thresh = 2.5

    # if cert_thresh.is_integer():
    #     cert_thresh = int(cert_thresh)

    # output_file_path = f"r_mt_bench_single_model_samples/2x1600_{model1}_{cert_thresh}_{model2}.txt"
    output_file_path = f"debug_mt_bench_single_model_samples/2x1600_{model1}.txt"

    if not os.path.exists(output_file_path):
        print("\n", "Running:", output_file_path, "\n")

        # 1. Set up asyncio and threading environment
        def start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=start_loop, args=(loop,), daemon=True)
        loop_thread.start()

        models = {
            "1b": "meta-llama/Llama-3.2-1B-Instruct",
            "3b": "meta-llama/Llama-3.2-3B-Instruct",
            "8b": "meta-llama/Llama-3.1-8B-Instruct",
            "70b": "meta-llama/Llama-3.1-70B-Instruct"
        }

        # Create a sampling params object.
        sampling_params = SamplingParams(best_of=1,
                                        temperature=0.0,
                                        top_p=1,
                                        top_k=-1,
                                        max_tokens=1,
                                        # presence_penalty=0,
                                        frequency_penalty=0,
                                        min_tokens=0)

        # Create LLMs.
        llm_1 = LLMWrapper(
                        model_name=models[model1], 
                        devices=[0,1,2,3],
                        semaphores=[],
                        memory_frac=0.8,
                        loop=loop
                        )

        llm_2 = LLMWrapper(
                        model_name=models[model2], 
                        devices=[0,1,2,3],
                        semaphores=[],
                        memory_frac=0.2,
                        loop=loop
                        )

        try:
            # Run workload.
            outputs, invocs = run_mt_bench(
                                llm_small=llm_1, 
                                llm_large=llm_2, 
                                cert_thresh=cert_thresh,
                                sampling_params=sampling_params,
                                num_requests=80)

            # with open("batch_exp_log_sample.txt", "a+") as f:
            #     f.write(f'{output_file_path}: {invocs}\n')


            # Print results / write to file.
            # for o in outputs:
            #     print("\n===========================\n")
            #     print(f"{o}\n")

            write_responses_to_file(outputs, output_file_path)
                
            print("COST:", invocs)
            
            # Shutdown and free all memory.
            llm_1.kill()
            llm_2.kill()

        except Exception:
            print(traceback.print_exc())

        finally:
            shutdown_vllm()