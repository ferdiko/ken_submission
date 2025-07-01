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


def run_vicuna(llm_small, llm_large, cert_thresh, sampling_params, num_requests=-1):
    # Create prompts.
    prompts = []
    benchmark_tasks = get_mt_bench_prompts(vicuna=True)
    for q in benchmark_tasks:
        # TODO: Maybe add system prompt? For multi turn the format is important
        prompts.append(f"Prompt: {q[0]}\nResponse: ")

    if num_requests > 0:
        prompts = prompts[:num_requests]

    # Add prompts to system.
    loop.run_until_complete(asyncio.gather(
                                llm_small.add_new_requests((prompts, sampling_params)),
                                llm_large.add_new_requests((prompts, sampling_params)),
                            ))

    # Dict of request token sequences.
    generated_tokens = {}
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    for i, p in enumerate(prompts):
        generated_tokens[i] = tokenizer.encode(p)

    # Do forward steps.
    num_steps = len(prompts)*1000
    tokens_generated_total = 0
    tokens_generated_large_model = 0

    start = time.time()
    for _ in range(num_steps):
        _, iter_tokens, iter_certainties = llm_small.pause_and_step_sync([])

        for k in iter_tokens.keys():
            if iter_certainties[k] < cert_thresh:
                tokens_generated_large_model += 1

                other_requests = [ki for ki in generated_tokens.keys() if ki != k]

                llm_large.add_tokens_to_request_sync((k, generated_tokens[k]))

                iter_tokens_large = {}
                while len(iter_tokens_large) < 1:
                    # Run prefill until a token is generated.
                    _, iter_tokens_large, _ = llm_large.pause_and_step_sync(other_requests)
                
                generated_tokens[k].append(iter_tokens_large[k])

                llm_large.resume_requests_sync(other_requests)
                llm_small.add_tokens_to_request_sync((k, generated_tokens[k])) # I think this should work since last token is next one to be propagated anyways and we only ever add one.
            else:
                generated_tokens[k].append(iter_tokens[k])

            tokens_generated_total += 1
        
        # TODO: Free sequences if other model in cascade freed them!!
        assert False

    print(f"COST: {tokens_generated_large_model}/{tokens_generated_total}")
    print("TIME:", time.time() - start)

    # Decode tokens into text.
    tasks = []
    for k in generated_tokens.keys():
        response = tokenizer.decode(generated_tokens[k])
        tasks.append(response)

    return tasks


def run_mt_bench(llm, sampling_params, num_requests=-1):
    # Get requests.
    benchmark_tasks = get_mt_bench_prompts(vicuna=False)
    
    max_turns = max([len(t) for t in benchmark_tasks])

    if num_requests > 0:
        benchmark_tasks = benchmark_tasks[:num_requests]

    # Dict of request token sequences.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Init data structures keeping track of request generations.
    # HACK: Different rounds have different request_id's that need to be mapped to same request.
    generated_tokens = {}
    logical_to_physical_id = {}

    # Data structure keeping track of request certainties.
    request_certainties = defaultdict(list)

    system_prompt = "You are a helpful assistant and need to answer user questions based on \
the previous chat history. In the chat history, the user's requests are preceeded by 'Task:' and the responses are preceeded by 'Response:'\n"

    for j, _ in enumerate(benchmark_tasks):
        generated_tokens[j] = tokenizer.encode(system_prompt) # Begin of sequence token
        logical_to_physical_id[j] = j
    

    # Experiment stats and hyperparams.
    tokens_generated_total = 0
    tokens_generated_large_model = 0
    certainty_horizon = 10
    num_steps = len(benchmark_tasks)*1000
    num_steps = 10000
    large_steps = 1 # run large model every N steps
    start = time.time()
    # max_turns = 1

    # Do i-th turn for all requests.
    req_id = 0
    for i in range(max_turns):

        print("\n", "="*20, f"Turn {i}", "\n")

        # 1. Prepare inputs for turn.
        sampling_params.max_tokens += 1000
        added_prompts = []
        added_request_ids = []

        for j, t in enumerate(benchmark_tasks):
            # All turns completed.
            if len(t) <= i:
                continue

            prev_turns = logical_to_physical_id[j]

            # Take previous response and add a turn. (Avoid begin of sequence token)
            generated_tokens[req_id] = generated_tokens[prev_turns] + tokenizer.encode(f"Task: {t[i]}\nResponse: ")[1:]
            added_prompts.append(tokenizer.decode(generated_tokens[req_id][1:])) # HACK: Cleaner to just add tokens, not NL prompt.
            added_request_ids.append(req_id)

            logical_to_physical_id[j] = req_id
            req_id += 1

        # 2. Add prompts to system.
        asyncio.run_coroutine_threadsafe(
            llm.add_new_requests((added_prompts, sampling_params)), loop
        ).result()

        # 3. Process requests (do forward steps)
        num_steps = 3000 # len(added_prompts)*1000
        for step_no in range(num_steps):

            # A. Do step with small LLM.
            _, iter_tokens, iter_certainties = llm.step_sync()
            tokens_generated_total += len(iter_tokens)

            # B. Check if we should cascade or record certainties for later check.
            for k in iter_certainties.keys():
                # request_certainties[k].append(iter_certainties[k])
                generated_tokens[k].append(iter_tokens[k])
                # llm.pause_sync([k])

            # for k in iter_certainties.keys():
            #     llm.resume_requests_sync([k])

        # 4. Prepare for next round: Remove EOS tokens and add new line character.
        end_of_sequence_token = 128009
        for k in generated_tokens.keys():
            if generated_tokens[k][-1] == end_of_sequence_token:
                generated_tokens[k] = generated_tokens[k][:-1]

            text = tokenizer.decode(generated_tokens[k]) + "\n\n" # new line can be part of tokens.
            generated_tokens[k] = tokenizer.encode(text)[1:]


    print(f"COST: {tokens_generated_large_model}/{tokens_generated_total}")
    print("TIME:", time.time() - start)

    # Decode tokens into text.
    tasks = []
    for k in logical_to_physical_id.keys():
        response = tokenizer.decode(generated_tokens[logical_to_physical_id[k]])
        tasks.append(response)

    invocs_string = f"{tokens_generated_large_model}/{tokens_generated_total}"
    return tasks, invocs_string



# def main(loop, model, model2, cert_thresh):
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run simple cascade model with specified parameters.")
    parser.add_argument("--model", type=str, required=True, help="The name of the first model (e.g., '3b').")

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    model = args.model

    # model = "8b"
    # model2 = "70b"
    # cert_thresh = 2.5

    # if cert_thresh.is_integer():
    #     cert_thresh = int(cert_thresh)

    print("\n", "="*10, f"mt_bench_sample_runs/prompted_baseline_{model}.txt", "="*10, "\n")

    if True or not os.path.exists(f"mt_bench_sample_runs/prompted_baseline_{model}.txt"):

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
                                        max_tokens=1000,
                                        # presence_penalty=0,
                                        frequency_penalty=0,
                                        min_tokens=0)

        # Create LLMs.
        llm = LLMWrapper(
                        model_name=models[model], 
                        devices=[0,1,2,3],
                        semaphores=[],
                        memory_frac=0.9,
                        loop=loop
                        )

        try:
            # Run workload.
            outputs, invocs = run_mt_bench(
                                llm=llm,
                                sampling_params=sampling_params,
                                num_requests=20)

            # with open("batch_exp_log_sample.txt", "a+") as f:
            #     f.write(f'prompted_baseline_{model}.txt: {invocs}\n')


            # # print results / write to file.
            # for o in outputs:
            #     print("\n===========================\n")
            #     print(f"{o}\n")

            # write_responses_to_file(outputs, f"mt_bench_sample_runs/prompted_baseline_{model}.txt")
                
            print("COST:", invocs)
            
            # Shutdown and free all memory.
            llm.kill()

        except Exception:
            print(traceback.print_exc())

        finally:
            shutdown_vllm()