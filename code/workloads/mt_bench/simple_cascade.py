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
# from concurrent.futures import Future
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


def step(llm, futures):
    for f in futures:
        f.result()

    return llm.step_sync()


def run_mt_bench(llm_small, llm_large, cert_thresh, sampling_params, thresh_queued, num_requests=-1):
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

    system_prompt = "You are a helpful assistant and need to answer user questions based on \
the previous chat history. In the chat history, the user's requests are preceeded by 'Task:' and the responses are preceeded by 'Response:'\n"

    for j, _ in enumerate(benchmark_tasks):
        generated_tokens[j] = tokenizer.encode(system_prompt) # Begin of sequence token
        logical_to_physical_id[j] = j


    # Experiment stats and hyperparams.
    tokens_generated_total = 0
    tokens_generated_large_model = 0
    large_interval = 4
    start = time.time()

    # Do i-th turn for all requests.
    # max_turns = 1
    req_id = 0
    large_req_queued = 0

    # Futures of LLM before step can be called.
    futures_small = []
    futures_large = []

    for i in range(max_turns):

        print("\n", "="*20, f"Turn {i}", "\n")

        # 1. Prepare inputs for turn.
        sampling_params.max_tokens += 2000
        added_prompts = []
        added_request_ids = []

        for j, t in enumerate(benchmark_tasks):
            # A. All turns completed.
            if len(t) <= i:
                continue

            prev_turns = logical_to_physical_id[j]

            # B. Take previous response and add a turn. (Avoid begin of sequence token)
            generated_tokens[req_id] = generated_tokens[prev_turns] + tokenizer.encode(f"Task: {t[i]}\nResponse: ")[1:]
            added_prompts.append(tokenizer.decode(generated_tokens[req_id][1:])) # HACK: Cleaner to just add tokens, not NL prompt.
            added_request_ids.append(req_id)

            logical_to_physical_id[j] = req_id
            req_id += 1

        # 2. Add prompts to LLMs.
        future1 = asyncio.run_coroutine_threadsafe(
            llm_small.add_new_requests((added_prompts, sampling_params)), loop
        )
        future2 = asyncio.run_coroutine_threadsafe(
            llm_large.add_new_requests((added_prompts, sampling_params)), loop
        )
        result1 = future1.result()
        result2 = future2.result()

        llm_large.pause_sync(added_request_ids)

        # 3. Process requests (do forward steps)
        num_steps = len(added_prompts)*1000
        for step_no in range(num_steps):

            # A. Do step with small LLM.
            _, iter_tokens, iter_certainties = step(llm_small, futures_small) # llm_small.step_sync()
            futures_small = []
            tokens_generated_total += len(iter_tokens)

            # B. If request finishes, also free it in other llm.
            # NOTE: If small LLM is uncertain, it will still be freed.
            finished_requests = llm_small.get_finished_request_ids_sync()
            if len(finished_requests) > 0:
                futures_large.append(
                    asyncio.run_coroutine_threadsafe(
                        llm_large.free_requests(finished_requests), loop
                    )
                )

                # llm_large.free_requests_sync(finished_requests) # TODO Now: Do with future

            # C. Cascade uncertain generations.
            uncertain_keys = [k for k in iter_tokens.keys() if iter_certainties[k] < cert_thresh
                and str(k) not in finished_requests]
            uncertain_tok_seqs = [generated_tokens[k] for k in uncertain_keys]

            futures_small.append(
                asyncio.run_coroutine_threadsafe(
                    llm_small.pause(uncertain_keys), loop
                )
            )

            futures_large.append(
                asyncio.run_coroutine_threadsafe(
                    llm_large.resume_requests(uncertain_keys), loop
                )
            )

            futures_large.append(
                asyncio.run_coroutine_threadsafe(
                    llm_large.add_tokens_to_request((uncertain_keys, uncertain_tok_seqs)), loop
                )
            )

            # llm_small.pause_sync(uncertain_keys) # TODO: Make async
            # llm_large.resume_requests_sync(uncertain_keys) # TODO: Make sync
            # llm_large.add_tokens_to_request_sync(uncertain_keys, uncertain_tok_seqs) # TODO: Make sync
            large_req_queued += len(uncertain_keys)

            # D. Append certain generations to tok seq.
            certain_keys = [k for k in iter_tokens.keys() if iter_certainties[k] >= cert_thresh
                and str(k) not in finished_requests]

            for k in iter_tokens.keys():
                generated_tokens[k].append(iter_tokens[k])

            # E. Trigger large LLM if enough requests queued.
            if large_req_queued >= large_interval: # TODO: What' a good policy here?
                # if step_no % large_interval == 0:

                # E.1. Make inference step (run until a token is generated for each request).
                iter_tokens_large = {}
                finished_reqs = []

                for f in futures_large:
                    f.result()
                futures_large = []

                while len(iter_tokens_large) < large_req_queued:
                    _, new_tokens, _ = llm_large.step_sync()
                    finished_reqs += llm_large.get_finished_request_ids_sync()
                    llm_large.pause_sync([k for k in new_tokens.keys()])
                    iter_tokens_large.update(new_tokens)

                tokens_generated_large_model += len(iter_tokens_large)

                # E.2. If request is finished, also finish in small LLM.
                futures_small.append(
                    asyncio.run_coroutine_threadsafe(
                        llm_small.free_requests(finished_reqs), loop
                    )
                )
                # llm_small.free_requests_sync(finished_reqs) # TODO: Make async

                # E.3. Pause requests on large LLM, resume on small LLM. TODO: Make async
                futures_large.append(
                    asyncio.run_coroutine_threadsafe(
                        llm_large.pause([k for k in iter_tokens_large.keys()]), loop
                    )
                )
                # llm_large.pause_sync([k for k in iter_tokens_large.keys()])
                futures_small.append(
                    asyncio.run_coroutine_threadsafe(
                        llm_small.resume_requests([k for k in iter_tokens_large.keys()]), loop
                    )
                )
                # llm_small.resume_requests_sync([k for k in iter_tokens_large.keys()])
                large_req_queued = 0

                # E.4. Add tokens to local sequences.
                for k in iter_tokens_large.keys():
                    generated_tokens[k].append(iter_tokens_large[k])

                # E.5. Add all tokens to small LLM and resume. TODO: Make async
                resume_req_ids = [k for k in iter_tokens_large.keys() if k not in finished_reqs]
                resume_tok_seqs = [generated_tokens[k] for k in resume_req_ids]
                futures_small.append(
                    asyncio.run_coroutine_threadsafe(
                        llm_small.add_tokens_to_request((resume_req_ids, resume_tok_seqs)), loop
                    )
                )
                # llm_small.add_tokens_to_request_sync(resume_req_ids, resume_tok_seqs) # I think this should work since last token is next one to be propagated anyways and we only ever add one.


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

    # model1 = "1b"
    # model2 = "70b"
    # cert_thresh = 2
    thresh_queued = 1

    # if cert_thresh.is_integer():
    #     cert_thresh = int(cert_thresh)

    out_file_path = f"mt_bench_runs_concise/casc_{model1}_{cert_thresh}_{model2}_2x2k.txt"
    print("\n", "="*10, out_file_path, "="*10, "\n")

    if not os.path.exists(out_file_path):

        # 1. Set up asyncio and threading environment
        def start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=start_loop, args=(loop,), daemon=True)
        loop_thread.start()

        # 2. Spawn LLMs with correct parameters.
        models = {
            "1b": "meta-llama/Llama-3.2-1B-Instruct",
            "3b": "meta-llama/Llama-3.2-3B-Instruct",
            "8b": "meta-llama/Llama-3.1-8B-Instruct",
            "70b": "meta-llama/Llama-3.1-70B-Instruct"
        }

        sampling_params = SamplingParams(best_of=1,
                                        temperature=0.0,
                                        top_p=1,
                                        top_k=-1,
                                        max_tokens=1,
                                        # presence_penalty=0,
                                        frequency_penalty=0,
                                        min_tokens=0)

        llm_1 = LLMWrapper(
                            model_name=models[model1],
                            devices=[0,1,2,3],
                            semaphores=[],
                            memory_frac=0.9,
                            loop=loop
                            )

        llm_2 = LLMWrapper(
                            model_name=models[model2],
                            devices=[4],
                            semaphores=[],
                            memory_frac=0.9,
                            loop=loop
                            )

        # 3. Run workload.
        try:
            outputs, invocs = run_mt_bench(
                                llm_small=llm_1,
                                llm_large=llm_2,
                                cert_thresh=cert_thresh,
                                sampling_params=sampling_params,
                                num_requests=10,
                                thresh_queued=thresh_queued)

            with open("batch_exp_log_concise.txt", "a+") as f:
                f.write(f'{out_file_path}: {invocs}\n')


            # # print results / write to file.
            # for o in outputs:
            #     print("\n===========================\n")
            #     print(f"{o}\n")

            write_responses_to_file(outputs, out_file_path)

            # Shutdown and free all memory.
            llm_1.kill()
            llm_2.kill()

        except Exception:
            print(traceback.print_exc())

        finally:
            shutdown_vllm()