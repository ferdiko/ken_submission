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
import threading
import json
from collections import defaultdict


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


def _prepare_finished_seq_for_next_turn(finished_requests, generated_tokens, prompts, benchmark_tasks, tokenizer):
    for k in finished_requests:
        # Remove End-Of-Sequence token if present.
        end_of_sequence_token = 128009
        if generated_tokens[k][-1] == end_of_sequence_token:
            generated_tokens[k] = generated_tokens[k][:-1]

        # Get prompt in text form. Remove Begin-Of-Sequence token. (TODO)
        prompts[k] = tokenizer.decode(generated_tokens[k][1:])

        # Use text string for next turn
        num_tasks = len(benchmark_tasks)
        if k + num_tasks < len(prompts) and prompts[k + num_tasks] is None:
            next_task = f"Task: {benchmark_tasks[k % num_tasks][1]}\nResponse:"
            next_turn_prompt = prompts[k] + "\n\n" + next_task

            prompts[k + num_tasks] = next_turn_prompt
            generated_tokens[k + num_tasks] = tokenizer.encode(next_turn_prompt)


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


def run_mt_bench(llm_small, 
                 llm_large, 
                 cert_thresh_mult, 
                 sampling_params, 
                 queued_thresh_mult, 
                 qps_trace,
                 num_requests=-1,
                 tokens_per_turn=1000):

    # 1. Get requests.
    benchmark_tasks = get_mt_bench_prompts(vicuna=False)
    if num_requests > 0:
        # Might need to repeat tasks so we can issue enough concurrently.
        benchmark_tasks *= int(num_requests / len(benchmark_tasks)) + 1
        benchmark_tasks = benchmark_tasks[:num_requests]
        assert len(benchmark_tasks) == num_requests

    total_queries = int(np.sum(qps_trace))
    num_tasks = len(benchmark_tasks)

    # 2. Prepare prompts and token sequences for each first-turn request.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    system_prompt = "You are a helpful assistant and need to answer user questions based on \
the previous chat history. In the chat history, the user's requests are preceeded by 'Task:' \
and the responses are preceeded by 'Response:'\n\n"

    prompts = [None for _ in range(total_queries)]
    generated_tokens = [None for _ in range(total_queries)]
    for k, task in enumerate(benchmark_tasks):
        k_prompt = system_prompt + f"Task: {task[0]}\nResponse: "
        k_tokens = tokenizer.encode(k_prompt)

        # print("\n\n", "turn", k)
        # print(prompts)
        # print(generated_tokens)

        idx = k
        while idx < total_queries:
            prompts[idx] = k_prompt
            generated_tokens[idx] = k_tokens
            idx += 2 * num_tasks

    print("MEAN:", np.mean([len(g) for g in generated_tokens if g is not None]))
    assert False

    # print(prompts)
    # print(generated_tokens)
    # assert False

    # 3. Init system state and experiment stats.
    req_id = 0
    large_req_queued = 0
    futures_small = []
    futures_large = []

    tokens_generated_total = 0
    tokens_generated_large_model = 0
    token_times = defaultdict(list) # To compute latencies
    start = time.time()

    # 4. Run one second after the other
    no_step = 0
    for qps in qps_trace:
        # print("QPS:", qps)

        # 4.1. Look up qps in gear plan.
        # if qps > 0:
        #     thresh_queued = queued_thresh_mult/qps
        # else:
        #     thresh_queued = 3.5 # TODO
        # cert_thresh = qps*cert_thresh_mult

        cert_thresh = cert_thresh_mult
        thresh_queued = queued_thresh_mult

        # 4.2. Add requests to LLMs.
        sampling_params.max_tokens += tokens_per_turn
        added_req_ids = [r for r in range(req_id, req_id + qps)]
        added_prompts = [prompts[r] for r in added_req_ids]
        assert None not in added_prompts
        req_id += qps

        future1 = asyncio.run_coroutine_threadsafe(
            llm_small.add_new_requests((added_prompts, sampling_params)), loop
        )
        future2 = asyncio.run_coroutine_threadsafe(
            llm_large.add_new_requests((added_prompts, sampling_params)), loop
        )
        result1 = future1.result()
        result2 = future2.result()

        llm_large.pause_sync(added_req_ids)

        # 4.3. Step for 1 second.
        start_second = time.time()
        while time.time() - start_second < 1:
            no_step += 1

            # 4.3.1. Do step with small LLM.
            for f in futures_small:
                f.result()
            futures_small = []

            _, iter_tokens, iter_certainties = llm_small.step_sync()
            tokens_generated_total += len(iter_tokens)

            # 4.3.2. If request finishes, also free it in other llm. Prepare next turn.
            # Note: If small LLM is uncertain, it will still be freed.
            finished_requests = llm_small.get_finished_request_ids_sync()
            if len(finished_requests) > 0:
                futures_large.append(
                    asyncio.run_coroutine_threadsafe(
                        llm_large.free_requests(finished_requests), loop
                    )
                )

                _prepare_finished_seq_for_next_turn(finished_requests, 
                                                    generated_tokens, 
                                                    prompts, 
                                                    benchmark_tasks,
                                                    tokenizer)

            # 4.3.3. Cascade uncertain generations.
            uncertain_keys = [k for k in iter_tokens.keys() if iter_certainties[k] < cert_thresh
                and k not in finished_requests]
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

            # llm_small.pause_sync(uncertain_keys)
            # llm_large.resume_requests_sync(uncertain_keys)
            # llm_large.add_tokens_to_request_sync(uncertain_keys, uncertain_tok_seqs)
            
            large_req_queued += len(uncertain_keys)

            # 4.3.4. Append certain generations to tok sequence.
            certain_keys = [k for k in iter_tokens.keys() if iter_certainties[k] >= cert_thresh
                and k not in finished_requests]

            for k in iter_tokens.keys():
                generated_tokens[k].append(iter_tokens[k])
                token_times[k].append(time.time())

            # 4.3.5. Trigger large LLM if enough requests queued.
            if no_step % thresh_queued == 0: #large_req_queued >= thresh_queued: # TODO: What's a good policy here?
                # if step_no % large_interval == 0:
                # 4.3.5.1. Make inference step (run until a token is generated for each request).
                iter_tokens_large = {}
                finished_requests = []

                for f in futures_large:
                    f.result()
                futures_large = []

                while len(iter_tokens_large) < large_req_queued:
                    _, new_tokens, _ = llm_large.step_sync()
                    finished_requests += llm_large.get_finished_request_ids_sync()
                    llm_large.pause_sync([k for k in new_tokens.keys()])
                    iter_tokens_large.update(new_tokens)

                tokens_generated_large_model += len(iter_tokens_large)

                # 4.3.5.2. If request is finished, also finish in small LLM.
                if len(finished_requests) > 0:
                    futures_small.append(
                        asyncio.run_coroutine_threadsafe(
                            llm_small.free_requests(finished_requests), loop
                        )
                    )


                    # llm_small.free_requests_sync(finished_reqs)

                for k in finished_requests:
                    token_times[k].append(time.time())

                _prepare_finished_seq_for_next_turn(finished_requests, 
                                                    generated_tokens, 
                                                    prompts, 
                                                    benchmark_tasks,
                                                    tokenizer)

                # 4.3.5.3. Pause requests on large LLM, resume on small LLM.
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

                # 4.3.5.4. Add tokens to local sequences.
                for k in iter_tokens_large.keys():
                    if k not in finished_requests:
                        generated_tokens[k].append(iter_tokens_large[k])
                        token_times[k].append(time.time())

                # 4.3.5.5. Add tokens to small LLM.
                resume_req_ids = [k for k in iter_tokens_large.keys() if k not in finished_requests]
                resume_tok_seqs = [generated_tokens[k] for k in resume_req_ids]
                futures_small.append(
                    asyncio.run_coroutine_threadsafe(
                        llm_small.add_tokens_to_request((resume_req_ids, resume_tok_seqs)), loop
                    )
                )
                # llm_small.add_tokens_to_request_sync(resume_req_ids, resume_tok_seqs) # I think this should work since last token is next one to be propagated anyways and we only ever add one.

    print(f"COST: {tokens_generated_large_model}/{tokens_generated_total}")
    print("TIME:", time.time() - start)

    invocs_string = f"{tokens_generated_large_model}/{tokens_generated_total}"
    return prompts, invocs_string, token_times


if __name__ == "__main__":
    # 1. Get experiment hyper params.
    parser = argparse.ArgumentParser(description="Run simple cascade model with specified parameters.")
    parser.add_argument("--model1", type=str, required=True, help="The name of the first model (e.g., '3b').")
    parser.add_argument("--model2", type=str, required=True, help="The name of the second model (e.g., '70b').")
    # parser.add_argument("--cert-thresh", type=float, required=True, help="The certification threshold (e.g., 3).")
    parser.add_argument("--cert-thresh-mult", type=float, required=True, help="The certification threshold (e.g., 3).")
    parser.add_argument("--queued-thresh-mult", type=float, required=True, help="The certification threshold (e.g., 3).")
    parser.add_argument("--qps-mult", type=float, required=True, help="The certification threshold (e.g., 3).")

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    model1 = args.model1
    model2 = args.model2
    cert_thresh_mult = args.cert_thresh_mult
    queued_thresh_mult = args.queued_thresh_mult
    qps_mult = args.qps_mult

    # model1 = "1b"
    # model2 = "1b"
    # cert_thresh = 2
    # thresh_queued = 1

    output_file_path = f"first_batch_e2e/qps{qps_mult}_batch{queued_thresh_mult}_cert{cert_thresh_mult}_{model1}_{model2}_2x2k_180s.txt"
    out_times_path = "first_batch_e2e_times/" + output_file_path.split("/")[-1]

    # 2. Read QPS trace.
    def replace_value(val):
        if val > 1:
            return [val]
        else: 
            n = int(1 / val)
            return [1] + [0] * (n - 1)

    qps_trace = np.loadtxt("twitter_qps.txt")*qps_mult
    qps_trace = np.concatenate([replace_value(x) for x in qps_trace])
    qps_trace = qps_trace[:60]
    qps_trace = np.append(qps_trace, np.zeros(10))
    qps_trace = np.array(qps_trace, dtype=int)

    qps_trace = np.append([80], np.zeros(50))
    qps_trace = np.array(qps_trace, dtype=int)

    # 3. Run experiment if it hasn't been run already
    if True or not os.path.exists(output_file_path):
        print("\n", "Running:", output_file_path, "\n")

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
                                         presence_penalty=0,
                                         frequency_penalty=0,
                                         min_tokens=0)

        if model1 == "8b":
            small_frac = 0.25
        else:
            small_frac = 0.15

        llm_1 = LLMWrapper(model_name=models[model1],
                           devices=[0,1,2,3],
                           semaphores=[],
                        #    memory_frac=0.85,
                           memory_frac=small_frac,
                           loop=loop)

        llm_2 = LLMWrapper(model_name=models[model2],
                           devices=[0,1,2,3],
                           semaphores=[],
                        #    memory_frac=0.15,
                           memory_frac=1-small_frac,
                           loop=loop)

        # 3. Run workload.
        try:
            # 3.1. Run workload.
            outputs, invocs, token_times = run_mt_bench(llm_small=llm_1,
                                                        llm_large=llm_2,
                                                        cert_thresh_mult=cert_thresh_mult,
                                                        sampling_params=sampling_params,
                                                        qps_trace=qps_trace,
                                                        num_requests=max(300, int(np.sum(qps_trace)/2)),
                                                        queued_thresh_mult=queued_thresh_mult,
                                                        tokens_per_turn=300,
                                                        )

            # 3.2. Log cost.
            with open("batch_exp_log_async.txt", "a+") as f:
                f.write(f'{output_file_path}: {invocs}\n')

            with open(out_times_path, 'w') as f:
                json.dump(token_times, f, indent=2)

            # 3.3. Print generations / write to file.
            # for o in outputs:
            #     print("\n===========================\n")
            #     print(f"{o}\n")

            write_responses_to_file(outputs, output_file_path)

            # 3.4. Shut down and free all memory.
            llm_1.kill()
            llm_2.kill()

        except Exception:
            print(traceback.print_exc())

        finally:
            shutdown_vllm()
