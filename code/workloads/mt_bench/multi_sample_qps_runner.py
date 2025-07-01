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
import sys


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


def log_times(iter_tokens, token_times):
    cur_time = time.time()
    for req_id in iter_tokens:
        token_times[req_id].append(cur_time)


def run_mt_bench(llm_small_1, 
                 llm_small_2, 
                 llm_large_1,
                 llm_large_2, 
                 cert_thresh, 
                 trigger_2,
                 sampling_params, 
                 queued_thresh,
                 queued_thresh_mult,
                 qps_trace,
                 small_iters,
                 ):

    cert_thresh_plan = cert_thresh

    # 1. Get requests.
    first_turn = get_first_turn_prompts()
    second_turn = get_second_turn_prompts()
    benchmark_tasks = first_turn + second_turn # NOTE: Is it good to just append here?
    # benchmark_tasks = [item for pair in zip(first_turn, second_turn) for item in pair]


    with open("mt_bench_single_model_samples/certs_3b_norm_prod_h1.json", "r") as f:
        cert_dict = {int(k): float(v) for k, v in json.load(f).items()}
    assert len(cert_dict) == len(benchmark_tasks)

    prompts = []
    certainties = []
    total_queries = int(sum(qps_trace))
    for i in range(total_queries):
        prompts.append(benchmark_tasks[i%len(benchmark_tasks)])
        certainties.append(cert_dict[i%len(benchmark_tasks)])

    # 1.1 Record num prefill tokens to use for gear plan decisions.
    num_tokens = []
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    num_tokens = [len(tokenizer.encode(p)) for p in prompts]

    # 3. Init system state and experiment stats.
    processed_until = 0
    futures_small = []
    futures_large = []

    evaluated_requests = set()

    # 5. Log metrics
    token_times = {} # To compute latencies
    request_models = {} # req_id -> model used

    # 4. Run one second after the other
    print("Start start running")
    for debug_idx, qps in enumerate(qps_trace):
        print("iter", debug_idx, "QPS:", qps)

        # 4.1. Look up qps in gear plan.
        queued_tokens_1 = asyncio.run_coroutine_threadsafe(
                                    llm_large_1.num_queued_tokens(), loop
                                )

        queued_tokens_2 = asyncio.run_coroutine_threadsafe(
                                    llm_large_2.num_queued_tokens(), loop
                                )

        queued_tokens_small_1 = asyncio.run_coroutine_threadsafe(
                                    llm_small_1.num_queued_tokens(), loop
                                )

        queued_tokens_small_2 = asyncio.run_coroutine_threadsafe(
                                    llm_small_2.num_queued_tokens(), loop
                                )

        if True or (queued_thresh <= 0 or queued_thresh > 100000):
            # 4.2. Determine which requests to add.
            added_req_ids_1 = []
            added_req_ids_2 = []
            for r in range(processed_until, processed_until + qps, 2):
                added_req_ids_1.append(r)
                if r + 1 < processed_until + qps:
                    added_req_ids_2.append(r+1)

            queued_tokens_1 = queued_tokens_1.result()
            queued_tokens_2 = queued_tokens_2.result()

            # added_req_ids = [r for r in range(processed_until, processed_until + qps)]

            # queued_tokens_1 = max(queued_tokens_1.result(), 1)
            # queued_tokens_2 = max(queued_tokens_2.result(), 1)

            # added_req_ids_1 = added_req_ids[:int(queued_tokens_1/(queued_tokens_1+queued_tokens_2)*len(added_req_ids))]
            # added_req_ids_2 = added_req_ids[int(queued_tokens_1/(queued_tokens_1+queued_tokens_2)*len(added_req_ids)):]


        else:
            added_req_ids = [r for r in range(processed_until, processed_until + qps)]

            queued_tokens_1 = queued_tokens_1.result()
            queued_tokens_2 = queued_tokens_2.result()

            added_req_ids_1 = []
            added_req_ids_2 = []
            
            sum_tokens_1 = queued_tokens_1
            sum_tokens_2 = queued_tokens_2
            
            for r_id in added_req_ids:
                tokens = num_tokens[r_id]
                # Assign the request to the queue with the smaller token sum
                if sum_tokens_1 <= sum_tokens_2:
                    added_req_ids_1.append(r_id)
                    sum_tokens_1 += tokens
                else:
                    added_req_ids_2.append(r_id)
                    sum_tokens_2 += tokens



        added_prompts_1 = [prompts[r] for r in added_req_ids_1]
        added_prompts_2 = [prompts[r] for r in added_req_ids_2]

        print("Added 1:", len(added_prompts_1), len(added_req_ids_1))
        print("Added 2:", len(added_prompts_2), len(added_req_ids_2))
        print()

        large_tokens_queued = sum(num_tokens[r] for r in added_req_ids_1 + added_req_ids_2)
        large_tokens_queued += queued_tokens_1 + queued_tokens_2

        # print(f"QUEUED: decision: {large_tokens_queued}, small:({queued_tokens_small_1.result(), queued_tokens_small_2.result()}, large({queued_tokens_1},{queued_tokens_2})")
        print(f"Queued: total: {large_tokens_queued}, 1: {queued_tokens_1} 2: {queued_tokens_2}")
        if large_tokens_queued > queued_thresh*trigger_2:
            cert_thresh = cert_thresh_plan/queued_thresh_mult
        elif large_tokens_queued > queued_thresh:
            cert_thresh = cert_thresh_plan
        else:
            cert_thresh = 0

        print("Cert thresh:", cert_thresh)

        # Add tokens to models.
        cur_time = time.time()
        for r_id in range(processed_until, processed_until + qps):
            token_times[r_id] = [cur_time]

        if cert_thresh > 0:
            futures_small.append(asyncio.run_coroutine_threadsafe(
                llm_small_1.add_new_requests((added_prompts_1, sampling_params, added_req_ids_1)), loop
            ))

            futures_small.append(asyncio.run_coroutine_threadsafe(
                llm_small_2.add_new_requests((added_prompts_2, sampling_params, added_req_ids_2)), loop
            ))
            
        else:
            futures_large.append(asyncio.run_coroutine_threadsafe(
                llm_large_1.add_new_requests((added_prompts_1, sampling_params, added_req_ids_1)), loop
            ))

            futures_large.append(asyncio.run_coroutine_threadsafe(
                llm_large_2.add_new_requests((added_prompts_2, sampling_params, added_req_ids_2)), loop
            ))

            for req_id in added_req_ids_1 + added_req_ids_2:
                request_models[req_id] = "large"

        processed_until += qps

        # 4.3. Step for 1 second.
        start_second = time.time()
        while time.time() - start_second < 1:

            # 4.3.1. Do step with small LLM.
            if cert_thresh > 0:
                # Make a step with small LLMs.
                for f in futures_small:
                    f.result()
                futures_small = []

                start = time.time()
                future_1 = asyncio.run_coroutine_threadsafe(
                    llm_small_1.step(), loop
                )

                future_2 = asyncio.run_coroutine_threadsafe(
                    llm_small_2.step(), loop
                )

                # Cascade iter tokens of first small LLM
                for future_idx, future in enumerate([future_1, future_2]):
                    iter_tokens = future.result()

                    # If it's the first token generated by the LLM, decide whether to cascade or not.
                    cascaded_req_ids = [req_id for req_id in iter_tokens if req_id 
                        not in evaluated_requests and certainties[req_id] < cert_thresh]
                    casc_prompts = [prompts[k] for k in cascaded_req_ids]

                    # print(f"future({future_idx})", iter_tokens, cascaded_req_ids)

                    if future_idx == 0:
                        futures_large.append(asyncio.run_coroutine_threadsafe(
                                llm_large_1.add_new_requests((casc_prompts, sampling_params, cascaded_req_ids)), loop))

                        futures_small.append(asyncio.run_coroutine_threadsafe(
                                llm_small_1.free_requests(cascaded_req_ids), loop))

                    else:
                        futures_large.append(asyncio.run_coroutine_threadsafe(
                                llm_large_2.add_new_requests((casc_prompts, sampling_params, cascaded_req_ids)), loop))

                        futures_small.append(asyncio.run_coroutine_threadsafe(
                                llm_small_2.free_requests(cascaded_req_ids), loop))


                    # Record whether large or small model will predict request.
                    cur_time = time.time()
                    for req_id in iter_tokens:
                        if req_id in evaluated_requests:
                            continue

                        if certainties[req_id] < cert_thresh:
                            request_models[req_id] = "large"
                        else:
                            token_times[req_id].append(cur_time) # Log generation time.
                            request_models[req_id] = "small"

                    evaluated_requests.update(iter_tokens)

                    print(f"small iter_tokens (cert {cert_thresh}):", len(iter_tokens))
                print("Small with cascade:", time.time() - start)

            else: # Step to finish pending requests. If no requests: 0.3ms
                # Wait for all futures.
                for f in futures_small:
                    f.result()
                futures_small = []

                # Do several iters so no additional small pass when next prefills arrive.
                start = time.time()
                for _ in range(small_iters):
                    # 1. Step with both LLMs.
                    future_1 = asyncio.run_coroutine_threadsafe(
                        llm_small_1.step(), loop
                    )

                    future_2 = asyncio.run_coroutine_threadsafe(
                        llm_small_2.step(), loop
                    )

                    # 2. Log times.
                    for future in [future_1, future_2]:
                        iter_tokens = future.result()
                        log_times(iter_tokens, token_times)

                        print(f"small iter_tokens ({cert_thresh}):", len(iter_tokens))
                    print("Small with cascade:", time.time() - start)

            # 4.3.2. Do step with large LLM.
            for f in futures_large:
                f.result()
            futures_large = []

            start = time.time()
            # 1. Step with both LLMs.
            future_1 = asyncio.run_coroutine_threadsafe(
                llm_large_1.step(), loop
            )

            future_2 = asyncio.run_coroutine_threadsafe(
                llm_large_2.step(), loop
            )

            # 2. Log times.
            # TODO: Bug: Only because it's scheduled doesn't mean prefill is done ...
            for future in [future_1, future_2]:
                iter_tokens = future.result()
                log_times(iter_tokens, token_times)

                print(f"large tokens {len(iter_tokens)}")
            print("Large time:", time.time() - start)
            # print("large:", start-time.time())
            # print()

    return request_models, token_times


if __name__ == "__main__":
    # 1. Get experiment hyper params.
    parser = argparse.ArgumentParser(description="Run simple cascade model with specified parameters.")
    parser.add_argument("--model1", type=str, required=False, default="3b", help="The name of the first model (e.g., '3b').")
    parser.add_argument("--model2", type=str, required=False, default="70b", help="The name of the second model (e.g., '70b').")
    parser.add_argument("--cert-thresh", type=float, required=True, help="The certification threshold (e.g., 3).")
    parser.add_argument("--queued-thresh", type=float, required=True, help="The certification threshold (e.g., 3).")
    parser.add_argument("--qps-mult", type=float, required=False, default=0.55, help="The certification threshold (e.g., 3).")
    parser.add_argument("--tokens-per-turn", type=int, required=False, default=1600, help="The certification threshold (e.g., 3).")
    parser.add_argument("--num-secs", type=int, required=False, default=42, help="The certification threshold (e.g., 3).")
    parser.add_argument("--small-iters", type=int, required=False, default=1, help="The certification threshold (e.g., 3).")
    parser.add_argument("--queued-thresh-mult", type=float, required=False, default=1, help="The certification threshold (e.g., 3).")
    parser.add_argument("--trigger2", type=float, required=True, default=1, help="The certification threshold (e.g., 3).")


    args = parser.parse_args()

    model1 = args.model1
    model2 = args.model2
    cert_thresh = args.cert_thresh
    queued_thresh = args.queued_thresh
    queued_thresh_mult = args.queued_thresh_mult
    qps_mult = args.qps_mult
    tokens_per_turn = args.tokens_per_turn
    num_secs = args.num_secs
    small_iters = args.small_iters
    trigger2 = args.trigger2

    # Exit if experiment has been run already.
    output_file_path = f"batch_3011_8gpu/basiclb1_qps{qps_mult}_queue{queued_thresh}_trigger2{trigger2}_queuemult{queued_thresh_mult}_cert{cert_thresh}_i{small_iters}_{model1}_{model2}_2x{tokens_per_turn}_{num_secs}s.json"
    out_times_path = output_file_path.split(".json")[0] + "_times.json"

    if os.path.exists(output_file_path):
        sys.exit(0)
    print("\n", f"Running:", output_file_path, "\n")

    # 2. Read QPS trace.
    qps_trace = get_qps_trace(scaling_factor=qps_mult, num_secs=num_secs)

    # 3. Set up asyncio and threading environment
    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=start_loop, args=(loop,), daemon=True)
    loop_thread.start()

    # 4. Spawn LLMs with correct parameters.
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
                                    max_tokens=tokens_per_turn,
                                    presence_penalty=0,
                                    frequency_penalty=0,
                                    min_tokens=20)

    if model1 == "8b":
        small_frac = 0.25
    elif model1 == "1b":
        small_frac = 0.15
    else:
        small_frac = 0.15

    # small_frac = 0.3

    llm_small_1 = LLMWrapper(model_name=models[model1],
                        devices=[0,1,2,3],
                        semaphores=[],
                        memory_frac=small_frac,
                        loop=loop)

    llm_small_2 = LLMWrapper(model_name=models[model1],
                        devices=[4,5,6,7],
                        semaphores=[],
                        memory_frac=small_frac,
                        loop=loop)

    llm_large_1 = LLMWrapper(model_name=models[model2],
                        devices=[0,1,2,3],
                        semaphores=[],
                        memory_frac=0.95-small_frac,
                        loop=loop)

    llm_large_2 = LLMWrapper(model_name=models[model2],
                        devices=[4,5,6,7],
                        semaphores=[],
                        memory_frac=0.95-small_frac,
                        loop=loop)

    # 5. Run workload.
    try:
        # 5.1. Run workload.
        request_models, token_times = run_mt_bench(llm_small_1=llm_small_1,
                                                   llm_small_2=llm_small_2,
                                                   llm_large_1=llm_large_1,
                                                   llm_large_2=llm_large_2,
                                                    cert_thresh=cert_thresh,
                                                    trigger_2=trigger2,
                                                    sampling_params=sampling_params,
                                                    qps_trace=qps_trace,
                                                    queued_thresh=queued_thresh,
                                                    queued_thresh_mult=queued_thresh_mult,
                                                    small_iters=small_iters,
                                                    )

        # 5.2. Log results.
        with open(out_times_path, 'w') as f:
            json.dump(token_times, f, indent=2)

        with open(output_file_path, 'w') as f:
            json.dump(request_models, f, indent=2)

        # 5.2.1 Aggregated latency metrics
        with open(f"batch_3011_8gpu/time_log.txt", "a") as f:
            f.write(f"----------------- {output_file_path}\n")

        print("request_models len:", len(request_models))
        compute_final_score(model1, model2, request_models)
        compute_ttft(token_times, qps_trace, False)
        compute_e2e(token_times, qps_trace, False)

        # 5.3. Shut down and free all memory.
        llm_small_1.kill()
        llm_small_2.kill()
        llm_large_1.kill()
        llm_large_2.kill()

    except Exception:
        print(traceback.print_exc())

    finally:
        print("shutdown disabled for batch runner")
        # shutdown_vllm()
