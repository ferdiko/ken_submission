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


def run_mt_bench(llm, 
                 sampling_params, 
                 qps_trace,
                 tokens_per_turn=1600):

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
    large_req_queued = 0 # Deprecated. Let's try with queued tokens.
    large_tokens_queued = 0
    small_tokens_queued = 0
    futures_small = []
    futures_large = []
    evaluated_requests = set()

    # 5. Log metrics
    token_times = defaultdict(list) # To compute latencies
    request_models = {} # req_id -> model used

    # 4. Run one second after the other
    # print("Start running")
    debug_idx = 0
    for qps in qps_trace:
        debug_idx += 1

        print("iter", debug_idx, "QPS:", qps)

        # 4.2. Determine which requests to add.
        sampling_params.max_tokens += tokens_per_turn
        added_req_ids = [r for r in range(processed_until, processed_until + qps)]
        added_prompts = [prompts[r] for r in added_req_ids]

        # Add tokens.
        futures_large.append(asyncio.run_coroutine_threadsafe(
            llm.add_new_requests((added_prompts, sampling_params, added_req_ids)), loop
        ))

        processed_until += qps

        # 4.3. Step for 1 second.
        start_second = time.time()
        while time.time() - start_second < 1:


            # 4.3.2. Do step with large LLM.
            large_time = time.time()
            for f in futures_large:
                f.result()
            futures_large = []

            iter_tokens = llm.step_sync()
            # large_tokens_queued = max(0, large_tokens_queued - 512) # NOTE: Assume max batch = 512

            log_times(iter_tokens, token_times)

            # stopped_req_ids = [req_id for req_id in iter_tokens.keys() if len(token_times[req_id]) >= tokens_per_turn]
            # futures_large.append(asyncio.run_coroutine_threadsafe(
            #     llm.free_requests(stopped_req_ids), loop))

    return request_models, token_times


if __name__ == "__main__":
    # 1. Get experiment hyper params.
    parser = argparse.ArgumentParser(description="Run simple cascade model with specified parameters.")
    parser.add_argument("--model1", type=str, required=False, default="3b", help="The name of the first model (e.g., '3b').")
    parser.add_argument("--qps-mult", type=float, required=False, default=0.275, help="The certification threshold (e.g., 3).")
    parser.add_argument("--tokens-per-turn", type=int, required=False, default=1600, help="The certification threshold (e.g., 3).")
    parser.add_argument("--num-secs", type=int, required=False, default=47, help="The certification threshold (e.g., 3).")
    parser.add_argument("--second", action="store_true", help="Set this flag to indicate the second command.")

    args = parser.parse_args()

    model1 = args.model1
    qps_mult = args.qps_mult
    tokens_per_turn = args.tokens_per_turn
    num_secs = args.num_secs
    second = args.second

    # Exit if experiment has been run already.
    output_file_path = f"batch_3011/vllm_qps{qps_mult}_{model1}_2x{tokens_per_turn}_{num_secs}s.json"
    out_times_path = output_file_path.split(".json")[0] + "_times.json"

    # print("\n", f"Scheduled ({second}):", output_file_path, "\n")
    # if os.path.exists(output_file_path):
    #     sys.exit(42)
    print("\n", f"Running ({second}):", output_file_path, "\n")

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


    if second:
        devices = [4,5,6,7]
    else:
        devices = [0,1,2,3]
    llm_1 = LLMWrapper(model_name=models[model1],
                        devices=devices,
                        semaphores=[],
                        memory_frac=0.9,
                        loop=loop)

    # 5. Run workload.
    try:
        # 5.1. Run workload.
        request_models, token_times = run_mt_bench(llm=llm_1,
                                                    sampling_params=sampling_params,
                                                    qps_trace=qps_trace,
                                                    # tokens_per_turn=tokens_per_turn,
                                                    )

        # 5.2. Log results.
        with open(out_times_path, 'w') as f:
            json.dump(token_times, f, indent=2)

        with open(output_file_path, 'w') as f:
            json.dump(request_models, f, indent=2)

        # 5.2.1 Aggregated latency metrics
        with open(f"batch_3011/time_log_{second}.txt", "a") as f:
            f.write(f"----------------- {output_file_path}\n")
        compute_ttft(token_times, qps_trace, second)
        compute_e2e(token_times, qps_trace, second)

        # 5.3. Shut down and free all memory.
        llm_1.kill()

    except Exception:
        print(traceback.print_exc())

    finally:
        # print("shutdown disabled for batch runner")
        shutdown_vllm()
