import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
import asyncio
import os
from vllm import LLM
import ray
import json
import numpy as np
from openai import OpenAI
import re
import time

import matplotlib.pyplot as plt

import torch
import gc
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import contextlib

# ===================================================================================
# Wrapper to run vLLM in separate processes.
# ===================================================================================

def llm_process(model_name, cuda_devices, memory_frac, semaphores, receive_q, send_q):
    # Instantiate LLM.
    cuda_devices_str = ",".join(map(str, cuda_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str

    # print_ctr = 0

    # if "70B" in model_name:
    #     print("LOADOAODAODOAOD 70 B")
    #     speculative_model="ibm-fms/llama3-70b-accelerator"
    # elif "8B" in model_name:
    #     print("LOADOAODAODOAOD 8 B")
    #     speculative_model="ibm-fms/llama3-8b-accelerator"
    # else:
    #     speculative_model=None

    max_num_batched_tokens = 1024
    llm = LLM(model=model_name, 
    max_model_len=3200,
    # speculative_model="ibm-fms/llama3-70b-accelerator",
        # speculative_model=speculative_model,
    # speculative_model="ibm-fms/llama3-8b-accelerator",
    enforce_eager=True,
    #   max_num_batched_tokens=max_num_batched_tokens,
    max_num_seqs=512,
    #   enable_chunked_prefill=True,
    tensor_parallel_size=len(cuda_devices),
    gpu_memory_utilization=memory_frac,
    
    trust_remote_code=True
    # quantization="fp8"
    )
    
    # Indicate that process ready.
    send_q.put("ok")

    # Check for requests.
    while True:
        # 1. Wait for input
        intent = receive_q.get()
        # print(" "*10, model_name, len(llm.llm_engine.scheduler[0].waiting) + len(llm.llm_engine.scheduler[0].running) + len(llm.llm_engine.scheduler[0].paused))

        if len(intent) == 2 and intent[0] == "pause & step":
            """Make one inference step. 
            - intent[1] ([int]): List of all request ids that should be paused.
            - output (({int -> [int]}, {int -> float})): Token sequeneces of currently 
                running requests, certainties of new predictions. keys are request ids
            """
            # Acquire semaphores.
            for s in semaphores:
                s.acquire()

            # Do step.
            out = llm.pause_and_step(intent[1])

            # Release semaphores and send results back
            for s in semaphores:
                s.release()

            send_q.put(out)

        elif len(intent) == 2 and intent[0] == "add tokens":
            """Add tokens to already present requests. 
            - intent[1] ({int -> [int]}): request id -> tokens.
            - No output
            """
            llm.add_tokens_to_request(*intent[1])
            send_q.put("ok")

        elif len(intent) == 4 and intent[0] == "add requests":
            """Add tokens to already present requests. 
            - intent[1] ([string], Optional[sampling_param], Optional[request_ids]): Prompts.
            - No output
            """
            llm.add_new_requests(prompts=intent[1],
                                 sampling_params=intent[2],
                                 request_ids=intent[3])
            send_q.put("ok")

        elif len(intent) == 2 and intent[0] == "resume":
            """Resume paused requests. 
            - intent[1] ([int]): Request ids to resume.
            - No output
            """
            llm.resume_requests(intent[1])
            send_q.put("ok")

        elif len(intent) == 2 and intent[0] == "free":
            req_ids = llm.free_requests(intent[1])
            send_q.put("ok")

        elif len(intent) == 2 and intent[0] == "pause":
            out = llm.pause_requests(intent[1])
            send_q.put("ok")

        elif intent == "finished req":
            req_ids = llm.get_finished_request_ids()
            send_q.put(req_ids)

        elif intent == "tokens":
            queued_tokens = llm.num_queued_tokens()
            send_q.put(queued_tokens)

        elif intent == "step":
            """Make one inference step. 
            - intent[1] ([int]): List of all request ids that should be paused.
            - output (({int -> [int]}, {int -> float})): Token sequeneces of currently 
                running requests, certainties of new predictions. keys are request ids
            """

            # print("step", model_name.split("/")[-1], end=" ")
            # llm.llm_engine.scheduler[0].print_req_lens()

            # Acquire semaphores.
            for s in semaphores:
                s.acquire()

            # Do step.
            # start = time.time()

            # if print_ctr % 40 == 0:
            # print("    ", model_name.split("/")[-1], end=" ")
            # print_ctr += 1

            out = llm.step()
            # print(model_name.split("/")[-1], time.time() - start)

            # Release semaphores and send results back
            for s in semaphores:
                s.release()

            send_q.put(out)

        elif intent == "kill":
            del llm.llm_engine.model_executor
            del llm
            destroy_model_parallel()
            destroy_distributed_environment()
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()

            send_q.put("ok")

        else:
            raise ValueError("Unknown intent passed to LLM process")


class LLMWrapper:

    def __init__(self, model_name, devices, memory_frac, semaphores, loop):  
        self.loop = loop
        
        # Initialize receive and send queues.
        self.receive_q = multiprocessing.Queue()
        self.send_q = multiprocessing.Queue()

        # Spawn process.
        self.proc = multiprocessing.Process(target=llm_process, 
                                            args=(model_name, 
                                                  devices, 
                                                  memory_frac,
                                                  semaphores,
                                                  self.send_q, 
                                                  self.receive_q))
        self.proc.start()

        # Wait for response indicating that model has been loaded.
        ok = self.receive_q.get()
        assert ok == "ok"


    async def _receive(self):
        """Block and wait to receive message."""
        loop = self.loop
        return await loop.run_in_executor(None, lambda: self.receive_q.get())


    async def _send(self, msg):
        """Send message."""
        loop = self.loop
        await loop.run_in_executor(None, lambda: self.send_q.put(msg))


    async def pause_and_step(self, paused_requests):
        """Deprecated"""
        await self._send(("pause & step", paused_requests))
        return await self._receive()


    async def pause(self, paused_requests):
        await self._send(("pause", paused_requests))
        return await self._receive()


    async def step(self):
        await self._send("step")
        return await self._receive()


    async def num_queued_tokens(self):
        await self._send("tokens")
        return await self._receive()


    async def free_requests(self, request_ids):
        await self._send(("free", request_ids))
        await self._receive()


    async def add_new_requests(self, args):
        await self._send(("add requests",) + args)
        return await self._receive()


    async def add_tokens_to_request(self, args):
        await self._send(("add tokens", args))
        return await self._receive()


    async def resume_requests(self, request_ids):
        await self._send(("resume", request_ids))
        return await self._receive()


    async def get_finished_request_ids(self):
        await self._send(("finished req"))
        return await self._receive()


    def resume_requests_sync(self, request_ids):
        asyncio.run_coroutine_threadsafe(self.resume_requests(request_ids), self.loop).result()


    def pause_and_step_sync(self, paused_requests):
        """Deprecated"""
        return asyncio.run_coroutine_threadsafe(self.pause_and_step(paused_requests), self.loop).result()


    def add_tokens_to_request_sync(self, args):
        asyncio.run_coroutine_threadsafe(self.add_tokens_to_request(args), self.loop).result()


    def add_new_requests_sync(self, args):
        asyncio.run_coroutine_threadsafe(self.add_new_requests(args), self.loop).result()


    def get_finished_request_ids_sync(self):
        return asyncio.run_coroutine_threadsafe(self.get_finished_request_ids(), self.loop).result()


    def free_requests_sync(self, request_ids):
        asyncio.run_coroutine_threadsafe(self.free_requests(request_ids), self.loop).result()


    def pause_sync(self, paused_requests):
        return asyncio.run_coroutine_threadsafe(self.pause(paused_requests), self.loop).result()


    def num_queued_tokens_sync(self):
        return asyncio.run_coroutine_threadsafe(self.num_queued_tokens(), self.loop).result()


    def step_sync(self):
        return asyncio.run_coroutine_threadsafe(self.step(), self.loop).result()


    def kill(self):
        asyncio.run_coroutine_threadsafe(self._send("kill"), self.loop).result()
        asyncio.run_coroutine_threadsafe(self._receive(), self.loop).result()
        self.proc.kill()

# ===================================================================================
# Get MT-Bench prompts.
# ===================================================================================

def get_mt_bench_prompts(vicuna=False):
    if vicuna:
        question_file = "vicuna_questions.jsonl"
    else:
        question_file = "mt_questions.jsonl"

    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, question_file), "r") as f:
        lines = f.readlines()
    
    tasks = []
    for line in lines:
        data = json.loads(line)
        tasks.append(data["turns"])

    return tasks


def get_second_turn_prompts():
    # HACK: Just use 70b generations for run_no turn.
    dir_path = "mt_bench_single_model_samples"
    with open(f"{dir_path}/2x1600_70b.txt", "r") as f:
        run_no_turn_prompts = f.readlines()[:80]
    return run_no_turn_prompts


def get_first_turn_prompts():

    system_prompt = "Answer user questions based on the previous chat history. \
In the chat history, the user's requests are preceeded by 'Task:' and the responses \
are preceeded by 'Response:'. Just fill in the last response field. Be concise.\n\n"

    tasks = get_mt_bench_prompts(vicuna=False)
    first_turn_prompts = []

    for t in tasks:
        first_turn_prompts.append(system_prompt + f"Task: {t[0]}\nResponse: ")

    return first_turn_prompts


def _replace_value(val):
    # QPS 0.5 --> 1, 0
    if val > 1:
        return [val]
    else: 
        n = int(1 / val)
        return [1] + [0] * (n - 1)


def get_qps_trace(scaling_factor, num_secs, qps_file="twitter_qps.txt"):
    qps_trace = np.loadtxt(qps_file)*scaling_factor
    qps_trace = qps_trace[350:490]
    # print(qps_trace)
    qps_trace = np.concatenate([_replace_value(x) for x in qps_trace])
    # qps_trace = qps_trace[:num_secs]

    # Make sure all requests finish.
    qps_trace = np.append(qps_trace, np.zeros(30))
    return np.array(qps_trace, dtype=int)


# ===================================================================================
# Rate responses with GPT-4o
# ===================================================================================

def rate_responses(responses):

    # HACK: Remove prompt to Llama
    responses = [r[221:] for r in responses]


    # Instantiate OpenAI responses
    with open('openai_siva.key', 'r') as key_file:
        api_key = key_file.read().strip()

    oai_client = OpenAI(
        api_key=api_key,
    )

    # System prompt.
#     prompt = "You need to judge responses of an LLM that is run on a multi-turn conversation benchmark. \
# The user prompt is indicated with 'Prompt:' and the LLM's response it indicated with 'Response:'. \
# Only judge the LLM's responses. Some responses are cut off because there's a maximum token limit --- don't deduce points for cut-off responses. \
# Reason about them and, in the end, output a score from 1 to 10 in the following format: \
# 'Final score: X'. The responses you should judge are given here:\n\n"

    prompt =  "Please act as an impartial judge and evaluate the quality of the response provided by \
an AI assistant to the user question displayed below. Only rate the last response. Your evaluation should consider factors such \
as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the last response only. \
Every answer has been limited in its length and you should not deduce points, if the answer is cut off abruptly due to its length limit.  \
Begin your evaluation by providing a short explanation. Be as objective as possible. After providing \
your explanation, you must rate the last response on a scale of 1 to 10 by strictly following this format: 'Final score: X'"

    prompt =  "Please act as an impartial judge and evaluate the quality of the responses provided by \
an AI assistant to the tasks displayed below. Your evaluation should consider factors such \
as the helpfulness, relevance, accuracy, depth, creativity, and level of detail. \
Begin your evaluation by providing a short explanation. Be as objective as possible. After providing \
your explanation, you must rate the responses on a scale of 1 to 10 by strictly following this format: 'Final score: X'"


    # Asynchronously process all requests.
    async def _process_all():
        tasks = [
            _call_openai(oai_client, (prompt + r)[:1600])
            for r in responses
        ]
        return await asyncio.gather(*tasks)

    # Run the async tasks and collect results
    all_judgments = asyncio.run(_process_all())

    all_scores = []
    for judge_response, r in zip(all_judgments, responses):
        # print("\n\n", "="*20)
        # print(prompt + r)
        # print("\n", "- -" * 10)
        # print(judge_response)

        # Parse out score.
        match = re.search(r"Final score: (\d+(\.\d+)?)", judge_response)
        if match:
            score_str = match.group(1)
            # Convert to int if no decimal point, otherwise to float
            score = float(score_str)
            all_scores.append(score)
            # print()
            # print("      Final score:", score)
        else:
            print(prompt+r)
            print("="*20)
            print(judge_response)
            # assert False

            all_scores.append(-1)
            print("No final score found.")

    return all_scores


def rate_responses_separately(responses):
    """Deprecated since now the output is seperate"""
    # Seperate out responses.
    split_responses = []

    for r in responses:
        split_responses += _divide_response(r)

    # Define prompt.
    prompt =  "Please act as an impartial judge and evaluate the quality of the response provided by \
an AI assistant to the user question displayed below. Only rate the last response. Your evaluation should consider factors such \
as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the last response only. \
Begin your evaluation by providing a short explanation. Be as objective as possible. After providing \
your explanation, you must rate the last response on a scale of 1 to 10 by strictly following this format: 'Final score: X'"

    # Rate.
    return rate_responses(split_responses, prompt)


# Async OpenAI call function remains the same
async def _call_openai(oai_client, prompt, model="gpt-4o"):
    # time.sleep(1) # For rate limit.
    # print("send")

    response = await asyncio.to_thread(
        oai_client.chat.completions.create,
        model=model,
        temperature=0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    
    response = json.loads(response.model_dump_json()) 
    return response['choices'][0]['message']['content'].strip()


# ===================================================================================
# Utils for writing / reading responses to / from file.
# ===================================================================================

def write_responses_to_file(responses, file_name):
    with open(file_name, "w") as file:
        for full_line in responses:
            # prompt = output.prompt
            # generated_text = output.outputs[0].text
            # full_line = prompt + generated_text
            escaped_string = full_line.replace("\n", "\\n")
            file.write(escaped_string + "\n")


def read_responses_from_file(file_name):
    with open(file_path, "r") as file:
        responses = [line.strip().replace("\\n", "\n") for line in file if len(line) > 5]
    return responses


# ===================================================================================
# Utils for computing latencies from dict.
# ===================================================================================

def compute_ttft(token_times, qps_trace, run_no):
    ttfts = {}

    for r_id, t in token_times.items():
        if len(t) >= 2:
            ttfts[r_id] = (t[1] - t[0])*1000
        else:
            print("Req not finished")

    ttfts = [ttfts[k] for k in sorted(ttfts.keys())]

    # For plotting.
    ttft_per_s = []
    start_index = 0
    for count in qps_trace:
        count = int(count)
        group = ttfts[start_index:start_index + count]
        ttft_per_s.append(np.mean(group))
        start_index += count
    plot(qps_trace, ttft_per_s, run_no)

    p95 = np.percentile(ttfts, 95)
    p50 = np.percentile(ttfts, 50)
    avg = np.mean(ttfts)
    print("TTFT 95th Percentile:", p95)
    print("TTFT 50th Percentile:", p50)
    print("TTFT Average:", avg)

    with open(f"batch_0312_2gpu/time_log_{run_no}.txt", "a") as f:
        f.write(f"TTFT 95th Percentile: {p95}\n")
        f.write(f"TTFT 50th Percentile: {p50}\n")
        f.write(f"TTFT Average: {avg}\n")

    return ttfts


def compute_e2e(token_times, qps_trace, run_no):
    e2e = {}

    for r_id, t in token_times.items():
        e2e[r_id] = (t[-1] - t[0])*1000
        if e2e[r_id] == 0:
            print("E2E is 0")
    
    e2e = [e2e[k] for k in sorted(e2e.keys())]

    # For plotting.
    # e2e_per_s = []
    # start_index = 0
    # for count in qps_trace:
    #     count = int(count)
    #     group = e2e[start_index:start_index + count]
    #     e2e_per_s.append(np.mean(group))
    #     start_index += count

    p95 = np.percentile(e2e, 95)
    p50 = np.percentile(e2e, 50)
    avg = np.mean(e2e)
    print("E2E 95th Percentile:", p95)
    print("E2E 50th Percentile:", p50)
    print("E2E Average:", avg)

    with open(f"batch_0312_2gpu/time_log_{run_no}.txt", "a") as f:
        f.write(f"E2E 95th Percentile: {p95}\n")
        f.write(f"E2E 50th Percentile: {p50}\n")
        f.write(f"E2E Average: {avg}\n")

    return e2e


def plot(qps_trace, y_data, run_no):
    # Create the plot
    fig, ax1 = plt.subplots()

    # Plot the first series on the primary y-axis
    ax1.plot(qps_trace, label='QPS Trace', color='blue')
    ax1.set_ylabel('QPS Trace', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a run_noary y-axis
    ax2 = ax1.twinx()
    ax2.plot(y_data, label='Latency', color='red', alpha=0.5)
    ax2.set_ylabel('Latency', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Optional: Add a title and show the plot
    plt.title("QPS Trace and TTFT")
    plt.savefig(f"latency_{run_no}.png")



########################### Compute score
def read_scores(filename):
    """
    Reads the scores from the given filename and returns a dictionary mapping filenames to score lists.
    """
    scores = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if 'mt_bench_single_model_samples' in line:
                parts = line.split(':', 1)
                filename_key = parts[0].split('/')[-1]
                rest = parts[1].strip()
                list_str_start = rest.find('[')
                if list_str_start != -1:
                    list_str = rest[list_str_start:]
                    try:
                        score_list = literal_eval(list_str)
                        scores[filename_key] = score_list
                    except:
                        pass
    return scores

def compute_final_score(model1, model2, request_models, scores_filename='pred_scores/scores.txt'):
    """
    Compute the final score given model1, model2, and request_models.

    Parameters:
    - model1: str, "1b", "3b", "8b", or "70b", denotes the "small" model
    - model2: str, "1b", "3b", "8b", or "70b", denotes the "large" model
    - request_models: dict, mapping from key to size ("small" or "large")
    - scores_filename: str, optional, the filename where the scores are stored

    Returns:
    - final_score: float, the average score computed from the scores_list, or None if no scores were computed
    """
    if not os.path.exists(scores_filename):
        print(f"File {scores_filename} does not exist.")
        return None
    scores_dict = read_scores(scores_filename)

    scores_list = []
    for key, size in request_models.items():
        key_mod = key % 160  # Adjust as necessary
        if size == 'small':
            model_filename = f'2x1600_{model1}.txt'
            scores_small = scores_dict.get(model_filename, [])
            if scores_small:
                if key_mod < len(scores_small):
                    scores_list.append(scores_small[key_mod])
                else:
                    print(f"Warning: Key {key} is out of range for scores_small.")
            else:
                print(f"Warning: Scores for model {model1} not found.")
        elif size == 'large':
            model_filename = f'2x1600_{model2}.txt'
            scores_large = scores_dict.get(model_filename, [])
            if scores_large:
                if key_mod < len(scores_large):
                    scores_list.append(scores_large[key_mod])
                else:
                    print(f"Warning: Key {key} is out of range for scores_large.")
            else:
                print(f"Warning: Scores for model {model2} not found.")
        else:
            print(f"Warning: Unknown size '{size}' for key {key}.")
    if scores_list:
        print("scores list:", len(scores_list), len(request_models))
        final_score = sum(scores_list) / len(scores_list)
    else:
        final_score = None

    with open(f"batch_0312_2gpu/time_log_{run_no}.txt", "a") as f:
        f.write(f"Score: {final_score}\n")

    return final_score




if __name__ == "__main__":

    print(get_qps_trace(1, 1))

    # files = [
    #     "vicuna_runs/1b_1.5_70b_diff.txt", # 9.11 [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 10.0, 10.0, 10.0, 9.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 10.0, 9.0, 10.0, 7.0, 9.0, 8.0, 8.0, 9.0, 1.0, 9.0, 8.0, 9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 7.0, 7.0, 7.0, 8.0, 10.0, 7.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 7.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 6.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.0, 9.0, 10.0, 10.0, 9.0, 10.0, 9.0, 9.0, 10.0, 9.0, 10.0]
    #     "vicuna_runs/llama_70b.txt", # 9.15 [9.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 9.0, 8.0, 10.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 10.0, 8.0, 9.0, 9.0, 8.0, 9.0, 1.0, 9.0, 6.0, 7.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    #     "vicuna_runs/llama_1b.txt", # 7.9 [10.0, 10.0, 7.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0, 9.0, 10.0, 9.0, 10.0, 9.0, 9.0, 9.0, 9.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 4.0, 8.0, 9.0, 9.0, 6.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0, 9.0, 5.0, 10.0, 10.0, 6.0, 3.0, 2.0, 10.0, 3.0, 9.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 10.0]
    #     "vicuna_runs/1b_2_70b_diff.txt", # 9.625 --- too few entries [9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 9.0, 9.0, 10.0, 9.0, 10.0]

    #     "mt_bench_runs/1b_2_70b_diff.txt", # no prompt ---- 7.475 [9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 2.0, 9.0, 10.0, 3.0, 8.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 5.0, 10.0, 8.0, 1.0, 9.0, 5.0, 10.0, 6.0, 10.0, 9.0, 10.0, 10.0, 2.0, 2.0, 2.0, 1.0, 1.0, 10.0, 8.0, 6.0, 7.0, 7.0, 10.0, 10.0, 3.0, 8.0, 8.0, 10.0, 5.0, 7.0, 4.0, 10.0, 4.0, 8.0, 10.0, 7.0, 10.0, 10.0, 6.0, 4.0, 8.0, 4.0, 9.0, 2.0, 3.0, 7.0, 9.0, 9.0, 8.0, 9.0, 10.0, 9.0, 9.0, 9.0, 4.0, 9.0, 9.0, 9.0, 9.0, 10.0]

    #     "mt_bench_runs/prompted_1b_2_70b_diff.txt", # 8.0125 [9.0, 9.0, 9.0, 9.0, 8.0, 9.0, 9.0, 9.0, 9.0, 5.0, 9.0, 9.0, 9.0, 9.0, 7.0, 9.0, 7.0, 9.0, 8.0, 10.0, 10.0, 9.0, 10.0, 8.0, 2.0, 7.0, 5.0, 10.0, 10.0, 9.0, 3.0, 3.0, 4.0, 8.0, 5.0, 1.0, 8.0, 9.0, 10.0, 9.0, 9.0, 9.0, 10.0, 8.0, 7.0, 10.0, 7.0, 5.0, 3.0, 8.0, 9.0, 10.0, 3.0, 3.0, 7.0, 9.0, 10.0, 6.0, 10.0, 9.0, 8.0, 4.0, 8.0, 9.0, 8.0, 9.0, 9.0, 10.0, 10.0, 7.0, 10.0, 10.0, 9.0, 10.0, 10.0, 9.0, 10.0, 9.0, 10.0, 9.0]
    #     "mt_bench_runs/prompted_1b_3_70b_diff.txt", # 8.475 [10.0, 9.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 2.0, 9.0, 9.0, 9.0, 7.0, 9.0, 10.0, 9.0, 10.0, 9.0, 9.0, 8.0, 10.0, 9.0, 8.0, 3.0, 10.0, 7.0, 10.0, 9.0, 10.0, 3.0, 3.0, 9.0, 9.0, 7.0, 1.0, 9.0, 9.0, 10.0, 9.0, 8.0, 10.0, 10.0, 9.0, 7.0, 10.0, 9.0, 7.0, 4.0, 10.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 5.0, 9.0, 9.0, 7.0, 9.0, 9.0, 8.0, 9.0, 9.0, 10.0, 9.0, 4.0, 8.0, 10.0, 9.0, 9.0, 10.0, 9.0, 9.0, 10.0, 10.0, 10.0]
    #     "mt_bench_runs/prompted_1b_4_70b_diff.txt", # 8.265822784810126 [9.0, 9.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 3.0, 9.0, 9.0, 9.0, 6.0, 9.0, 10.0, 8.0, 10.0, 9.0, 10.0, 9.0, 10.0, 9.0, 9.0, 2.0, 5.0, 10.0, 9.0, 9.0, 3.0, 5.0, 9.0, 8.0, 8.0, 1.0, 9.0, 3.0, 10.0, 10.0, 8.0, 10.0, 9.0, 9.0, 6.0, 10.0, 8.0, 6.0, 4.0, 10.0, 7.0, 10.0, 10.0, 9.0, 9.0, 10.0, 9.0, 10.0, 7.0, 9.0, 9.0, 6.0, 9.0, 8.0, 7.0, 9.0, 9.0, 10.0, 9.0, 3.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 10.0, 9.0, 10.0, 10.0]
    #     "mt_bench_runs/prompted_3b_3_70b_diff.txt", # 8.4 [9.0, 10.0, 8.0, 10.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 9.0, 10.0, 10.0, 10.0, 7.0, 10.0, 8.0, 8.0, 10.0, 6.0, 10.0, 10.0, 10.0, 10.0, 1.0, 3.0, 8.0, 3.0, 7.0, 1.0, 9.0, 1.0, 10.0, 9.0, 10.0, 10.0, 10.0, 10.0, 6.0, 9.0, 10.0, 2.0, 7.0, 9.0, 7.0, 9.0, 10.0, 8.0, 8.0, 8.0, 9.0, 9.0, 4.0, 7.0, 9.0, 5.0, 10.0, 9.0, 10.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 8.0, 8.0]

    #     "mt_bench_runs/prompted_8b_1_70b_diff.txt",
    #     "mt_bench_runs/prompted_8b_2_70b_diff.txt", # 8.6125 [9.0, 9.0, 9.0, 9.0, 8.0, 9.0, 8.0, 9.0, 10.0, 10.0, 9.0, 9.0, 9.0, 7.0, 10.0, 9.0, 10.0, 10.0, 10.0, 9.0, 8.0, 9.0, 9.0, 9.0, 9.0, 4.0, 10.0, 9.0, 10.0, 10.0, 2.0, 8.0, 10.0, 5.0, 8.0, 1.0, 10.0, 1.0, 10.0, 3.0, 9.0, 10.0, 10.0, 9.0, 4.0, 7.0, 9.0, 8.0, 7.0, 10.0, 10.0, 9.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 10.0, 9.0, 9.0, 7.0, 9.0, 9.0, 9.0, 6.0, 9.0, 10.0, 9.0, 8.0, 10.0, 10.0, 10.0, 9.0, 10.0, 9.0, 10.0, 9.0, 10.0, 10.0]
    #     "mt_bench_runs/prompted_8b_3_70b_diff.txt", # 8.4875 [9.0, 9.0, 9.0, 9.0, 8.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 5.0, 10.0, 9.0, 10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 4.0, 4.0, 7.0, 9.0, 10.0, 10.0, 1.0, 5.0, 6.0, 3.0, 6.0, 1.0, 7.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 7.0, 10.0, 8.0, 2.0, 9.0, 10.0, 8.0, 10.0, 10.0, 5.0, 10.0, 10.0, 9.0, 10.0, 10.0, 7.0, 9.0, 9.0, 9.0, 10.0, 10.0, 9.0, 9.0, 10.0, 10.0, 6.0, 10.0, 10.0, 9.0, 9.0, 10.0, 9.0, 10.0, 9.0, 9.0, 9.0]

    #     "mt_bench_runs/llama_1b_diff.txt", # 7.5125 [9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 2.0, 9.0, 10.0, 3.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 10.0, 5.0, 10.0, 7.0, 1.0, 10.0, 4.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 2.0, 2.0, 2.0, 1.0, 1.0, 10.0, 4.0, 3.0, 7.0, 7.0, 10.0, 10.0, 3.0, 7.0, 9.0, 10.0, 5.0, 7.0, 4.0, 10.0, 3.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 4.0, 9.0, 4.0, 9.0, 2.0, 3.0, 8.0, 9.0, 8.0, 8.0, 9.0, 10.0, 9.0, 9.0, 10.0, 4.0, 9.0, 9.0, 9.0, 10.0, 10.0]
    #     "mt_bench_runs/llama_70b_diff.txt", # 8.7375 [9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 9.0, 10.0, 8.0, 10.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 10.0, 10.0, 9.0, 10.0, 10.0, 6.0, 8.0, 4.0, 9.0, 10.0, 10.0, 10.0, 10.0, 8.0, 4.0, 10.0, 2.0, 8.0, 1.0, 10.0, 2.0, 10.0, 10.0, 9.0, 10.0, 10.0, 9.0, 8.0, 10.0, 9.0, 3.0, 9.0, 9.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 7.0, 10.0, 7.0, 9.0, 9.0, 10.0, 9.0, 10.0, 10.0, 9.0, 5.0, 9.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 10.0, 10.0]
    #     ]
    # file_path = "mt_bench_runs/prompted_8b_1_70b_diff.txt"

    # file_paths = [
    #     "prompted_1b_1_70b_diff.txt",
    #     "prompted_1b_1.5_70b_diff.txt",
    #     "prompted_1b_2.5_70b_diff.txt",
    #     "prompted_1b_3.5_70b_diff.txt",
    #     "prompted_3b_1_70b_diff.txt",
    #     "prompted_3b_1.5_70b_diff.txt",
    #     "prompted_3b_2_70b_diff.txt",
    #     "prompted_3b_2.5_70b_diff.txt",
    #     "prompted_3b_3.5_70b_diff.txt",
    #     "prompted_8b_2.5_70b_diff.txt",
    #     "prompted_8b_3.5_70b_diff.txt",
    # ]

    # file_paths = ["mt_bench_runs_async/prompted_8b_1.0_70b_diff.txt",
    #                 "mt_bench_runs_async/prompted_8b_1.5_70b_diff.txt",
    #                 "mt_bench_runs_async/prompted_8b_2.0_70b_diff.txt",
    #                 "mt_bench_runs_async/prompted_8b_2.5_70b_diff.txt",
    #                 "mt_bench_runs_async/prompted_8b_3.0_70b_diff.txt",
    #                 "mt_bench_runs_async/prompted_8b_3.5_70b_diff.txt"]

    # file_paths = ["mt_bench_runs_concise/casc_8b_2_70b_2x2k.txt"]

    # # New, with samples.

    # # for file_path in file_paths:
    # for file_path in os.listdir("mt_bench_single_model_samples"):
    #     file_path = "mt_bench_single_model_samples/" + file_path

    #     if "1b" in file_path:
    #         continue
    #     if "8b" in file_path:
    #         continue
    #     if".json" in file_path:
    #         continue

    #     # file_path = "mt_bench_runs/" + model

    #     responses = read_responses_from_file(file_path)

    #     scores = rate_responses(responses)

    #     with open("mt_bench_single_model_samples.txt", "a+") as f:
    #         f.write(f'{file_path}_cut1600: {float(np.mean(scores))}, {scores} \n')
    #     print("MEAN:", np.mean(scores))
    #     time.sleep(5)

