import asyncio
import os.path

import numpy as np
import ray
import torch
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from online.model_queue import BertQueue, BertQueue2
from utils.logger_setup import setup_logger

import time

# ======================================================================================================================
# Single GPU server
# ======================================================================================================================
@ray.remote(num_gpus=1)
class GpuServer:

    def __init__(self, server_id, gear_plan, models, num_queues=3, qps_fac=1, max_bs=1000, max_queue_len=8):
        """
        GpuServer manages queues and models
        :param server_id:
        :param gear_plan:
        :param models:
        :param num_queues:
        """
        # Needed for ExLlama: Initialize torch cuda state.
        torch.cuda._lazy_init()

        self.logger = setup_logger()
        self.server_id = 0 # TODO server_id
        self.gear_plan = gear_plan

        # Load models.
        self.models = models
        for m in self.models:
            self.logger.debug(f"Server {server_id} loading model {m}")
            self.models[m].load()
            self.logger.debug(f"Server {server_id} loaded model {m}")

        # Initialize model queues.
        self.queues = []
        for _ in range(num_queues):
            if max_bs == 0:
                self.queues.append(BertQueue())
            else:
                self.queues.append(BertQueue2(max_bs=max_bs))

        # Gear attributes.
        self.cur_qps = 0
        self.cur_ensemble = gear_plan.gears[0].models
        self.cur_threshs = gear_plan.gears[0].threshs
        self.cur_batch_sizes = [1] * num_queues # TODO: gear_plan.gears[0].batch_sizes

        # Grid search.
        self.qps_mul = qps_fac
        self.max_queue_len = max_queue_len


    def alive(self):
        """
        Check if actor alive.
        :return:
        """
        return 1


    def warmup(self, samples, num_runs=2):
        """
        Run samples through all models
        :param samples:
        :return:
        """
        samples = (samples[0].to("cuda"), samples[1].to("cuda"))

        for _ in range(num_runs):
            for model_name in self.models:
                model = self.models[model_name]
                model.forward(samples, [0])

        return 0


    def switch_gear(self, qps):
        self.logger.debug(f"Switch gear {self.cur_qps} -> {qps}: Q: {self.queues[0].size()}")

        qps *= self.qps_mul

        qps = int(min(len(self.gear_plan.gears)-1, qps))

        # 1. Make sure there are only few samples in queue before downgrading.
        if self.cur_qps > qps and self.max_queue_len*self.queues[0].size() > qps:
            return

        # 2. Switch to new cascade.
        self.cur_qps = qps
        self.cur_ensemble = self.gear_plan.gears[qps].models
        self.cur_threshs = self.gear_plan.gears[qps].threshs
        #self.cur_batch_sizes = self.gear_plan.gears[qps].server_batch_sizes[self.server_id] # TODO


    def add_to_queue(self, sample_ids, samples, queue_idx):
        # assert type(samples) == list
        start = time.time()
        self.logger.debug(f"add to q {queue_idx}")
        self.queues[queue_idx].add(sample_ids, samples)
        self.logger.debug(f"added to q {queue_idx}, {time.time()-start}")



    def set_qps_debug(self, qps):
        self.cur_qps = qps
        self.cur_ensemble = self.gear_plan.gears[qps].models
        self.cur_threshs = self.gear_plan.gears[qps].threshs
        self.cur_batch_sizes = self.gear_plan.gears[qps].server_batch_sizes[self.server_id]


    def check_queue(self, queue_id):
        self.logger.debug(f"poll q {queue_id}")

        if queue_id == 0:
            self.logger.debug(f"Queue sizes: {[q.size() for q in self.queues]}")

        # 1. Check if queue full enough for inference.
        queue = self.queues[queue_id]
        if queue_id >= len(self.cur_ensemble) or queue.size() < self.cur_batch_sizes[queue_id]:
            return None, None

        # 2. Predict and get certainties.
        model_name = self.cur_ensemble[queue_id]
        model = self.models[model_name]
        #self.logger.debug(f"tmp: {queue_id}, {[q.size() for q in self.queues]}, {[len(q.samples) for q in self.queues]}")

        samples, sample_ids = queue.get(-1) #queue.get(self.cur_batch_sizes[queue_id])
        certs, preds = model.forward(samples, sample_ids)
        self.logger.debug(f"Server {self.server_id} {model_name}: "
                          f"models: {self.cur_ensemble}, bs: {self.cur_batch_sizes}")

        # 3. Cascade samples below certainty threshold.
        # NOTE: This back and fourth to CPU here is inefficient and can easily be avoided.
        if queue_id >= len(self.cur_ensemble) - 1:
            return sample_ids, preds

        casc_indexes = torch.where(certs < self.cur_threshs[queue_id+1])[0]
        # NOTE: The indexing operation below is super costly (can be up to 100ms if many samples)
        casc_samples = tuple(sample[casc_indexes] for sample in samples) # TODO: BERT specific.
        assert casc_samples[0].is_cuda and casc_samples[1].is_cuda # TODO: BERT specific.
        casc_sample_ids = [sample_ids[i] for i in casc_indexes]
        done_sample_ids = [id for id in sample_ids if id not in casc_sample_ids]
        done_preds = [p for i, p in enumerate(preds) if i not in casc_indexes]
        assert len(done_preds) == len(done_sample_ids)
        cur = time.time()

        # NOTE: Assume all models on server for now.
        assert self.cur_ensemble[queue_id+1] in self.models
        self.add_to_queue(casc_sample_ids, casc_samples, queue_id+1)

        # self.logger.debug(f"Completed: {done_sample_ids}; Cascaded: {casc_sample_ids}")

        return done_sample_ids, done_preds


    def close(self):
        return 0


# ======================================================================================================================
# Multi GPU server
# ======================================================================================================================

# TODO: Hangs when sending python objects over. For now we just send string and need to change
#  the import to correct dict here.
# from workloads.llama.llama_model import *
from workloads.twitter.twitter_models import *


def model_process(task_queue, response_queue, semaphores):
    # Needed for ExLlama: Initialize torch cuda state.
    torch.cuda._lazy_init()

    # Load model.
    model_name = task_queue.get()
    model = get_model_dict("supercloud")[0][model_name]
    model.load()
    response_queue.put("ok")

    # Check for requests from head process.
    while True:
        # 1. Wait for input
        data = task_queue.get()
        samples, sample_ids, thresh = data

        if samples[0].shape[0] == 0:
            print(samples)
            # TODO: Debug this!! You shouldn't enter here.
            assert False
            response_queue.put((torch.tensor([]), torch.tensor([])))
            continue

        # 2. Do inference.
        # NOTE: With multiple GPUs and a list of semaphores, you'll need to do something like the follwing:
        # with ExitStack() as stack:
        #     # Acquire all semaphores
        #     for semaphore in semaphores:
        #         stack.enter_context(semaphore)
        with semaphores[0]:
            certs, preds = model.forward(samples, sample_ids)

        # 3. Get which samples need to be cascded and which are done.
        # if queue_id >= len(self.cur_ensemble) - 1:
        #     return sample_ids, preds

        casc_indexes = torch.where(certs < thresh)[0]
        # NOTE: The indexing operation below is super costly (can be up to 100ms if many samples)
        casc_samples = tuple(sample[casc_indexes] for sample in samples) # TODO: BERT specific.
        # assert casc_samples[0].is_cuda and casc_samples[1].is_cuda # TODO: BERT specific.
        casc_sample_ids = [sample_ids[i] for i in casc_indexes]
        done_sample_ids = [id for id in sample_ids if id not in casc_sample_ids]
        done_preds = [p for i, p in enumerate(preds) if i not in casc_indexes]
        assert len(done_preds) == len(done_sample_ids)

        # 4. Send results back to main process.
        response_queue.put((casc_samples, casc_sample_ids, done_preds, done_sample_ids))


@ray.remote(num_gpus=1)
class MultiGpuServer:

    def __init__(self, server_id, gear_plan, models, num_queues=3):
        """
        MultiGpuServer contains multiple GPUs. Inference can occur in parallel for models on disjoint sets of GPUs.
        :param server_id:
        :param gear_plan:
        :param models:
        :param num_queues:
        """
        self.logger = setup_logger()
        self.server_id = server_id
        self.gear_plan = gear_plan

        self.loop = asyncio.get_running_loop()

        # Concurreny control: Don't alllow for more than one concurrent inference per GPU.
        self.semaphores = [multiprocessing.Semaphore(1)] # TODO: Implement multiple GPUs per server
        # self.gpu_0_queue = []
        # self.gpu_1_queue = []
        # self.gpu_0 = asyncio.Condition()
        # self.gpu_1 = asyncio.Condition()

        # Spawn model processes and load models.
        self.model_process = {}
        self.procs = []
        for m in models:
            # Create send and receive queue for process.
            self.model_process[m] = (multiprocessing.Queue(), multiprocessing.Queue()) # send and rec queue
            self.model_process[m][0].put(m)

            # Spawn process.
            proc = multiprocessing.Process(target=model_process, args=self.model_process[m]+(self.semaphores,))
            proc.start()
            self.procs.append(proc)

            # Wait for response indicating that model has been loaded.
            ok = self.model_process[m][1].get()
            assert ok == "ok"
            self.logger.debug(f"Server {server_id} loaded model {m}")

        # Initialize model queues.
        self.queues = []
        for _ in range(num_queues):
            self.queues.append(BertQueue())

        # Gear attributes.
        self.cur_qps = 0
        self.cur_ensemble = gear_plan.gears[0].models
        self.cur_threshs = gear_plan.gears[0].threshs
        self.cur_batch_sizes = [1] * num_queues # TODO: gear_plan.gears[0].batch_sizes
        self.q_counter = 0


    def set_qps_debug(self, qps):
        self.cur_qps = qps
        self.cur_ensemble = self.gear_plan.gears[qps].models
        self.cur_threshs = self.gear_plan.gears[qps].threshs
        self.logger.debug(f"Set QPS {qps}")
        # self.cur_batch_sizes = self.gear_plan.gears[qps].server_batch_sizes[self.server_id]


    def alive(self):
        """
        Check if actor alive.
        :return:
        """
        return 1


    def switch_gear(self, qps):

        # qps *= 4
        # fac = 0.2
        # qps = self.cur_qps*fac + (1-fac)*qps
        # qps = int(qps)

        qps = min(qps, len(self.gear_plan.gears)-1)

        self.logger.debug(f"switch gear {self.cur_qps} -> {qps}: Q: {self.queues[0].size()}")

        # 1. Make sure there are only few samples in queue before downgrading.
        if self.cur_qps > qps and 8 * self.queues[0].size() > qps:
            return

        # 2. Switch to new cascade.
        self.cur_qps = qps
        self.cur_ensemble = self.gear_plan.gears[qps].models
        self.cur_threshs = self.gear_plan.gears[qps].threshs
        # self.cur_batch_sizes = self.gear_plan.gears[qps].server_batch_sizes[self.server_id] #TODO


    def warmup(self, samples, num_runs=2):
        """
        Run samples through all models
        :param samples:
        :return:
        """
        samples = (samples[0].to("cuda"), samples[1].to("cuda")) # TODO: BERT specific

        for _ in range(num_runs):
            # Send inputs to models.
            for model_name in self.model_process:
                process_queue = self.model_process[model_name][0]
                process_queue.put((samples, [0], 0.0))

            # Wait for models to finish prediction.
            for model_name in self.model_process:
                process_queue = self.model_process[model_name][1]
                process_queue.get()

        return 0


    def add_to_queue(self, sample_ids, samples, queue_idx):
        start = time.time()
        self.logger.debug(f"add to queue {queue_idx}")
        self.queues[queue_idx].add(sample_ids, samples)
        self.logger.debug(f"added to queue {queue_idx}, {time.time() - start}")


    async def infer(self, queue_id):
        """
        Do inference in other process.
        :param queue_id:
        :return:
        """
        # 1. Load balance among GPUs on server.
        # # TODO: Need to integrate this with gear plan placement.
        # q_counter = self.q_counter
        # if model_name in self.model_process:
        #     # TODO: Hacky
        #     if model_name == "llama_70b":
        #         replica = -1
        #     elif model_name == "llama_07b":
        #         replica = 1
        #     else:
        #         replica = 0
        #
        #     #assert model_name == "llama_70b"
        #     #replica = -1 # TODO: Assumes only llama_70b enters here
        #     process_queue = self.model_process[model_name]
        # else:
        #     replica = self.q_counter % 2
        #     process_queue = self.model_process[f"{model_name}_repl_{replica}"]
        #
        # self.q_counter += 1

        # 1. Get samples and model to run.
        model_name = self.cur_ensemble[queue_id]
        if queue_id >= len(self.cur_ensemble) - 1:
            thresh = -1.0
        else:
            thresh = self.cur_threshs[queue_id+1]
        samples, sample_ids = self.queues[queue_id].get() # TODO: batch size
        process_queue = self.model_process[model_name]

        # 2. Asynchronously run inference on other process.
        await self.loop.run_in_executor(None, lambda: process_queue[0].put((samples, sample_ids, thresh)))

        casc_samples, casc_sample_ids, done_preds, done_sample_ids \
            = await self.loop.run_in_executor(None, lambda: process_queue[1].get())
        self.logger.debug(f"Done with inference: {model_name} ({self.cur_ensemble}), samples: {len(sample_ids)}") #"\n Completed: {done_sample_ids}; Cascaded: {casc_sample_ids}")

        return casc_samples, casc_sample_ids, done_preds, done_sample_ids


    async def check_queue(self, queue_id):
        if queue_id == 0:
            self.logger.debug(f"Queue sizes: {[q.size() for q in self.queues]}")

        # 1. Check if queue full enough to trigger inference.
        queue = self.queues[queue_id]
        batch_size = 1 # TODO: self.cur_batch_sizes[queue_id]
        if queue_id >= len(self.cur_ensemble) or queue.size() < batch_size:
            return None, None

        # 2. Do inference (actual inference is in other process).
        casc_samples, casc_sample_ids, done_preds, done_sample_ids = await self.infer(queue_id)

        # 3. Cascade samples that need to be cascaded, return done ones.
        # NOTE: Assume all models on server for now.
        if queue_id < len(self.cur_ensemble) - 1:
            self.add_to_queue(casc_sample_ids, casc_samples, queue_id + 1)

        self.logger.debug(f"Return: {done_sample_ids}")
        return done_sample_ids, done_preds


    def close(self):
        for proc in self.procs:
            proc.terminate()

        return 0


if __name__ == "__main__":
    from offline.gear_plan import load_gear_plan
    from workloads.llama.llama_model import get_model_dict
    import time

    # 1. Initialize
    tmp_path = f"/state/partition1/user/{os.environ['USER']}/raytmp"
    os.makedirs(tmp_path, exist_ok=True)
    os.environ['TMPDIR'] = tmp_path  # Create symlink to shorten tmp dir path for ray.

    gear_plan = load_gear_plan("../offline/cached_runs/hellaswag/debug")
    model_dict = get_model_dict("supercloud")
    server_model_dict = {}
    server_model_dict["llama_13b_repl_0"] = model_dict["llama_13b_repl_0"]
    server_model_dict["llama_13b_repl_1"] = model_dict["llama_13b_repl_1"]

    server = MultiGpuServer.remote(server_id=0,
                                    gear_plan=gear_plan,
                                    models=server_model_dict)

    ray.get(server.alive.remote())
    warm_up_q = ("The question is ", ["live or not to live.", "a good one.", "wet rainy weather.", "option 4."])

    # 2. Fill queue
    server.queues[0].add(list(range(12)), [warm_up_q]*12)

    # Warm up
    ray.get([server.check_queue() for _ in range(2)])

    # Measure.
    start = time.time()
    tasks = [server.check_queue() for _ in range(2)]
    ray.get(tasks)
    print("TIME:", time.time() - start)
