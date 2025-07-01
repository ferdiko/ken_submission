from model import ExLlama, ExLlamaCache
from tokenizer import ExLlamaTokenizer
# from generator import ExLlamaGenerator
import perplexity
from perplexity import Perplexity
import time
import torch
import torch.nn.functional as F
import argparse
import json
import math
import sys
import os
import glob
import ray

from llama_helpers import get_config



@ray.remote(num_gpus=2)
class TryActor:

    def __init__(self):
        torch.cuda._lazy_init()
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.set_printoptions(precision=10)
        self.torch_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        print(self.torch_devices)

        model_dir = "/home/gridsan/fkossmann/models/Llama-2-13B-chat-GPTQ"
        tokenizer_path = os.path.join(model_dir, "tokenizer.model")
        self.config = get_config(model_dir, "4,4")

        self.model = self.timer("Load model", lambda: ExLlama(self.config))
        self.tokenizer = self.timer("Load tokenizer", lambda: ExLlamaTokenizer(tokenizer_path))

        self.mem_base = {}
        self.mem_last = {}
        for dev in self.torch_devices:
            torch.cuda.reset_peak_memory_stats(dev)
            self.mem_base[dev] = self.mem_last[dev] = torch.cuda.max_memory_allocated(dev)

        torch.cuda.reset_peak_memory_stats("cuda")
        self.mem("Model")

        self.cache = ExLlamaCache(self.model)
        self.mem("Cache")

    def next_logits(self, input_ids, last_id_only=True, input_mask=None):
        # global model, cache

        n_logits = self.model.forward(input_ids, self.cache, last_id_only, lora=None, input_mask=input_mask)
        return n_logits

    def test(self):
        gen_tokens = 128
        max_seq_len = self.config.max_seq_len
        ids = torch.randint(0, 31999, (1, max_seq_len - gen_tokens)).cuda()

        # Warm up
        for i in range(1, 3):
            self.begin()
            print(f" -- Warmup pass {i}...")
            logits = self.timer("Warmup", lambda: self.next_logits(ids))

        # Actual benchmark

        self.begin()

        t = time.time()

        print(" -- Inference, first pass.")
        logits = self.timer("Inference", lambda: self.next_logits(ids))

        t = time.time() - t
        print(f" ** Speed: {ids.shape[-1] / t:.2f} tokens/second")

        for j in range(2):

            t = time.time()
            print(f" -- Generating {gen_tokens} tokens, {ids.shape[-1]} token prompt...")
            for i in range(gen_tokens):
                logits = logits[0, -1, :]
                token = torch.argmax(logits)
                next_id = token.unsqueeze(0).unsqueeze(0)
                logits = self.next_logits(next_id)

            t = time.time() - t
            print(f" ** Speed: {gen_tokens / t:.2f} tokens/second")

            ids = ids[:, :4]
            self.cache.current_seq_len = 4

        self.mem("Inference")
        self.mem("Total", total=True)

    def begin(self):
        if self.cache is None:
            self.cache = ExLlamaCache(self.model)
        else:
            self.cache.current_seq_len = 0


    #
    # Helpers
    #
    def mem(self, name, total = False):
        #global mem_base, mem_last

        res = f" ** VRAM, {name}: "
        first = True

        for device in self.torch_devices:
            mem_c = torch.cuda.max_memory_allocated(device)
            mem_this = mem_c - self.mem_last[device] if not total else mem_c - self.mem_base[device]
            self.mem_last[device] = mem_c

            if not first: res += " - "
            first = False
            res += f"[{device}] {mem_this / (1024 ** 2):,.2f} MB"

        print(res)

    def timer(self, name, func):
        t = time.time()
        ret = func()
        t = time.time() - t
        print(f" ** Time, {name}: {t:.2f} seconds")
        return ret

# def tokenize(text):
#     global tokenizer
#
#     return tokenizer.encode(text)



if __name__ == "__main__":
    os.environ['TMPDIR'] = "/home/gridsan/fkossmann/ray_tmp"
    ray.init()
    tryactor = TryActor.remote()
    ray.get(tryactor.test.remote())



