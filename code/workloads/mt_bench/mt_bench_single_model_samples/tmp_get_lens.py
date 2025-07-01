from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np

with open("2x1600_70b.txt", "r") as f:
    lines = f.readlines()
    lines = lines[80:]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

lens = [len(tokenizer.encode(l)) for l in lines]

print(lens)

print()
print(np.mean(lens), np.max(lens), np.min(lens))