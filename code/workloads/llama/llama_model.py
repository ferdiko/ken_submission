import numpy as np
import torch.nn.functional as F

from workloads.llama.model import ExLlama, ExLlamaCache
from workloads.llama.tokenizer import ExLlamaTokenizer
from workloads.llama.llama_helpers import *


class LlamaModel:

    def __init__(self, model_dir, model_name):
        if model_name == "llama_70b":
            self.config = get_config(model_dir, gpu_split="17,17")
        elif model_name == "llama_07b":
            self.config = get_config(model_dir, gpu_split="0,10")
        elif model_name == "llama_13b_repl_0":
            self.config = get_config(model_dir, gpu_split="20,0")
        elif model_name == "llama_13b_repl_1":
            self.config = get_config(model_dir, gpu_split="0,20")
        elif model_name == "llama_03b_repl_0":
            self.config = get_config(model_dir, gpu_split="20,0")
        elif model_name == "llama_03b_repl_1":
            self.config = get_config(model_dir, gpu_split="0,20")
        else:
            self.config = get_config(model_dir)
        self.model_dir = model_dir

        self.model = None
        self.tokenizer = None
        self.cache =  None


    def load(self):
        self.model = ExLlama(self.config)
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.model")
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)
        self.cache =  ExLlamaCache(self.model)


    @staticmethod
    def certainty(preds):
        scores_sorted = sorted(preds)
        scores_sorted = np.array(scores_sorted)
        probabilities = scores_sorted / scores_sorted.sum()
        return - (probabilities[3] - probabilities[2])


    def forward_hellaswag(self, samples):
        # TODO: Support batching
        prompt, answers = samples[0]

        scores = []
        for a in answers:
            # 1. Tokenize
            self.cache.current_seq_len = 0
            input_ids = self.tokenizer.encode(prompt+a)
            answer_ids = self.tokenizer.encode(a)

            # 2. Predict.
            logits = self.model.forward(input_ids, self.cache, lora = None, last_id_only=False)
            log_probs = F.log_softmax(logits, dim=-1)
            relevant_log_probs = log_probs[:, -answer_ids.shape[1]-1:-1]

            # Compute answer prob, average over seq length
            seq_prob = 0
            for i in range(answer_ids.shape[-1]):
                correct_token = int(answer_ids[0, i])
                seq_prob += float(relevant_log_probs[0, i, correct_token])
            scores.append(seq_prob / float(answer_ids.shape[-1]))

        # 3. Compute certainty score and return.
        return self.certainty(scores), [np.argmax(scores)]


def get_model_dict(machine):
    # Can only run on supercloud.
    assert machine == "supercloud"

    # Path to models.
    model_paths = {
        "llama_03b": "/home/gridsan/fkossmann/models/open_llama_3b_4bit_128g",
        "llama_07b": "/home/gridsan/fkossmann/models/LLaMa-7B-GPTQ/",
        "llama_13b": "/home/gridsan/fkossmann/models/LLaMa-13B-GPTQ/",
        "llama_70b": "/home/gridsan/fkossmann/models/Llama-2-70B-GPTQ/"
    }

    # Build model dict and return it.
    model_dict = {
        "llama_03b": LlamaModel(model_paths["llama_03b"], "llama_03b"),
        "llama_03b_repl_0": LlamaModel(model_paths["llama_03b"], "llama_03b_repl_0"),
        "llama_03b_repl_1": LlamaModel(model_paths["llama_03b"], "llama_03b_repl_1"),
        "llama_07b": LlamaModel(model_paths["llama_07b"], "llama_07b"),
        "llama_13b": LlamaModel(model_paths["llama_13b"], "llama_13b"),
        "llama_13b_repl_0": LlamaModel(model_paths["llama_13b"], "llama_13b_repl_0"),
        "llama_13b_repl_1": LlamaModel(model_paths["llama_13b"], "llama_13b_repl_1"),
        "llama_70b": LlamaModel(model_paths["llama_70b"], "llama_70b")
    }

    return model_dict


if __name__ == "__main__":
    from workloads.llama.llama_helpers import *

    samples = get_hellaswag(10)
    model_dict = get_model_dict("supercloud")
    model = model_dict["llama_70b"]
    print("load")
    model.load()
    print("done load")

    for s in samples:
        model.forward_hellaswag([s])
