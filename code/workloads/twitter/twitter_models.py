import time

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import numpy as np

from utils.logger_setup import setup_logger
from workloads.twitter.twitter_helpers import *

"""
Wrapper for model
"""
class TwitterModel(nn.Module):

    def __init__(self, path, model_name, max_len=55, queue_space=1000):
        super(TwitterModel, self).__init__()

        self.path = path
        # self.batch_size = 32
        self.max_len = max_len
        # self.queue_start = 0
        self.num_queued = 0
        # self.queue_len = queue_space
        # self.x_queue = None
        # self.x_att_queue = None
        # self.queued_ids = None
        self.model = None
        # self.vec_len = max_len
        self.logger = setup_logger()
        # self.prep = TwitterPrep(tokenizer_path)


        # TODO Tmp hack: Until we have new model weights, just use old, cached predictions.
        self.counter = 0
        pred_path = os.path.join("/home/gridsan/fkossmann/ensemble_serve/profiling/predictions/twitter", model_name+".csv")

        # read in certs and preds
        with open(pred_path, "r") as f:
            lines = f.readlines()
        self.preds = [int(l.split(",")[0]) for l in lines]
        self.certs = [float(l.split(",")[2]) for l in lines]

        # 2. Read in tweets.
        # prep = TwitterPrep("/home/gridsan/fkossmann/prelim/mixture-of-experts/models/bert_large_tokenizer")
        # tweets = get_tweets(500)
        # tweets = prep.prep(tweets)
        # tweets = [tweets[i] for i in range(100)]
        # #tweets = [(a.to("cuda"), b.to("cuda")) for a,b in tweets]
        #
        #
        # x = torch.concat([s[0] for s in tweets])
        # x_att = torch.concat([s[1] for s in tweets])
        #
        # self.x = x.to("cuda")
        # self.x_att = x_att.to("cuda")
        #
        # print("X IMP", self.x.shape)
        # print("ATT IMP", self.x_att.shape)



    def load(self, device="cuda"):
        """
        Initialize model and queues on device
        :return:
        """
        # self.x_queue = torch.empty((self.queue_len, self.max_len), dtype=torch.int, device=torch.device(device))
        # self.x_att_queue = torch.empty((self.queue_len, self.max_len), dtype=torch.int, device=torch.device(device))
        # self.queued_ids = torch.empty((self.queue_len,), dtype=torch.int, device=torch.device(device))
        self.model = BertForSequenceClassification.from_pretrained(self.path).to(device)
        self.model.eval()


    def add_to_queue(self, x, x_att, ids):
        """
        Add samples to queue
        :param x: tokenized x vector
        :param x_att: attention mask
        :param ids: ids of added samples
        :return:
        """
        # self.logger.debug(f"add to queue  {x.shape}, {x_att.shape}")
        assert len(x.shape) == 2
        num_samples = x.shape[0]

        # TODO: We don't do the ring buffer thing right because we assume
        # we never fill the queue.
        start = self.add_start
        end = self.add_start + num_samples

        try:
            self.x_queue[start:end, :] = x[:, :self.vec_len]
        except RuntimeError as e:
            raise RuntimeError("Model queue ran out of space", e)

        self.x_att_queue[start:end] = x_att[:, :self.vec_len]
        self.queued_ids[start:end] = ids

        # TODO: We don't do the ring buffer thing right because we assume
        # we never fill the queue.
        self.num_queued += num_samples
        self.add_start = (self.add_start + num_samples) % self.queue_len


    def get_pass_on(self, pass_on):
        return self.queued_ids[:pass_on.shape[0]][pass_on], \
            self.x_queue[:pass_on.shape[0]][pass_on], \
            self.x_att_queue[:pass_on.shape[0]][pass_on]

    def forward(self, samples, sample_ids):
        # start = time.time()

        # 1. Get inputs.
        x, x_att = samples

        # 2. Do inference.
        with torch.no_grad():
            y = self.model(x, x_att)
            y = y["logits"]
            preds = torch.argmax(y, dim=1)
            certs = torch.abs(y[:, 0] - y[:, 1])

        # 3. TODO temporary hack until we trained new weights
        certs = []
        preds = []

        for s_id in sample_ids:
            s_id = s_id % len(self.certs)
            certs.append(self.certs[s_id])
            preds.append(self.preds[s_id])

        certs = torch.tensor(certs)
        # preds = torch.tensor(preds)

        # end = time.time()
        # self.logger.debug(f"{self.path[-8:]}: inference time: {end-start}, sample_size: {len(samples)}")
        return certs, preds


    def forward_direct(self, x, x_att):
        # for profiling
        return self.model(x, x_att)

"""
Wrapper for preprocessing (tokenizer)
"""
class TwitterPrep:

    def __init__(self, tokenizer_path, max_len=55):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len

    def prep(self, x):
        assert type(x) == list
        X = []
        X_att = []
        X_comb = []
        for sample in x:
            x_tokenized = self.tokenizer.encode(sample)[:self.max_len]
            x_padded = np.array([x_tokenized + [0] * (self.max_len - len(x_tokenized))])
            x_attention_masked = np.where(x_padded != 0, 1, 0)

            X.append(x_padded)
            X_att.append(x_attention_masked)
            X_comb.append((torch.tensor(x_padded), torch.tensor(x_attention_masked)))


        X = np.array(X)
        X_att = np.array(X_att)
        X = torch.tensor(X).squeeze(1)
        X_att = torch.tensor(X_att).squeeze(1)

        return X, X_att


def get_model_dict(machine):
    """
    Load models to cpu and return list with references, sorted from small to large
    :return:
    """

    # paths to model weights on supercloud
    hub_dir = "/home/gridsan/fkossmann/prelim/mixture-of-experts/models"
    supercloud_paths = {
        "tiny": "bert_tiny",
        "mini": "bert_mini",
        "small": "bert_small",
        "medium": "bert_medium",
        "base": "bert_base",
        # "large": "bert_large",
        # "distil": "distilbert"
    }

    for k, v in zip(supercloud_paths.keys(), supercloud_paths.values()):
        supercloud_paths[k] = os.path.join(hub_dir, v)

    supercloud_tokenizer = os.path.join(hub_dir, "bert_large_tokenizer")

    # paths to model weights on macbook
    # hub_dir = "~/.cache/huggingface/hub/"
    macbook_paths = {
        "tiny": "google/bert_uncased_L-2_H-128_A-2",
        "mini": "google/bert_uncased_L-4_H-256_A-4",
        "small": "google/bert_uncased_L-4_H-512_A-8",
        "medium": "google/bert_uncased_L-8_H-512_A-8",
        "base": "google/bert_uncased_L-12_H-768_A-12",
        # "large": "bert-large-uncased",
        # "distil": "models--distilbert-base-uncased"
    }

    macbook_tokenizer = macbook_paths["tiny"]

    # select which paths to use
    if machine == "supercloud":
        paths = supercloud_paths
        tokenizer_path = supercloud_tokenizer
    elif machine == "macbook":
        paths = macbook_paths
        tokenizer_path = macbook_tokenizer
    else:
        assert False, f"{machine} not recognized: Should be 'supercloud' or 'macbook'"

    # init models and tokenizer
    models = {}
    for model_id in paths:
        models[model_id] = TwitterModel(path=paths[model_id], model_name=model_id)

    prep = TwitterPrep(tokenizer_path)

    return models, prep

if __name__ == "__main__":
    tweets = ["hello my name is bert."]*16
    models, prep = get_model_dict("supercloud")

    X = prep.prep(tweets)
    print("Shape", X[0][0].shape)

    # x, x_att = prep.prep(["It's beautiful"])
    models["tiny"].load("cuda")
    for _ in range(5):
        print(models["tiny"](X))
        print()
