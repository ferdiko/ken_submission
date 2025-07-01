from twitter_models import *
import ray
import pandas as pd
import torch
import time
import sys
import os
sys.path.append("../../")
from online.frontend import Frontend

NUM_TRAIN_SAMPLES = 0

ISSUE_INTERVAL = 0.0005

def iterative_inference(frontend, prep, dataset, qps, counter):

    issue_interval = 1/qps

    for i, text in enumerate(dataset['text'][counter:counter+qps]):
        # Infer.
        x = prep.prep([text])
        frontend.infer(x)

        # if i == 10:
        #     batch_size_const = 8
        #     cert_thresh_const = 0.2
        #     batch_sizes = {}
        #     cert_threshs = {}
        #     for m in models:
        #         batch_sizes[m] = batch_size_const
        #         cert_threshs[m] = cert_thresh_const
        #
        #     frontend.set_gear(["tiny"], cert_threshs, batch_sizes)

        time.sleep(issue_interval)

    print(i)

def run(frontend, prep):
    # get qps history
    qps = [5] * 5
    qps += [20] * 5
    counter = 0

    # read dataset
    dataset = pd.read_csv("twitter_sentiment.csv", encoding='latin-1',
                          names=['target', 'id', 'date', 'query', 'username', 'text'])
    if NUM_TRAIN_SAMPLES > 0:
        dataset = dataset.sample(frac=NUM_TRAIN_SAMPLES / len(dataset))

    for q in qps:
        iterative_inference(frontend, prep, dataset, q, counter)
        counter += q


if __name__ == "__main__":

    # get environment
    num_gpus = torch.cuda.device_count()

    # if there are GPUs assume you are on supercloud. otherwise assume local
    if num_gpus > 0:
        machine = "supercloud"
        print("supercloud, num gpus:", num_gpus)

        # use symlink to shorten tmp dir path for ray (needed on supercloud)
        os.environ['TMPDIR'] = "/home/gridsan/fkossmann/ray_tmp"

    else:
        machine = "macbook"
        num_gpus = 1

    ray.init()

    # TODO: At some point you need to use the finetuned models
    models, prep = get_model_dict(machine)

    # Initialize frontend.
    frontend = Frontend(num_gpus=num_gpus, models=models)

    time.sleep(2)

    # Run workload.
    # iterative_inference(f, prep)
    run(frontend, prep)

    time.sleep(10)
