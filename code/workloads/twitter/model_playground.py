from twitter_models import *
import ray
import pandas as pd
import torch
import time
import sys
import os
sys.path.append("../../")
from online.frontend import Frontend


ISSUE_INTERVAL = 0.0005


def iterative_inference(frontend, prep, models, NUM_TRAIN_SAMPLES=0, baseline=False):
    dataset = pd.read_csv("twitter_sentiment.csv", encoding='latin-1', names=['target', 'id', 'date', 'query', 'username', 'text'])
    if NUM_TRAIN_SAMPLES > 0:
        dataset = dataset.sample(frac=NUM_TRAIN_SAMPLES/len(dataset))

    if baseline:
        models[0].load("cpu")
        id_counter = 0

    for i, text in enumerate(dataset['text']):

        x = prep.prep(text)

        if i == 50000:
            time.sleep(10)
            break

        if not baseline:
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


        else:
            assert False
            ids = []
            for _ in range(x[0].shape[0]):
                # sample_id = self.sample_store.put([x[i] for x in samples])
                # id = self.sample_store.put(sample_ref)
                # ids.append(sample_id)

                ids.append(id_counter)
                # sample_ref = ray.put([x[i] for x in samples])
                # self.sample_store[self.id_counter] = sample_ref
                id_counter += 1

            # add to queue
            ids_vec = torch.IntTensor(ids)

            models[0].add_to_queue(x[0], x[1], ids_vec)

            if id_counter % 8 == 0:
                models[0]()

        time.sleep(ISSUE_INTERVAL)


if __name__ == "__main__":
    # get environment
    num_gpus = torch.cuda.device_count()
    TRAIN = False
    BATCH_SIZE = 8

    # if there are GPUs assume you are on supercloud. otherwise assume local
    if num_gpus > 0:
        machine = "supercloud"
        print("supercloud, num gpus:", num_gpus)

        # create symlink to shorte tmp dir path for ray
        os.environ['TMPDIR'] = "/home/gridsan/fkossmann/ray_tmp"

    else:
        machine = "macbook"
        num_gpus = 1

    # get path to model weights
    models, prep = get_model_dict(machine)
    print("got model dict")

    # make batches for inference
    dataset = pd.read_csv("twitter_sentiment.csv", encoding='latin-1', names=['target', 'id', 'date', 'query', 'username', 'text'])

    for i, text in enumerate(dataset['text']):

       x = prep.prep(text)
