import os

import matplotlib.pyplot as plt
import json

import numpy as np


def get_acc(filename):
    with open(filename, "r") as f:
        lines = f.readlines()[:500]

    corr = 0
    incorr = 0
    for l in lines:
        l = l.split(",")
        if l[0] == l[1]:
            corr += 1
        else:
            incorr += 1
    return corr/(incorr+corr)

if __name__ == "__main__":
    # visualize RL results
    filename = "../offline/cached_runs/wikitext/perlim.json"
    with open(filename, 'r') as file:
        data = json.load(file)

    plt.scatter(data["runtime"], data["accs"], marker='o', label="Ensembles found by RL")

    # get single model accs
    dir = "../profiling/predictions/wikitext5"
    accs = []
    for f in os.listdir(dir):
        if f[-4:] == ".csv" and f[:7] != "ignore_":
            print(f)
            a = get_acc(os.path.join(dir, f))
            accs.append(a)

    # get single model costs
    filename = "../profiling/models/llama.json"
    with open(filename, "r") as f:
        data = json.load(f)

    costs = []
    for m in data:
        if m != "large" and m != "llama_03b":
            costs.append(np.sum(data[m]["runtime"]["default"]))

    # TODO: For now just sort to match
    accs = sorted(accs)
    costs = sorted(costs)
    plt.scatter(costs, accs, label="Single BERT models")

    plt.legend()
    plt.title('Acc vs. avg. GPU runtime')
    plt.xlabel('Avg. GPU runtime to predict a batch of 32')
    plt.ylabel('Accuracy')
    plt.show()