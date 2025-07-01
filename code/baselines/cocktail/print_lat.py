"""
TODO: Move to plotting when done with debugging
"""

import numpy as np
from utils.helpers import WorkloadConfig, WorkloadTrace
from workloads.llama.llama_helpers import get_hellaswag
import sys


def get_time_from_line(line):
    # Assume: All types of files have time stamp first.
    return float(line.split(",")[0])


def get_cost(prefix):
    with open(prefix+"cost.txt", "r") as f:
        cost_lines = f.readlines()

    if len(cost_lines) == 1:
        return float(cost_lines.split(",")[1])

    total_time = get_time_from_line(cost_lines[-1]) - get_time_from_line(cost_lines[0])
    gpu_seconds = 0
    for i0, i1 in zip(cost_lines[:-1], cost_lines[1:]):
        i0 = i0.split(",")
        i1 = i1.split(",")
        time_interval = float(i1[0]) - float(i0[0])
        gpu_seconds += float(i0[1]) * time_interval

    return gpu_seconds/total_time


if __name__ == "__main__":

    warmup = 0

    w_config = WorkloadConfig("twitter")

    prefix = sys.argv[1]

    if prefix[-1] != "_":
        prefix += "_"

    # Read in consumer log. Sort according to sample id.
    with open(prefix+"consumer.txt", "r") as f:
        cons_lines = f.readlines()[warmup:]

    cons_lines_parse = []
    for l in cons_lines:
        # Format: time, id, pred
        l = l.split(",")
        cons_lines_parse.append((int(l[1]), float(l[0]), int(l[2])))

    cons_lines = sorted(cons_lines_parse)

    with open(prefix+"producer.txt", "r") as f:
        # NOTE: Assume sorted according to id
        prod_lines = f.readlines()[warmup:]

    with open(prefix+"cost.txt", "r") as f:
        cost_lines = f.readlines()
        cost_lines = [(float(l.split(",")[0]), int(l.split(",")[1])) for l in cost_lines]

    # trace = WorkloadTrace(w_config.workload_trace, 1.4)
    # history = trace.get_history(1200)
    history = np.load("/home/gridsan/fkossmann/ensemble_serve/profiling/traces/twitter_qps.npy")
    history *= 95

    with open("../../profiling/predictions/twitter/tiny.csv", "r") as f:
        lines = f.readlines()
        labels = [int(l.split(",")[1]) for l in lines]

    print("Qs in history:", np.sum(history), "Qs in consumer:", len(cons_lines), "Qs in producer:", len(prod_lines))

    # Print.
    lats = []
    line_ctr = 0
    corr = 0
    incorr = 0

    diff_count = 0
    for qps in history:

        print("QPS", qps, "="*10)

        if qps == 0:
            continue

        q_lats = []
        q_ids = []
        q_enter = []
        q_leave = []

        for q in range(qps):
            if line_ctr >= len(cons_lines):
                print("TOO FEW LiNES IN CONSUMER FILE", "="*10)
                break

            if line_ctr >= len(prod_lines):
                print("TOO FEW LiNES IN PRODUCER FILE", "="*10)
                break

            # Get sample from producer log.
            p = prod_lines[line_ctr]
            p = p.split(",")
            sample_id = int(p[1])
            assert sample_id >= warmup # Warmup is legacy.

            q_ids.append(sample_id)
            p_time = float(p[0])

            # Get sample from consumer log.
            c = cons_lines[sample_id]
            # assert c[0] == sample_id # TODO: Comment back in
            c_time = c[1]
            pred = c[2]

            # Get sample correctness.
            sample_id = c[0]
            label = labels[sample_id%len(labels)]
            if pred == label:
                corr += 1
            else:
                incorr += 1

            # Get sample latency.
            lats.append(c_time-p_time)
            q_lats.append(c_time-p_time)
            q_enter.append(p_time)
            q_leave.append(c_time)

            # Print is rescaled.
            if cost_lines[0][0] < p_time:
                print(f"{'-'*20}> Rescaled: {cost_lines[0][1]} GPU servers")
                cost_lines = cost_lines[1:]

            line_ctr += 1

        if len(q_lats) > 0:
            max_q = np.argmax(q_lats)
            print(f"max lat: {q_lats[max_q]}, q: {q_ids[max_q]}, proc: {q_enter[max_q]} to {q_leave[max_q]}")

        if line_ctr >= len(cons_lines) or line_ctr >= len(prod_lines):
            break

    # print acc.
    print("\nAcc:")
    print(corr/(incorr+corr))

    # Print cost.
    print("\nCost:")
    print("Average GPUs:", get_cost(prefix))

    # get latecies.
    print("\nLatencies:")
    lats = sorted(lats)
    print("p99:   ", lats[int(len(lats)*0.99)])
    print("p95:   ", lats[int(len(lats)*0.95)])
    print("p90:   ", lats[int(len(lats)*0.90)])
    print("p50:   ", lats[int(len(lats)*0.50)])
