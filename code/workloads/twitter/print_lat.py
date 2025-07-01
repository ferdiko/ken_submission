"""
TODO: Move to plotting when done with debugging
"""

import numpy as np
from utils.helpers import WorkloadConfig, WorkloadTrace
from workloads.llama.llama_helpers import get_hellaswag
import sys

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

    #with open("simulated_preds.csv", "r") as f:
    #    sim_lines_raw = f.readlines()
    #    sim_lines = [l for l in sim_lines_raw if l[0] != " "]


    # trace = WorkloadTrace(w_config.workload_trace, 1.4)
    # history = trace.get_history(1200)
    history = np.load("/home/gridsan/fkossmann/ensemble_serve/profiling/traces/twitter_qps.npy")
    history *= 100
    #history = [1000]*3 + [3000, 3000, 1000, 1000]

    # history = [1000]*3 + [5000, 1000]

    with open("../../profiling/predictions/twitter/tiny.csv", "r") as f:
        lines = f.readlines() #[:500]
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

            line_ctr += 1

        if len(q_lats) > 0:
            max_q = np.argmax(q_lats)
            print(f"max lat: {q_lats[max_q]}, q: {q_ids[max_q]}, proc: {q_enter[max_q]} to {q_leave[max_q]}")

        if line_ctr >= len(cons_lines) or line_ctr >= len(prod_lines):
            break

    # print acc.
    print("\nAcc:")
    print(corr/(incorr+corr))

    # get latecies.
    print("\nLatencies:")
    lats = sorted(lats)
    print("p99:   ", lats[int(len(lats)*0.99)])
    print("p95:   ", lats[int(len(lats)*0.95)])
    print("p90:   ", lats[int(len(lats)*0.90)])
    print("p50:   ", lats[int(len(lats)*0.50)])
