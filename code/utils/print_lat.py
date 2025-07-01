"""
TODO: Move to plotting when done with debugging
"""

import numpy as np
from utils.helpers import WorkloadConfig, WorkloadTrace
from workloads.llama.llama_helpers import get_hellaswag

if __name__ == "__main__":

    warmup = 8

    w_config = WorkloadConfig("hellaswag")


    prefix = "6gpu_fixed_"

    with open(prefix+"consumer.txt", "r") as f:
        cons_lines = f.readlines()[warmup:]

    with open(prefix+"producer.txt", "r") as f:
        prod_lines = f.readlines()[warmup:]

    with open("simulated_preds.csv", "r") as f:
        sim_lines_raw = f.readlines()
        sim_lines = [l for l in sim_lines_raw if l[0] != " "]


    trace = WorkloadTrace(w_config.workload_trace, w_config.scaling_factor, seed=2)
    history = trace.get_history(1200) #len(cons_lines))

    _, labels = get_hellaswag(500)


    #history = np.concatenate((history, [0,0,0,0,0,1,0,0,1,0,0,1,1,0,0]))
    print(history)

    print("Qs in history:", np.sum(history), "Qs in consumer:", len(cons_lines), "Qs in producer:", len(prod_lines), "Qs in sim:", len(sim_lines))

    # Print.
    lats = []
    line_ctr = 0
    corr = 0
    incorr = 0
    
    diff_count = 0
    for qps in history:
        print("QPS", qps, "="*10)
        for q in range(qps):
            if line_ctr >= len(cons_lines):
                print("TOO FEW LiNES IN CONSUMER FILE", "="*10)
                break

            #c = cons_lines[line_ctr]
            p = prod_lines[line_ctr]
            #c = c.split(",")
            p = p.split(",")

            sample_id = int(p[1])
            assert sample_id >= warmup
            c = cons_lines[sample_id-warmup].split(",")


            c_time = float(c[0])
            p_time = float(p[0])
            print(c_time-p_time, end="")
            lats.append(c_time-p_time)

            # get acc
            pred_id = (int(c[1])-warmup)%500
            pred = int(c[2])
            if pred == labels[pred_id]:
                corr += 1
            else:
                incorr += 1
            
            
            sim_pred = int(sim_lines[pred_id].split(",")[0])
            if pred != int(sim_pred):
                diff_count += 1
                print("\tFalse", int(c[1]))
            else:
                print()
            

            # TODO: print if matches simulated

            line_ctr += 1


        if line_ctr >= len(cons_lines):
            break

    # print acc.
    print("\nAcc:")
    print(corr/(incorr+corr))

    print("diff:", diff_count, diff_count/(incorr+corr))

    # get latecies.
    print("\nLatencies:")
    lats = sorted(lats)
    print("p99:   ", lats[int(len(lats)*0.99)])
    print("p95:   ", lats[int(len(lats)*0.95)])
    print("p90:   ", lats[int(len(lats)*0.90)])
