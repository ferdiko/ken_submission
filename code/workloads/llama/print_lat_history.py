"""
TODO: Move to plotting when done with debugging
"""

import numpy as np
from utils.helpers import WorkloadConfig, WorkloadTrace

if __name__ == "__main__":

    warmup = 4

    w_config = WorkloadConfig("hellaswag")

    with open("first_try_log_consumer.txt", "r") as f:
        cons_lines = f.readlines()[warmup:]

    with open("first_try_log_producer.txt", "r") as f:
        prod_lines = f.readlines()[warmup:]

    trace = WorkloadTrace(w_config.workload_trace, w_config.scaling_factor)
    history = trace.get_history(1200) #len(cons_lines))
    
    #history = np.concatenate((history, [0,0,0,0,0,1,0,0,1,0,0,1,1,0,0]))
    print(history)
    
    print("SUM:", np.sum(history), len(cons_lines))

    # history = [0, 6, 2, 1, 0, 0, 0, 4, 1, 2, 0, 9, 3, 0, 0, 0, 0, 1, 0, 0]

    # Print.
    lats = []
    line_ctr = 0
    for qps in history:
        print("QPS", qps, "="*10)
        for q in range(qps):
            c = cons_lines[line_ctr]
            p = prod_lines[line_ctr]
            c = float(c.split(",")[0])
            p = float(p.split(",")[0])
            line_ctr += 1
            print(c-p)
            lats.append(c-p)

    
    # get latecies.
    print("\nLatencies:")
    lats = sorted(lats)
    print("p99:   ", lats[int(len(lats)*0.99)]) 
    print("p95:   ", lats[int(len(lats)*0.95)])
    print("p90:   ", lats[int(len(lats)*0.90)])
