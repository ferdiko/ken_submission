"""
TODO: Move to plotting when done with debugging
"""

from utils.helpers import WorkloadConfig, WorkloadTrace

if __name__ == "__main__":

    w_config = WorkloadConfig("hellaswag")

    with open("first_try_log_consumer.txt", "r") as f:
        cons_lines = f.readlines()[1:]

    with open("first_try_log_producer.txt", "r") as f:
        prod_lines = f.readlines()[1:]

    trace = WorkloadTrace(w_config.workload_trace)
    history = trace.get_history(len(cons_lines))
    # history = [0, 6, 2, 1, 0, 0, 0, 4, 1, 2, 0, 9, 3, 0, 0, 0, 0, 1, 0, 0]

    # Print.
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