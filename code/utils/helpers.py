import copy
import json
import os
import numpy as np
from math import floor
from collections import Counter


class WorkloadTrace:
    """
    How many QPS
    """
    def __init__(self, path, scaling_factor=1.0, seed=2):
        self.path = path
        np.random.seed(seed)

        # Read in trace.
        # NOTE Format: Float time stamps with second offset
        with open(path, "r") as f:
            lines = f.readlines()
        timestamps = [float(l) for l in lines]

        # Get QPS history (for each second how many QPS).
        rounded_down = [floor(t) for t in timestamps]
        counts = Counter(rounded_down)

        # Create final list and scale.
        max_value = max(counts)
        self.qps_history = [int(counts.get(i, 0)*scaling_factor) for i in range(max_value + 1)]
        self.qps_history = np.array([i if i is not None else 0 for i in self.qps_history])

        # Make histogram.
        counts = Counter(self.qps_history)
        max_value = max(counts)
        self.qps_histogram = [counts.get(i, 0) for i in range(max_value + 1)]
        self.qps_histogram = np.array([i if i is not None else 0 for i in self.qps_histogram], dtype="float64")
        self.qps_histogram /= np.sum(self.qps_histogram)


    def get_histogram(self):
        return self.qps_histogram


    def history_to_histogram(self, history):
        counts = Counter(history)
        max_value = max(counts)
        histogram = [counts.get(i, 0) for i in range(max_value + 1)]
        histogram = np.array([i if i is not None else 0 for i in histogram], dtype="float64")
        histogram /= np.sum(histogram)
        return histogram


    def get_history(self, length=-1):
        if length == -1:
            return self.qps_history

        # Sample from histogram to reach desired length.
        prob_dist = self.get_histogram()
        return np.random.choice(prob_dist.shape[0], size=length, p=prob_dist)


class WorkloadConfig:
    """
    Configuration for running each workload, including paths, etc.
    """
    def __init__(self, workload):
        # Keep all paths here.
        path_dict = {
            "twitter": {
                "predictions": "../profiling/predictions/twitter",
                "workload": "../profiling/traces/twitter_trace_sf5000.csv",
                # "ensemble": "ensembles/bert_ensembles_bs32_l512.json",
                "ensemble": "../offline/cached_runs/twitter/ensembles_old_preds_new_code2.json",
                "model_profile": "../profiling/models/bert.json", #"profile/bert_profile.json"
                "gear_plan_dir": "../offline/cached_runs/twitter/new_after_hs_w_bs"
            },
            "hellaswag": {
                "predictions": "../profiling/predictions/hellaswag_5000",
                "workload": "../profiling/traces/msft_all_apps.csv",
                "ensemble": "cached_runs/hellaswag/ensembles_500.json",
                "model_profile": "../profiling/models/llama.json",
                "gear_plan_dir": "cached_runs/hellaswag/acc_54"
            },
            "wikitext": {
                "predictions": "../profiling/predictions/wikitext5",
                "workload": "../profiling/traces/msft_all_apps.csv",
                "ensemble": "cached_runs/wikitext/perlim.json",
                "model_profile": "../profiling/models/llama.json",
                "gear_plan_dir": "cached_runs/wikitext/2gpu/"
            }
        }

        # Other config params.
        params = {
            "twitter": {
                "scaling_factor": 1.0 # 1.0 but we already read the trace scaled by 5000
            },
            "hellaswag": {
                "scaling_factor": 0.5
            },
            "wikitext": {
                "scaling_factor": 0.75 # should probably be higher
            }
        }

        # Store paths of workload as attributes.
        workload_dict = path_dict[workload]
        offline_dir = os.path.join(os.path.dirname(__file__), "../offline")
        self.pred_dir = os.path.join(offline_dir, workload_dict["predictions"])
        self.workload_trace = os.path.join(offline_dir, workload_dict["workload"])
        self.ensemble_path = os.path.join(offline_dir, workload_dict["ensemble"])
        self.model_profile = os.path.join(offline_dir, workload_dict["model_profile"])
        self.hardware_profile = os.path.join(offline_dir, "../profiling/hardware/v100_gpu.json")
        self.plan_dir = os.path.join(offline_dir, workload_dict["gear_plan_dir"])

        # Store other params as attributes.
        param_dict = params[workload]
        self.scaling_factor = param_dict["scaling_factor"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = WorkloadConfig("hellaswag")
    trace = WorkloadTrace(config.workload_trace, config.scaling_factor)
    dist = trace.get_histogram()[:35]

    print("samples", np.sum(dist*3600))

    plt.bar(list(range(dist.shape[0]-1)), dist[1:])
    plt.show()