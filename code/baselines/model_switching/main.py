"""
Create gear plan for model switching.
https://www.usenix.org/conference/hotcloud20/presentation/zhang
"""
import json

import numpy as np

from offline.gear import Gear
from offline.gear_plan import GearPlan


def read_in_max_throughput(file):
    with open(file, "r") as f:
        data = json.load(f)

    max_throughput = {}

    for model_name in data:
        # Some profiled models are not used in our experiments.
        if model_name == "large":
            continue

        max_batch_size = max([bs for bs in data[model_name]['runtime'] if bs != "default"])
        max_throughput[model_name] = 1.0/float(data[model_name]['runtime'][max_batch_size])*int(max_batch_size)

    return max_throughput


def gear_plan_from_throughput(max_throughput, max_qps=7000, num_gpus=2):
    # Initialize gear plan.
    sorted_models = sorted((value, key) for key, value in max_throughput.items())
    gears = []
    cur_model_idx = 0
    models_used = [] # Assume they fit onto 1 device for now.
    for qps in range(0, 7000+1):
        t_put, model = sorted_models[cur_model_idx]

        while t_put*num_gpus < qps:
            cur_model_idx += 1
            t_put, model = sorted_models[cur_model_idx]

        gears.append(Gear(qps=qps, models=[model], threshs=[0.5], accuracy=0.0))
        if model not in models_used:
            models_used.append(model)

    # Create gear plan
    qps_dist = np.zeros(max_qps+1, dtype=float)
    qps_dist[0] = 1
    gear_plan = GearPlan(num_gpus=num_gpus, qps_distribution=qps_dist, gears=gears)
    gear_plan.gpus_on_server = [1]*num_gpus
    gear_plan.model_placement = []
    for _ in range(num_gpus):
        gear_plan.model_placement.append(models_used)

    return gear_plan


if __name__ == "__main__":
    num_gpus = 1

    profile_path = "../../profiling/models/bert.json"
    max_throughput = read_in_max_throughput(profile_path)

    store_path = "../../offline/cached_runs/twitter/model_switch"
    gear_plan = gear_plan_from_throughput(max_throughput)
    gear_plan.print()
    gear_plan.store(store_path)
