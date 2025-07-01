import json

import numpy as np

from offline.gear_plan import GearPlan
from simulator.simulator import Simulation
from utils.helpers import WorkloadConfig, WorkloadTrace


"""
NOTE: We don't use this offline planning as a baseline since we
found just batchin the whole queue online gives better performance.
"""

def create_plan(model, w_config, num_gpus):
    # Get workload characteristics.
    trace = WorkloadTrace(w_config.workload_trace, w_config.scaling_factor)
    qps_distribution = trace.get_histogram()

    # Load model profile.
    with open(w_config.model_profile, 'r') as f:
        model_profile = json.load(f)
        default_rt = np.sum(model_profile[model]["runtime"]["default"])

        profile = {}
        for bs in model_profile[model]["runtime"].keys():
            if bs != "default":
                profile[int(bs)] = model_profile[model]["runtime"][bs]

    ensemble_dict = {
        "accs": [0.5],
        "runtime": [[default_rt]],
        "ensembles": [[model]],
        "threshs": [[0.5]],
    }

    # Create gear plan.
    gear_plan = GearPlan(num_gpus=num_gpus,
                         qps_distribution=qps_distribution,
                         ensemble_dict=ensemble_dict,
                         init_best=True)

    gear_plan.model_placement = [[model] for _ in range(num_gpus)]

    # Get best batch size.
    sim = Simulation(pred_dir=w_config.pred_dir, profiling_file=w_config.model_profile)

    last_best = [1]
    for qps in range(1, 200): #qps_distribution.shape[0]):
        best_lat = np.inf
        best_bs = 0
        for batch_size in range(min(last_best[-5:]), max(profile) + 1):
            gear_plan.gears[qps].server_batch_sizes = [[batch_size] for _ in range(num_gpus)]
            lat = sim.get_latency(gear_plan.gears[qps], qps)

            if lat < best_lat:
                best_lat = lat
                best_bs = batch_size
            elif batch_size > 8:
                break

        last_best.append(best_bs)
        print(qps, last_best[-10:])
        gear_plan.gears[qps].server_batch_sizes = [[best_bs] for _ in range(num_gpus)]

    # NOTE: For BERT, all QPS > 280 have max BS (100)
    for qps in range(200, qps_distribution.shape[0]):
        gear_plan.gears[qps].server_batch_sizes = [[100] for _ in range(num_gpus)]

    gear_plan.gpus_on_server = [1] * num_gpus
    gear_plan.finalize()
    return gear_plan

if __name__ == "__main__":
    # Experiment parameters.
    w_config = WorkloadConfig(workload="twitter")
    model = "tiny"
    num_gpus = 8

    # Create plan.
    gear_plan = create_plan(model, w_config, num_gpus)
    gear_plan.print()
    gear_plan.store(w_config.plan_dir)

