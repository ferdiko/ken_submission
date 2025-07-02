import json
from queue import PriorityQueue

import numpy as np

from offline.gear_plan import GearPlan
from placement.placement_optimizer import PlacementOptimizer
from utils.helpers import *


# def compute_start_plan(gear_distribution, ensemble_dict, num_gpus, max_qps=-1):
#     """
#     Start plan: Use ensemble with highest accuracy for each QPS.
#     :param gear_distribution:
#     :param ensemble_dict:
#     :param num_gpus:
#     :param max_qps:
#     :return: gear plan
#     """
#     if max_qps == -1:
#         max_qps = len(gear_distribution)
#
#     desc = np.zeros(max_qps, dtype=int)
#     return GearPlan(desc, gear_distribution, ensemble_dict, num_gpus)


def optimize_acc(paths, p_latency, num_gpus, lat_slo=None, gpu_capacity=1.0):
    # Get QPS distribution of workload trace.
    trace = WorkloadTrace(paths.workload_trace, paths.scaling_factor)
    qps_distribution = trace.get_histogram()

    # Load model profile
    # with open(paths.model_profile, 'r') as f:
    #     model_profile = json.load(f)

    # Get system info
    with open(paths.hardware_profile, 'r') as f:
        gpu_profile = json.load(f)

    gpu_memory = gpu_profile["gpu_memory"]
    # gpu_capacity = 1 # NOTE: Always 1, just used for debugging

    # Read ensembles from file
    with open(paths.ensemble_path, 'r') as f:
        ensemble_dict = json.load(f)

    # compute start node
    start_plan = GearPlan(num_gpus=num_gpus,
                          qps_distribution=qps_distribution,
                          ensemble_dict=ensemble_dict)

    # init placement optimizer
    po = PlacementOptimizer(num_gpus=num_gpus,
                            gpu_memory=gpu_memory,
                            gpu_compute=gpu_capacity,
                            ensembles=ensemble_dict["ensembles"],
                            threshs=ensemble_dict["threshs"],
                            w_config=paths,
                            lat_slo=lat_slo)

    # do beam search over plans
    current_plan = start_plan
    error_code = ""
    while error_code != "success":
        # print("top level", current_plan.desc)

        # for i, d in enumerate(current_plan.desc):
        #     if i % 10 == 0:
        #         print(f"{i}: {d}")

        error_code, broken_qps = po.optimize(current_plan, p_latency=p_latency)

        if error_code == "latency_slo":
            print("================== downgrade qps", broken_qps, "==================")
            # put neighbours into priority queue
            current_plan.downgrade(broken_qps)
            print("new acc:", current_plan.total_acc())
            # neighbours = current_plan.downgrade(broken_qps)
            # for n in neighbours:
            #     ensemble_queue.put(n)
            # current_plan = ensemble_queue.get()

    # Hack
    current_plan.finalize()

    # Print and return final plan.
    current_plan.print()
    return current_plan


def optimize_lat(paths, num_gpus, acc_slo):
    # Get QPS distribution of workload trace.
    trace = WorkloadTrace(paths.workload_trace, paths.scaling_factor)
    qps_distribution = trace.get_histogram()

    # Load model profile
    # with open(paths.model_profile, 'r') as f:
    #     model_profile = json.load(f)

    # Get system info
    with open(paths.hardware_profile, 'r') as f:
        gpu_profile = json.load(f)

    gpu_memory = gpu_profile["gpu_memory"]
    gpu_capacity = 1000 # NOTE: Always 1, just used for debugging

    # Read ensembles from file
    with open(paths.ensemble_path, 'r') as f:
        ensemble_dict = json.load(f)

    # compute start node
    start_plan = GearPlan(num_gpus=num_gpus,
                          qps_distribution=qps_distribution,
                          ensemble_dict=ensemble_dict,
                          init_best=False)

    # init placement optimizer
    po = PlacementOptimizer(num_gpus=num_gpus,
                            gpu_memory=gpu_memory,
                            gpu_compute=gpu_capacity,
                            ensembles=ensemble_dict["ensembles"],
                            threshs=ensemble_dict["threshs"],
                            w_config=paths,
                            lat_slo=-1)

    # do beam search over plans
    current_plan = start_plan
    error_code = ""
    num_gears = qps_distribution.shape[0]
    cur_load = np.linspace(1, num_gears-1, num_gears-1) * np.full(num_gears-1, ensemble_dict["runtime"][-1]) # NOTE: Leave 0 at worst which can be problematic ...
    while current_plan.score < acc_slo:
        # 1. Get effect on latency if upgrade
        # NOTE: Just pick lowest load ...
        alter_qps = np.argmin(cur_load)

        print(alter_qps, ":", cur_load)

        # 2. Upgrade.
        new_gear_idx = current_plan.upgrade(alter_qps+1)

        # 3. Update cur_load.
        if new_gear_idx == 0:
            cur_load[alter_qps] = np.inf
        else:
            cur_load[alter_qps] = (alter_qps+1) * ensemble_dict["runtime"][new_gear_idx]

    # 4. Do placement optimization.
    error_code, broken_qps = po.optimize(current_plan, p_latency=-1)
    if error_code not in ["success", "latency_slo"]:
        print("urgh")


    # Print and return final plan.
    current_plan.finalize()
    current_plan.print()
    print(cur_load)
    return current_plan


if __name__ == "__main__":
    # set up
    num_gpus = 1
    gpu_capacity = 1
    paths = WorkloadConfig(workload="twitter")

    # run offline phase
    # gear_plan = optimize_lat(paths, num_gpus=num_gpus, acc_slo=0.55)
    gear_plan = optimize_acc(paths, p_latency=0.95, num_gpus=num_gpus, lat_slo=0.62, gpu_capacity=gpu_capacity)

    # store gear plan and gears
    gear_plan.store(paths.plan_dir)
