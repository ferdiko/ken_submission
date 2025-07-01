import json
from copy import copy
import numpy as np
from offline.batch_size.latency_simulator import LatencySimulator


class BatchSizeOptimizer:

    def __init__(self, paths):

        self.paths = paths

        # load dict with runtimes for batch sizes of different model
        with open(paths.model_profile, "r") as f:
            self.profile = {}
            profile_json = json.load(f)
            for k_model in profile_json.keys():
                self.profile[k_model] = {}
                for k_bs in profile_json[k_model].keys():
                    self.profile[k_model][int(k_bs)] = profile_json[k_model][k_bs]

            self.max_bs = max(self.profile[k_model].keys())


    def optimize_bs_on_gpu(self, gear, gpu_idx, p_latency, slo):
        print("enter", gpu_idx)
        # init simulator
        sim = LatencySimulator(batch_size_profiles=self.profile,
                               p_latency=p_latency,
                               pred_dir=self.paths.pred_dir)

        # start all at bs 1
        batch_sizes = np.ones(len(gear.gpu_models[gpu_idx]), dtype=int)
        gear.gpu_batch_sizes[gpu_idx] = batch_sizes

        # hill climbing
        while True:
            change_dir = np.zeros_like(batch_sizes)
            util = np.zeros_like(batch_sizes) # runtime improvement

            cur_runtime = sim.simulate_const_qps(gear, gpu_idx)

            print("here", gear.total_qps, batch_sizes, cur_runtime)

            for i, bs in enumerate(batch_sizes):
                if bs > 1:
                    bs_copy = copy(batch_sizes)
                    bs_copy[i] -= 1
                    gear.gpu_batch_sizes[gpu_idx] = bs_copy
                    sim.reset()
                    new_runtime = sim.simulate_const_qps(gear, gpu_idx)
                    util[i] = max(0, cur_runtime - new_runtime)
                    change_dir[i] = -1

                if bs < self.max_bs and util[i] == 0:
                    bs_copy = copy(batch_sizes)
                    bs_copy[i] += 1
                    gear.gpu_batch_sizes[gpu_idx] = bs_copy
                    sim.reset()
                    new_runtime = sim.simulate_const_qps(gear, gpu_idx)
                    util[i] = max(0, cur_runtime - new_runtime)
                    change_dir[i] = 1

            # make step up-hill
            argmax_util = np.argmax(util)
            max_util = util[argmax_util]
            if max_util == 0:
                if cur_runtime <= slo:
                    return True
                else:
                    return False

            batch_sizes[argmax_util] = batch_sizes[argmax_util] + change_dir[argmax_util]
            gear.gpu_batch_sizes[gpu_idx] = batch_sizes


    def optimize(self, gear, p_latency, slo=np.inf):
        for gpu_idx in range(len(gear.gpu_batch_sizes)):
            ok = self.optimize_bs_on_gpu(gear, gpu_idx, p_latency, slo)
            if not ok:
                return False

        return True
