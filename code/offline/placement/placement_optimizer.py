import json
import math
import os

import numpy as np

from offline.batch_size.batch_size_handcrafted import BatchSizeOptimizerManual
from offline.placement.load_balancer import LoadBalancer
# from offline.batch_size.batch_size_optimizer import BatchSizeOptimizer
from utils.logger_setup import setup_logger
from simulator.simulator import Simulation


logger = setup_logger()

class PlacementOptimizer:

    def __init__(self, num_gpus, gpu_memory, gpu_compute, ensembles, threshs, w_config, lat_slo):
        """
        Optimize the placement of models on GPUs
        :param num_gpus:
        :param gpu_memory:
        :param gpu_compute: float for debugging
        :param ensembles:
        :param threshs:
        :param w_config:
        :param lat_slo:
        """
        self.num_gpus = num_gpus
        self.gpu_memory = gpu_memory
        self.gpu_compute = gpu_compute # NOTE: This is only for debugging
        self.ensembles = ensembles
        self.threshs = threshs
        self.w_config = w_config
        self.lat_slo = lat_slo

        # NOTE: Delete, this is just to cache stuff when doing long runs
        self.done_until = -1

        # Get model memory and number of GPUs per unit
        with open(w_config.model_profile, 'r') as f:
            model_profile = json.load(f)
        self.model_memory_requ = {}
        for model in model_profile:
            self.model_memory_requ[model] = model_profile[model]["memory"]
        self.num_gpus_per_server = len(model_profile[model]["memory"])
        assert self.num_gpus_per_server <= self.num_gpus

        # Init load balancer.
        self.lb = LoadBalancer(self.num_gpus, self.num_gpus_per_server, self.gpu_compute)


    def get_prune_utility(self, replicas, replicas_gpu, gpu_avail_mem, comp_demand_per_model):
        """
        Compute the utility of pruning each replica (higher is better). 0 if the model cannot be pruned.
        :param replicas:
        :param replicas_gpu:
        :param gpu_avail_mem: with current placement, how much memory is avail on each gpu (can be negative)
        :param comp_demand_per_model:
        :return: list of len(replicas) with pruning utility of each replica.
        """
        utility = []
        for pruned_idx, (model, gpu) in enumerate(zip(replicas, replicas_gpu)):

            # Only consider first parts when pruning.
            if model[-1] != "0":
                utility.append(0)
                continue

            # Check if other GPUs have enough compute at every QPS for pruning to be possible.
            prunable = True
            max_gpu_util = 0
            for qps_comp_demand_per_model in comp_demand_per_model:
                prunable, gpu_util = self.lb.get_prunability_score(replicas,
                                                                   replicas_gpu,
                                                                   pruned_idx,
                                                                   qps_comp_demand_per_model)

                max_gpu_util = max(max_gpu_util, gpu_util)

            # Compute the utility of pruning the model.
            if prunable:
                mem_feasibility_score = 0
                model_base_name = model[:-1]
                for i, mem_saving in enumerate(self.model_memory_requ[model_base_name]):
                    gpu_idx = (gpu + i) % self.num_gpus
                    score = - mem_saving / gpu_avail_mem[gpu_idx]
                    score = max(0, min(1, score)) # 0 <= score <= 1
                    mem_feasibility_score += score

                # Intuition: Put little comp. pressure on busy GPUs while saving as much memory as possible.
                utility.append(mem_feasibility_score/max_gpu_util)
            else:
                utility.append(0)

        return utility


    def prune_model(self, replicas, replicas_gpu, gpu_memory_util, compute_per_model, lp_computational_demand):
        """
        Greedily prune models from GPUs until all replicas fit in the GPU's memory.
        :param replicas:
        :param replicas_gpu:
        :param gpu_memory_util:
        :param compute_per_model:
        :param lp_computational_demand:
        :return: bool if successful
        """
        while not np.all(gpu_memory_util <= self.gpu_memory):
            # 1. Get best model to prune (greedy).
            utility = self.get_prune_utility(replicas=replicas,
                                             replicas_gpu=replicas_gpu,
                                             gpu_avail_mem=-(gpu_memory_util - self.gpu_memory),
                                             comp_demand_per_model=lp_computational_demand)

            best_prune = np.argmax(utility)

            # Return: No model can be pruned
            if utility[best_prune] == 0:
                return False

            # 2. Delete pruned model parts from replica lists.
            assert replicas[best_prune][-1] == "0"
            pruned_model_base_name = replicas[best_prune][:-1]
            pruned_gpu = replicas_gpu[best_prune]

            for i in range(self.num_gpus_per_server):
                # Search for model parts to be deleted.
                gpu_idx = (pruned_gpu + i) % self.num_gpus
                part_name = f"{pruned_model_base_name}{i}"
                for j, (gr, r) in enumerate(zip(replicas_gpu, replicas)):
                    if gr == gpu_idx and r == part_name:
                        del replicas[j]
                        del replicas_gpu[j]

            # 3. Update GPU memory for all GPUs containing distributed parts of model.
            for i in range(self.num_gpus_per_server):
                gpu_idx = (pruned_gpu + i) % self.num_gpus
                gpu_memory_util[gpu_idx] -= self.model_memory_requ[pruned_model_base_name][i]

        return True


    def optimize(self, gear_plan, p_latency):
        # 1. Build union graph
        model_union = []
        for e in gear_plan.desc:
            for m in self.ensembles[e]:
                if m not in model_union:
                    model_union.append(m)

        logger.debug(f"(init) Model union: {model_union}")

        # 2. Replicate all model parts on all GPUs.
        # TODO: We cannot replicate multi-gpu models on each GPU because online in ray,
        #  we cannot have overlap between GPU units -- might have to change that here.
        all_model_parts = []
        for m in model_union:
            for i in range(self.num_gpus_per_server):
                all_model_parts.append(f"{m}{i}")

        replicas = []
        replicas_gpu = []
        for i in range(self.num_gpus):
            replicas += all_model_parts
            replicas_gpu += [i] * len(all_model_parts)

        # 3. Initialize memory: all models replicated on each gpu
        union_memory_requirement = 0
        for m in model_union:
            union_memory_requirement += np.sum(self.model_memory_requ[m])
        gpu_memory_used = [union_memory_requirement] * self.num_gpus

        logger.debug(f"(init) Memory used: {gpu_memory_used}")
        gpu_memory_used = np.array(gpu_memory_used)

        # 4. Initialize compute: all models replicated on each gpu
        model_comp_demand = [] # For each QPS, how many GPU-seconds does each model run
        sim = Simulation(pred_dir=self.w_config.pred_dir, profiling_file=self.w_config.model_profile, flop_thresh=1)
        for g_qps, g_ens in enumerate(gear_plan.desc):
            # Start at QPS = 1. TODO: Also do for qps 0? We didn't do it bc old load balancing had /0 errors with it.
            if g_qps == 0:
                continue

            # Get cost for each model.
            sim.set_cascade(self.ensembles[g_ens], self.threshs[g_ens])
            _, per_model_cost = sim.simulate(duplicate_cost=False, per_model=True)
            for m in per_model_cost.keys():
                per_model_cost[m] *= g_qps / 16 # TODO: 16 default batch size for bert

            # Done for this QPS. Return error if infeasible.
            model_comp_demand.append(per_model_cost)
            # TODO: Only commented out for now for acc SLO
            # TODO: Default cost and batch size etc. This needs to be looked at.
            if np.sum([float(c) for c in per_model_cost.values()]) > self.num_gpus*self.gpu_compute:
                return "latency_slo", g_qps

        # 5. Prune out models.
        success = self.prune_model(replicas=replicas,
                                   replicas_gpu=replicas_gpu,
                                   gpu_memory_util=gpu_memory_used,
                                   compute_per_model=model_comp_demand,
                                   lp_computational_demand=model_comp_demand)
        if not success:
            return "OOM", None

        # 6. Map GPUs to servers.
        # TODO: This works for our workloads but not generally.
        gpu_to_server = {}
        num_servers = math.ceil(self.num_gpus/self.num_gpus_per_server)
        gear_plan.gpus_on_server = [0] * num_servers
        for gpu_idx in range(self.num_gpus):
            server_idx = gpu_idx // self.num_gpus_per_server
            gear_plan.gpus_on_server[server_idx] += 1
            gpu_to_server[gpu_idx] = server_idx

        # 7. Fill out placement union in gear plan.
        model_placement = [set() for _ in gear_plan.gpus_on_server]
        for gpu, model in zip(replicas_gpu, replicas):
            server_idx = gpu_to_server[gpu]
            model_placement[server_idx].add(model[:-1])

        model_placement = [list(p) for p in model_placement]
        gear_plan.model_placement = model_placement

        # 8. Balance load and fill out in gear plan.
        self.lb.fill_in_gear_plan(gear_plan, replicas, replicas_gpu, model_comp_demand, gpu_to_server)

        # 9. Call batch size optimizer
        # TODO: Pass cost dict through
        # TODO: Pass down SLO
        bso = BatchSizeOptimizerManual(self.w_config)
        for qps, gear in enumerate(gear_plan.gears):
            gear = gear_plan.gears[qps]

            # TODO: All batch sizes = 1 for now.
            gear.server_batch_sizes = [[] for _ in gear_plan.model_placement]
            # for models_on_server in gear_plan.model_placement:
            #     gear.server_batch_sizes.append([16] + [1] * (len(models_on_server)-1))

            # TODO: Do for all servers
            ok = bso.optimize(gear, p_latency=p_latency, slo=self.lat_slo, gpu_id=0)

            if not ok:
                return "latency_slo", qps

            self.done_until = qps
            # gear.store(os.path.join(self.w_config.plan_dir, f"gear_{qps}.json"))

        # 10. Store gear plan.
        gear_plan.store(self.w_config.plan_dir)
        return "success", None