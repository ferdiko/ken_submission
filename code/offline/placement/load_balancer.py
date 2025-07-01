import numpy as np
from scipy.optimize import linprog

from utils.logger_setup import setup_logger


logger = setup_logger()

class LoadBalancer:

    def __init__(self, num_gpus, num_gpus_per_unit, gpu_compute):
        self.num_gpus = num_gpus
        self.num_gpus_per_unit = num_gpus_per_unit
        self.gpu_compute = gpu_compute


    def check_if_prunable(self, replicas, gpu_replicas, pruned_idx, qps_computational_demand, max_util=1.0):
        """
        Linear program for load balancing:
         - Decision variables: How much work assigned to each replica
         - Constraints:
           - Each GPU < self.gpu_capacity
           - Each model >= work imposed on it
         - Optimization objective:
           - Currently just minimize total work imposed so we don't overasign work ...

        :param replicas:
        :param gpu_replicas:
        :param pruned_idx:
        :param qps_computational_demand:
        :param max_util: Maximum GPU utilization allowed
        :return:
        """
        assert replicas[pruned_idx][-1] == "0"
        pruned_model_base_name = replicas[pruned_idx][:-1]
        pruned_gpu = gpu_replicas[pruned_idx]
        all_models = list(qps_computational_demand.keys())
        num_models = len(all_models)

        # 1. Create candidate replicas where the pruned model parts are already deleted.
        cand_replicas = []
        cand_gpu_replicas = []
        pruned_parts = []
        for i in range(self.num_gpus_per_unit):
            gpu_idx = (pruned_gpu + i) % self.num_gpus
            model_part_name = f"{pruned_model_base_name}{i}"
            pruned_parts.append((gpu_idx, model_part_name))

        for gr, r in zip(gpu_replicas, replicas):
            if (gr,r) not in pruned_parts:
                cand_replicas.append(r)
                cand_gpu_replicas.append(gr)

        cand_replicas = np.array(cand_replicas)
        cand_gpu_replicas = np.array(cand_gpu_replicas)

        # 2. Initialize linear program tensors.
        # c are optimization weights: How much work assigned to each replica (just simply sum total work).
        c = np.ones(cand_replicas.shape)

        # b are upper bounds.
        b = np.zeros(self.num_gpus + num_models)

        # A is inequality weighting matrix.
        A = np.zeros((b.shape[0], c.shape[0]), dtype=int)

        # 3. Encode inequalities.
        # Sum of work over replicas on same GPU <= self.gpu_capacity
        for gpu_idx in range(self.num_gpus):
            A[gpu_idx, :] = cand_gpu_replicas == gpu_idx
            b[gpu_idx] = max_util * self.gpu_compute

        # Sum of work over replicas of same model on different GPUs >= work required to serve repliated model.
        for i, m in enumerate(all_models):
            A[self.num_gpus+i, :] -= cand_replicas == m
            b[self.num_gpus+i] = -qps_computational_demand[m]

        # 4. Solve the linear program and return if solution found.
        res = linprog(c, A_ub=A, b_ub=b, method="highs")
        return res.success, res.x


    def get_prunability_score(self, replicas, gpu_replicas, pruned_idx, qps_computational_demand, min_probe=0.5, num_probes=5):
        """
        Returns a bool if the model is prunable at all, and the GPU utilization (lower is better).
        :param replicas:
        :param gpu_replicas:
        :param pruned_idx:
        :param qps_computational_demand:
        :param min_probe:
        :param num_probes:
        :return: is_prunable (bool), min achievable GPU utilization
        """
        probes = np.linspace(1.0, min_probe, num_probes)
        for i in range(num_probes):
            prunable, _ = self.check_if_prunable(replicas,
                                                 gpu_replicas,
                                                 pruned_idx,
                                                 qps_computational_demand,
                                                 max_util=probes[i])

            if not prunable:
                return i > 0, probes[i]

        return True, min_probe


    def get_load_distribution(self, replicas, gpu_replicas, qps_computational_demand, min_util=0.2, num_probes=18):
        all_models = list(qps_computational_demand.keys())
        num_models = len(all_models)
        np_replicas = np.array(replicas)
        np_gpu_replicas = np.array(gpu_replicas)

        # Search for lowest util load assignment.
        load_assignment = None
        max_util_probes = np.linspace(1.0, min_util, num_probes)
        for max_util in max_util_probes:
            # 1.. Initialize linear program tensors.
            # c are optimization weights: How much work assigned to each replica (just simply sum total work).
            c = np.ones(np_replicas.shape)

            # b are upper bounds.
            b = np.zeros(self.num_gpus + num_models)

            # A is inequality weighting matrix.
            A = np.zeros((b.shape[0], c.shape[0]), dtype=int)

            # 2.. Encode inequalities.
            # Sum of work over replicas on same GPU <= self.gpu_capacity
            for gpu_idx in range(self.num_gpus):
                A[gpu_idx, :] = np_gpu_replicas == gpu_idx
                b[gpu_idx] = max_util * self.gpu_compute

            # Sum of work over replicas of same model on different GPUs >= work required to serve repliated model.
            for i, m in enumerate(all_models):
                A[self.num_gpus+i, :] -= np_replicas == m
                b[self.num_gpus+i] = -qps_computational_demand[m]

            # 4. Solve the linear program.
            res = linprog(c, A_ub=A, b_ub=b, method="highs")
            if res.success:
                load_assignment = res.x
            else:
                if max_util <= 0.5:
                    logger.critical(f"Load balancer only failed at GPU utilization of {max_util}")
                return load_assignment

        logger.critical(f"Load balancer found assignment with GPU utilization of {min_util} (even lower might be possible)")
        return load_assignment


    def fill_in_gear_plan(self, gear_plan, replicas, gpu_replicas, computational_demand, gpu_to_server, min_probe=0.2, num_probes=16):
        """
        Assign workload to each replica at each QPS. Fill into gear plan.
        :param gear_plan:
        :param replicas:
        :param gpu_replicas:
        :param computational_demand:
        :param gpu_to_server:
        :param min_probe:
        :param num_probes:
        :return:
        """
        for qps, qps_computational_demand in enumerate(computational_demand):
            # 1. Get load assignment
            assignment = self.get_load_distribution(replicas,
                                                    gpu_replicas,
                                                    qps_computational_demand,
                                                    min_probe,
                                                    num_probes)

            # 2. Format to dict: Model -> list with fration of workload assigned to each GPU.
            qps_models = list(qps_computational_demand.keys())
            num_servers = len(set(gpu_to_server.values()))
            dict_assignment = {}
            for m in qps_models:
                # Get work done by each GPU.
                gpu_work = np.zeros(num_servers)
                model_base_name = m[:-1]
                for i, (r, gr) in enumerate(zip(replicas, gpu_replicas)):
                    server_idx = gpu_to_server[gr]
                    r_base_name = r[:-1]

                    if r_base_name == model_base_name:
                        gpu_work[server_idx] += assignment[i]

                # Normalize and store in dict.
                total = np.sum(gpu_work)
                if total > 0:
                    gpu_work /= total

                dict_assignment[model_base_name] = list(gpu_work)

            # 3. Store in gear
            gear_plan.gears[qps+1].load_assignment = dict_assignment
