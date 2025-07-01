import json
import torch
import numpy as np
import os
from collections import defaultdict

# from operators import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def one_hot(dim, idx):
    if not isinstance(idx, list):
        idx = [idx]

    v = np.zeros(dim)
    for d in idx:
        v[d] = 1
    return v

"""
Parse files containing simulated values
"""
def parse_sim_files(sim_files):
    all_vals = []

    for filename in sim_files:
        with open(filename, "r") as f:
            lines = f.readlines()

        vals = []
        for l in lines:
            vals.append(float(l.split(",")[0]))

        all_vals.append(vals)

    # iterate through first file and add ground truth label
    with open(filename, "r") as f:
        lines = f.readlines()

    vals = []
    for l in lines:
        vals.append(float(l.split(",")[1]))

    all_vals.append(vals)

    return tuple(all_vals)


"""
Parse all sim files in the directory.
"""
def parse_sim_dir(dir_path):
    # returned dicts
    cert_dict = {}
    pred_dict = {}
    labels = []

    # go through all sim files
    for file in os.listdir(dir_path):
        # check if sim file
        if file[-4:] != ".csv":
            continue
        if file[:7] == "ignore_":
            continue

        # read in certs and preds
        # NOTE: sim file format has to be csv with pred,label,cert[,...]
        model_name = file[:-4]
        with open(os.path.join(dir_path, file), "r") as f:
            lines = f.readlines() #[:500]
        pred_dict[model_name] = [int(l.split(",")[0]) for l in lines]
        cert_dict[model_name] = [float(l.split(",")[2]) for l in lines]

        # read in / check labels
        if len(labels) == 0:
            labels = [int(l.split(",")[1]) for l in lines]
        else:
            comp_label = [int(l.split(",")[1]) for l in lines]
            assert labels == comp_label, "labels in sim files don't match"

    return pred_dict, cert_dict, labels


class Simulation:

    def __init__(self, pred_dir, profiling_file, flop_thresh=1, alpha=1):
        # Init simulation state,
        self.cost = 0
        self.acc = 0

        # Init ensemble,
        self.models = []
        self.threshs = [0.5,]

        # Parse sim files.
        self.preds, self.certs, self.labels = parse_sim_dir(pred_dir)
        self.num_models = len(self.preds)
        self.num_samples = len(list(self.preds.values())[0])

        # Read in cost.
        with open(profiling_file, "r") as f:
            cost_data = json.load(f)

        self.cost_dict = {}
        for model in self.preds.keys():
            self.cost_dict[model] = np.sum([float(c) for c in cost_data[model]["runtime"]["default"]])

        self.batch_size_profiles = {}
        for model in self.preds.keys():
            self.batch_size_profiles[model] = {}
            for k_bs in cost_data[model]["runtime"].keys():
                if k_bs != "default":
                    self.batch_size_profiles[model][int(k_bs)] = float(cost_data[model]["runtime"][k_bs])

        # Training hyper params
        self.flop_thresh = flop_thresh
        self.histo_threshs = np.linspace(0, 0.3, 4)  # TODO: Need to pass that in as hyperparam. and the 0, 0.3 are bullshit
        self.alpha = alpha # mult. factor weight of flops

        # Latency simulator.
        self.queues = []
        self.added_to_queue_ctr = []
        self.entry_time = {}
        self.latency = []


    def get_sample_counts(self):
        _, cost = self.simulate(per_model=True)

        for m in cost:
            cost[m] *= self.num_samples / self.cost_dict[m[:-1]][0]

        # for i, m in enumerate(self.models):
        #     cost[i] /= self.cost_dict[m][0]

        return cost

    def get_model_pred_fractions(self):
        """
        For each model in theWhat fraction of the samples are predicted
        :return:
        """
        _, cost = self.simulate(per_model=True)
        for m in cost:
            cost[m] /= self.cost_dict[m]

        return cost



    def reset(self):
        """
        Reset simulation state and cascade
        :return: torch tensor; Inital state vector
        """
        self.cost = 0
        self.acc = 0
        self.models = []
        self.threshs = [0.5,]
        self.queues = []
        self.added_to_queue_ctr = []
        return self.route_state_vector()


    def set_cascade(self, models, threshs):
        self.models = models
        self.threshs = threshs


    def write_out_predictions(self, history, gear_plan, out_file):
        query_counter = 0
        total_counter = 0
        if out_file is not None:
            f = open(out_file, "w+")
        corr = 0
        incorr = 0
        for qps in history:
            gear = gear_plan.gears[qps]

            for _ in range(qps):
                cert = 0
                pred = -1

                # Cascade.
                # sim.models = g.models
                # sim.threshs = g.threshs

                for thresh, model in zip(gear.threshs, gear.models):
                    if model == "noop":
                        break
                    elif cert < thresh:
                        pred = self.preds[model][query_counter]
                        cert = self.certs[model][query_counter]
                        if out_file is not None:
                            f.write(f"      decide: {cert} {thresh}\n")
                            f.write(f"      qps: {qps} id: {total_counter} model: {model} cert: {cert} {gear.threshs}, {gear.models}\n")
                    else:
                        break

                if pred == self.labels[query_counter]:
                    corr += 1
                else:
                    incorr += 1
                if out_file is not None:
                    f.write(f"{pred},{self.labels[query_counter]}\n")
                query_counter = (query_counter+1)%len(self.preds[model])
                total_counter += 1

        if out_file is not None:
            f.close()
        print("="*10, f"Accuracy: {corr/(corr+incorr)}", "="*10)


    def simulate(self, duplicate_cost=True, per_model=False, casc1=True):
        """
        Simulate ensemble
        :param duplicate_cost: Count cost twice same model called twice?
        :param per_model: Return cost per model?
        :return:
        """
        # Init cost and acc vars.
        corr = 0
        incorr = 0
        if per_model:
            cost = defaultdict(int)
            cost_dict = self.cost_dict
        else:
            cost = 0
            cost_dict = {}
            # for m in self.cost_dict:
            #     cost_dict[m] = np.sum(self.cost_dict[m])

            # cost_dict["llama_03b"] /= 2
            # if casc1:
            #     cost_dict["llama_13b"] /= 2
            # else:
            #     cost_dict["llama_70b"] /= 2

        # Simulate prediction on all samples.
        for i in range(self.num_samples):
            cert = 0
            pred = -1
            models_used = []

            # Cascade.
            for thresh, model in zip(self.threshs, self.models):
                if model == "noop":
                    break
                elif cert < thresh:
                    pred = self.preds[model][i]
                    cert = self.certs[model][i]
                    if duplicate_cost or not model in models_used:
                        if per_model:
                            cost[model] += cost_dict[model]
                        else:
                            cost += cost_dict[model]
                            # cost += self.cost_dict[model][0] # TODO: tmp

                        models_used.append(model)
                else:
                    break

            # Check prediction vs label.
            if pred == self.labels[i]:
                corr += 1
            else:
                incorr += 1

        # Normalize cost.
        if per_model:
            for c in cost:
                cost[c] /= self.num_samples
        else:
            cost /= self.num_samples

        return corr/(corr+incorr), cost

    # ============================================================================
    # Below methods are for latency simulator.
    # ============================================================================

    def predict(self, server_id, queue_id, samples):
        # If last model, no samples are cascaded further.
        if queue_id >= len(self.threshs) - 1:
            return []

        thresh = self.threshs[queue_id + 1]
        model = self.models[queue_id]
        cascaded = []
        for s in samples:
            if self.certs[model][s] < thresh:
                cascaded.append(s)

        return cascaded


    def infer(self, server_id, queue_id, batch_sizes, cur_time, clear_queue=False):
        # Get trigger for inference.
        trigger = batch_sizes[server_id][queue_id]

        if clear_queue:
            trigger = 1

        if self.added_to_queue_ctr[server_id][queue_id] >= trigger:
            cur_time[server_id] += self.batch_size_profiles[self.models[queue_id]][len(self.queues[server_id][queue_id])]

            # Get which samples are not cascaded further.
            to_predict = self.queues[server_id][queue_id]
            cascaded = self.predict(server_id, queue_id, to_predict)
            for x in to_predict:
                if x not in cascaded:
                    self.latency.append(cur_time[server_id] - self.entry_time[x])

            # Add cascaded samples to next queue.
            self.queues[server_id][queue_id] = []
            self.added_to_queue_ctr[server_id][queue_id] = 0

            if queue_id == len(self.queues[server_id]) - 1:
                assert cascaded == []
            elif len(cascaded) > 0:
                self.queues[server_id][queue_id + 1] += cascaded
                self.added_to_queue_ctr[server_id][queue_id + 1] += 1

        return cur_time


    def get_latency(self, gear, qps, p_latency=0.95, num_secs=10):
        """
        This is the new way to do it, after this works, delete all others.
        Error: If a queue overflows, return error and first index of overflowing queue
        :param gear_plan:
        :return:
        """
        # NOTE: Maybe just eliminate the 0 QPS throughout the code base
        if qps == 0:
            return 0

        # gear = gear_plan.gears[qps]
        self.models = gear.models
        self.threshs = gear.threshs
        server_batch_sizes = gear.server_batch_sizes
        num_servers = len(server_batch_sizes)

        # Simulate quques: These lists will hold which samples are inside them.
        for batch_sizes in server_batch_sizes:
            assert len(batch_sizes) == len(self.models), \
                "We're currently assuming each model is on each server, but not true" # TODO
            self.queues.append([[] for _ in batch_sizes])
            self.added_to_queue_ctr.append([0 for _ in batch_sizes])

        arrivals = np.linspace(0, num_secs, qps*num_secs+1)

        # Simulate.
        cur_time = [0.0]*num_servers
        for i in range(qps*num_secs):
            self.entry_time[i] = arrivals[i]

            # Put sample into queue.
            # TODO: Do actual load balancing. For now we assume each model on every server and just do round-robin.
            server_id = i % num_servers
            self.queues[server_id][0].append(i)
            self.added_to_queue_ctr[server_id][0] += 1

            cur_time[server_id] = max(cur_time[server_id], arrivals[i])

            # Check if a queue is full and we should trigger inference on all queues.
            for server_id in range(num_servers):
                for queue_id in range(len(self.queues[server_id])):
                    cur_time = self.infer(server_id, queue_id, server_batch_sizes, cur_time)

        # Get p latency.
        sorted_lat = sorted(self.latency)

        # np.save("tmp_meeting_sim_fig", sorted_lat)
        return sorted_lat[int(p_latency*len(sorted_lat))]


    # ============================================================================
    # Below methods are legacy but still used in RL.
    # ============================================================================

    """
    histogram on effect of thresholds on overall performance
    """
    def thresh_histogram(self):
        histogram = []
        for t in self.histo_threshs:
            self.threshs.append(t)
            # print(self.threshs)

            acc, cost = self.simulate()
            histogram.append(acc)
            histogram.append(cost/self.flop_thresh)

            self.threshs = self.threshs[:-1]

        # print("HISTO", histogram)

        return np.array(histogram)


    """
    Add model to cascade. this will return input vector for thresh agent
    """
    def add_model(self, model):
        self.models.append(model-1)
        done = len(self.models) >= self.num_models or self.models[-1] == -1

        if done:
            thresh_input = None
        else:
            thresh_input = self.thresh_state_vector()

        # print("Models:", self.models)
        # print("threshs:", self.threshs)

        return thresh_input, done


    """
    Add threshold
    """
    def add_thresh(self, thresh):
        thresh = self.histo_threshs[thresh]
        self.threshs.append(thresh)


    """
    input vector to routing model
    """
    def route_state_vector(self):
        one_hot_prev = one_hot(self.num_models, self.models)
        vec = np.concatenate((one_hot_prev, [self.cost/self.flop_thresh, self.acc]))
        vec = np.array(vec)
        return torch.from_numpy(vec).float()


    """
    input vector to thresh model
    """
    def thresh_state_vector(self):
        one_hot_prev = one_hot(self.num_models, self.models[:-1])
        one_hot_new = one_hot(self.num_models, self.models[-1])
        histogram = self.thresh_histogram()
        vec = np.concatenate((one_hot_prev, one_hot_new, histogram))
        vec = np.array(vec)
        return torch.from_numpy(vec).float()


    """
    sum acc and cost diff with weighting factors
    """
    def reward_add(self, new_acc, new_cost):
        reward = (new_acc - self.acc)
        reward -= self.alpha*(new_cost - self.cost)/self.flop_thresh
        reward = max(-0.1, reward)
        print((new_acc - self.acc), self.alpha*(new_cost - self.cost)/self.flop_thresh)

        return reward


    """
    reward only based on acc target
    """
    def reward_sigmoid(self, new_acc, new_cost):

        reward = sigmoid((new_acc - 0.85)*100) - 0.5
        reward -= self.alpha*sigmoid((new_cost - self.flop_thresh)/self.flop_thresh*2) - 0.5
        # reward /= 10
        # print(sigmoid((new_acc - 0.85)*100) - 0.5, self.alpha*sigmoid((new_cost - self.flop_thresh)/self.flop_thresh*2) - 0.5)
        return reward


    """
    reward only based on acc target
    """
    def reward_sigmoid2(self, new_acc, new_cost):
        # print("ACC", new_acc, self.acc)
        reward = sigmoid((new_acc - self.acc)*100) - 0.5
        reward -= self.alpha*sigmoid((new_cost - self.flop_thresh)/self.flop_thresh*2) - 0.5
        # reward /= 10
        # print(sigmoid((new_acc - self.acc)*100) - 0.5, sigmoid((new_cost - self.flop_thresh)/self.flop_thresh*2) - 0.5)
        return reward

    """
    reward based on flop and acc target, absolute distance
    """
    def reward3(self, new_acc, new_cost):
        reward = (new_acc - self.acc)
        reward -= self.alpha*(new_cost - self.cost)/self.flop_thresh
        reward = max(-0.1, reward)
        return reward

    """
    reward based on flop and acc target, quadratic distance
    """
    def reward4(self, new_acc, new_cost):
        reward = (new_acc - self.acc)
        reward -= self.alpha*(new_cost - self.cost)/self.flop_thresh
        reward = max(-0.1, reward)
        print((new_acc - self.acc), self.alpha*(new_cost - self.cost)/self.flop_thresh)
        return reward


    """
    Compute next step based on action chosen by RL agent
    """
    def step(self):
        # check if done
        done = len(self.models) >= self.num_models or self.models[-1] == -1

        # compute reward
        new_acc, new_cost = self.simulate()

        if done:
            reward = 0 #self.reward_sigmoid(new_acc, new_cost)
        else:
            reward = self.reward_sigmoid2(new_acc, new_cost) #self.reward_add(new_acc, new_cost) #self.reward_sigmoid2(new_acc, new_cost)

        # get state vector
        self.acc = new_acc
        self.cost = new_cost
        route_state_vec = self.route_state_vector()

        # additional info for debugging
        debug_info = {}

        return route_state_vec, reward, done, debug_info


    def print_ensemble(self):
        print("MODELS:", self.models)
        print("THRESH:", self.threshs)

    def print_goal(self):
        print(self.cost)
        print(f"ACC:{self.acc}       FLOPS:{self.cost/self.flop_thresh}")

    def get_goal(self, duplicate_cost=False):
        acc, cost = self.simulate(duplicate_cost=duplicate_cost)
        # executed = []
        # for m in self.models:
        #     if m != -1 and not m in executed:
        #         cost += self.flop_dict[m]*
        #         executed.append(m)

        return acc, cost #self.cost/self.flop_thresh


if __name__ == "__main__":
    from offline.gear_plan import load_gear_plan
    from utils.helpers import WorkloadTrace, WorkloadConfig

    test_gear_plan = False
    wconfig = WorkloadConfig("twitter")
    sim = Simulation(pred_dir=wconfig.pred_dir, profiling_file=wconfig.model_profile)

    # if test_gear_plan:
    #
    #     for i in range(100):
    #         gear_plan = load_gear_plan(wconfig.plan_dir)
    #         trace = WorkloadTrace(wconfig.workload_trace, wconfig.scaling_factor, seed=2)
    #         history = trace.get_history(1200)
    #         print(np.max(history))
    #
    #         try:
    #             print(i)
    #             sim.write_out_predictions(history, gear_plan, None)
    #         except:
    #             continue
    #
    # else:
    #     print(wconfig.pred_dir)
    #     # sim.set_cascade(["tiny", "base"], [0.5, 0.66])
    #     # sim.set_cascade(["medium", "base"], [0.5, 0.52])
    #
    #     sim.set_cascade(["small", "medium", "base"], [0.5, 0.4, 0.4])
    #
    #     print(sim.get_sample_counts())

    plan = load_gear_plan(wconfig.plan_dir)
    sim.set_cascade(["tiny", "base"], [0.5, 0.4])
    print(sim.get_latency(plan, 5))








