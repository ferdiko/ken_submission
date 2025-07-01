import os
import random

import numpy as np

from simulator.simulator import Simulation

"""
Instead of RL, just use random sampling to find the cascades.
"""
class Sampler:

    def __init__(self, pred_dir, profiling_file):
        self.pred_dir = pred_dir
        self.profiling_file = profiling_file

        # read in model names
        self.models = []
        for file in os.listdir(pred_dir):
            # check if sim file
            if file[-4:] != ".csv":
                continue
            if file[:7] == "ignore_":
                continue

            self.models.append(file[:-4])

    def search(self, num_samples=10000, max_len=3, min_thresh=0, max_thresh=0.5, num_threshs=20):
        # chosable thresholds
        threshs = np.linspace(min_thresh, max_thresh, num_threshs)

        # init simulator
        sim = Simulation(pred_dir=self.pred_dir,
                         profiling_file=self.profiling_file)

        # sample
        all_sampled_models = []
        all_sampled_threshs = []
        all_sampled_accs = []
        all_sampled_costs = []
        for prog in range(num_samples):
            # progress indicator
            if prog % 1000 == 0:
                print(f"Sampling progress: {int(prog/num_samples*100)}%")

            # sample a cascade
            cascade_len = random.randint(2, max_len)
            sampled_models = random.sample(self.models, cascade_len) # without replacement
            sampled_threshs = np.random.choice(threshs, cascade_len-1) # with replacement
            sampled_threshs = [0.5] + list(sampled_threshs)

            # simulate
            sim.models = sampled_models
            sim.threshs = sampled_threshs
            acc, cost = sim.simulate()

            # append
            all_sampled_models.append(sampled_models)
            all_sampled_threshs.append(sampled_threshs)
            all_sampled_accs.append(acc)
            all_sampled_costs.append(cost)

        return None, all_sampled_accs, all_sampled_costs, all_sampled_models, all_sampled_threshs


    def hand_crafted(self):

        models = ["tiny", "mini", "small", "medium", "base"]
        thresh = [0.5]

        sim = Simulation(pred_dir=self.pred_dir,
                         profiling_file=self.profiling_file)

        all_sampled_models = []
        all_sampled_threshs = []
        all_sampled_accs = []
        all_sampled_costs = []

        for m in models:
            # simulate
            sim.models = [m]
            sim.threshs = thresh
            acc, cost = sim.simulate()

            all_sampled_models.append([m])
            all_sampled_threshs.append(thresh)
            all_sampled_accs.append(acc)
            all_sampled_costs.append(cost)

        # # init simulator
        # sim = Simulation(pred_dir=self.pred_dir,
        #                  profiling_file=self.profiling_file)
        #
        # sampled_models = ["llama_13b", "llama_70b"]
        #
        # # sample
        # all_sampled_models = []
        # all_sampled_threshs = []
        # all_sampled_accs = []
        # all_sampled_costs = []
        #
        # threshs = np.linspace(0.0, 0.21, 100)
        # for t in threshs:
        #     sampled_threshs = [0.5, t]
        #
        #     # simulate
        #     sim.models = sampled_models
        #     sim.threshs = sampled_threshs
        #     acc, cost = sim.simulate()
        #
        #     # append
        #     all_sampled_models.append(sampled_models)
        #     all_sampled_threshs.append(sampled_threshs)
        #     all_sampled_accs.append(acc)
        #     all_sampled_costs.append(cost)

        return None, all_sampled_accs, all_sampled_costs, all_sampled_models, all_sampled_threshs
