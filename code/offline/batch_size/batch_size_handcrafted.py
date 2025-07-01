import json
import math

from utils.logger_setup import setup_logger
from simulator.simulator import Simulation


logger = setup_logger()

class BatchSizeOptimizerManual:
    """
    Idea behind this batch size optimizer:
    There's one original batch size BS. if a model gets x% of the samples, its batch size is x% of BS.
    Iterate through all BS until throughput requirement is met -- then check for latency requirement.
    """

    def __init__(self, paths):
        self.paths = paths

        # Read in model profile.
        with open(paths.model_profile, "r") as f:
            profile_json = json.load(f)

        # Parse out runtimes for different batch sizes.
        self.profile = {}
        for k_model in profile_json.keys():
            self.profile[k_model] = {}
            for k_bs in profile_json[k_model]["runtime"].keys():
                if k_bs != "default":
                    self.profile[k_model][int(k_bs)] = profile_json[k_model]["runtime"][k_bs]

    def optimize(self, gear, p_latency, slo, gpu_id):
        total_qps = gear.qps

        # Run simulator to get fraction of samples forwarded to each model.
        sim = Simulation(pred_dir=self.paths.pred_dir, profiling_file=self.paths.model_profile)
        sim.set_cascade(gear.models, gear.threshs)
        sample_frac = sim.get_model_pred_fractions()

        # Determine first batch size that can handle throughput.
        for first_bs in range(1, max(self.profile[gear.models[0]]) + 1):
            batch_sizes = [first_bs] + [1]*(len(gear.models)-1)

            # Skip if cannot match throughput.
            sum = 0
            for m, bs in zip(gear.models, batch_sizes):
                sum += self.profile[m][bs] * math.ceil(total_qps/first_bs)

            if sum > 1:
                continue

            # Check if batch sizes fulfill latency SLO.
            gear.server_batch_sizes[gpu_id] = batch_sizes
            lat = sim.get_latency(gear, gear.qps, p_latency=p_latency)

            if lat <= slo:
                gear.server_batch_sizes[gpu_id] = batch_sizes
                return True

        # If it doesn't work: look into this, it should work for our SF
        logger.debug(f"Couldn't find batch size: {total_qps} qps, models: {gear.models}, fwd fractions: {sample_frac}")
        return False