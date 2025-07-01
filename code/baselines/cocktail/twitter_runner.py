import time
import argparse
import os
import math

from baselines.cocktail.frontend import CocktailFrontend

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import torch

from workloads.twitter.twitter_models import *
from workloads.twitter.twitter_helpers import *
from offline.gear_plan import load_gear_plan
from utils.helpers import WorkloadTrace, WorkloadConfig
from utils.logger_setup import setup_logger

logger = setup_logger()


def get_workload_forecast(counter, planning_interval, qps_history):
    relevant_history = qps_history[counter+1:counter+planning_interval]
    relevant_history = np.array(relevant_history)
    if relevant_history.shape[0] == 0:
        return 1000 # Doesn't matter since this won't take effect before workload over

    # qps_len = relevant_history.shape[0]
    # sorted_history = np.sort(relevant_history)
    # return sorted_history[int(0.9*qps_len)]
    return np.max(relevant_history)


def run_workload(frontend, workload_config, prep, scaling_factor, planning_interval, packing_factor):
    # 1. Read in workload trace.
    # trace = WorkloadTrace(workload_config.workload_trace, scaling_factor) #workload_config.scaling_factor)
    # qps_history = trace.get_history(length=1200)
    qps_history = np.load("/home/gridsan/fkossmann/ensemble_serve/profiling/traces/twitter_qps.npy")
    qps_history *= int(scaling_factor)

    # 2. Read in tweets.
    num_queries = 16000
    tweets = get_tweets(num_queries)
    X, X_att = prep.prep(tweets)

    # 3. Bootstrap: Define initial provisioning.
    predicted_workload = get_workload_forecast(0, planning_interval, qps_history)
    replicas = math.ceil(predicted_workload/packing_factor)
    frontend.bootstrap(["base"]*replicas)
    time.sleep(1)

    # Keep issueing interval fixed and vary batch size (batch size below != model batch size).
    rps = 20 # requests per second
    q_ctr = 0
    sleep_interval = 1/rps

    counter = 0

    for qps in qps_history:
        logger.debug(f"Issued QPS: {qps}")
        batch_size = int(qps/rps)

        # Issue QPS.
        for _ in range(rps):
            start = time.time()

            if q_ctr + batch_size >= num_queries:
                q_ctr = 0

            frontend.infer((X[q_ctr:q_ctr+batch_size], X_att[q_ctr:q_ctr+batch_size]))
            q_ctr += batch_size

            time.sleep(max(0.0, sleep_interval - (time.time() - start)))

        # Trigger replanning from runner: Since we give ground truth workload forecast,
        # this needs to be synchronized with queries being issued.
        counter += 1
        if counter % planning_interval == 0:
            frontend.replan(get_workload_forecast(counter, planning_interval, qps_history))

    # 5. Done.
    logger.info("Workload runner done")
    time.sleep(1)
    frontend.shutdown()


if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument("-l", "--log", type=str, required=True, help="Log prefix to producer, consumer logs")
    parser.add_argument("--debug", action="store_true", help="Execute in debug mode (interactive slurm job)")

    # For debugging.
    parser.add_argument("-n", type=int, required=True, help="Maximum number of GPU servers")
    parser.add_argument("-pi", type=int, default=60, required=False, help="Planning interval in seconds")
    parser.add_argument("-pf", type=float, required=True, help="Packing factor")
    parser.add_argument("-s", type=float, default=1.0, required=False, help="Scale factor of workload")
    parser.add_argument("-b", type=int, default=0, required=False, help="Max batch size. If 0, infinite.")


    args = parser.parse_args()
    log_path = args.log
    debug = args.debug
    num_servers = args.n
    scaling_factor = args.s
    max_bs = args.b
    planning_interval = args.pi
    packing_factor = args.pf


    # Check that GPUs available.
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0

    # Init ray.
    if debug:
        tmp_path = f"/state/partition1/user/{os.environ['USER']}/raytmp"
        os.makedirs(tmp_path, exist_ok=True)
        os.environ['TMPDIR'] = tmp_path # Cannot use default because too long on supercloud.

        print("ray init")
        ray.init(num_cpus=4, num_gpus=num_gpus)
        print("ray init done")
    else:
        ray.init(os.environ["ip_head"])

    logger.info(f"Resources: {ray.nodes()}")

    # Get workload config, gear plan and models.
    machine = "macbook"
    if torch.cuda.device_count() > 0:
        machine = "supercloud"

    models, prep = get_model_dict(machine)
    workload_config = WorkloadConfig("twitter")

    # Generate warm up query that Cocktail can use to warm its instances.
    warm_up_q = prep.prep(["This is a really positive tweet!"])

    # Init frontend.
    frontend = CocktailFrontend(model_dict=models,
                                log_prefix=log_path,
                                max_gpus=num_servers,
                                max_bs=max_bs,
                                packing_factor=packing_factor,
                                warmup_sample=warm_up_q)

    # Run workload.
    logger.info("Workload runner start")
    run_workload(frontend, workload_config, prep, planning_interval=planning_interval, scaling_factor=scaling_factor, packing_factor=packing_factor)
    frontend.shutdown()
