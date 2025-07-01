import time
import argparse
import os

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import torch

from workloads.twitter.twitter_models import *
from workloads.twitter.twitter_helpers import *
from offline.gear_plan import load_gear_plan
from utils.helpers import WorkloadTrace, WorkloadConfig
from utils.logger_setup import setup_logger
from online.frontend import Frontend

logger = setup_logger()


def run_workload(frontend, workload_config, prep, scaling_factor):
    # 1. Read in workload trace.
    # trace = WorkloadTrace(workload_config.workload_trace, scaling_factor) #workload_config.scaling_factor)
    # qps_history = trace.get_history(length=1200)
    qps_history = np.load("/home/gridsan/fkossmann/ensemble_serve/profiling/traces/twitter_qps.npy")
    qps_history *= int(scaling_factor)

    # 2. Read in tweets.
    num_queries = 16000
    tweets = get_tweets(num_queries)
    X, X_att = prep.prep(tweets)

    # 3. Warm up.
    warm_up_q = prep.prep(["This is a really positive tweet!"])
    frontend.warmup(warm_up_q)
    time.sleep(1)

    # Keep issueing interval fixed and vary batch size (batch size below != model batch size).
    rps = 20 # requests per second
    q_ctr = 0
    sleep_interval = 1/rps

    #qps_history = [1000]*3 + [3000, 3000, 1000, 1000]

    for qps in qps_history:
        # frontend.set_qps(qps)
        logger.debug(f"Issued QPS: {qps}")
        batch_size = int(qps/rps)

        for _ in range(rps):
            start = time.time()

            if q_ctr + batch_size >= num_queries:
                q_ctr = 0

            frontend.infer((X[q_ctr:q_ctr+batch_size], X_att[q_ctr:q_ctr+batch_size]))
            q_ctr += batch_size

            time.sleep(max(0, sleep_interval - (time.time() - start)))

    # 5. Done.
    logger.info("Workload runner done")
    time.sleep(1)


if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument("-g", "--gear-dir", type=str, required=True, help="The path to the gear directory")
    parser.add_argument("-l", "--log", type=str, required=True, help="Log prefix to producer, consumer logs")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-n", type=int, required=True, help="Log prefix to producer, consumer logs")

    parser.add_argument("-q", type=float, default=1.0, required=False, help="QPS measured multiplier")
    parser.add_argument("-s", type=float, default=1.0, required=False, help="Scale factor of workload")
    parser.add_argument("-b", type=int, default=0, required=False, help="Max batch size. If 0, infinite.")
    parser.add_argument("-ql", type=int, default=8, required=False, help="Max queue len before down scale")
    parser.add_argument("-p", type=float, default=0.1, required=False, help="Producer, QPS polling interval.")

    args = parser.parse_args()
    gear_dir = args.gear_dir
    log_path = args.log
    debug = args.debug
    num_servers = args.n
    scaling_factor = args.s
    qps_fac = args.q
    max_bs = args.b
    max_queue_len = args.ql
    qps_polling_interval = args.p

    # Check that GPUs available.
    num_visible_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert num_visible_gpus >= num_servers

    # Init ray.
    if debug:
        tmp_path = f"/state/partition1/user/{os.environ['USER']}/raytmp"
        os.makedirs(tmp_path, exist_ok=True)
        os.environ['TMPDIR'] = tmp_path # Cannot use default because too long on supercloud.

        ray.init(num_cpus=4, num_gpus=num_visible_gpus)
    else:
        ray.init(os.environ["ip_head"])

    logger.info(f"Resources: {ray.nodes()}")

    # Get workload config, gear plan and models.
    machine = "macbook"
    if torch.cuda.device_count() > 0:
        machine = "supercloud"

    models, prep = get_model_dict(machine)
    workload_config = WorkloadConfig("twitter")
    gear_plan = load_gear_plan(gear_dir)

    # Init frontend.
    frontend = Frontend(model_dict=models,
                        gear_plan=gear_plan,
                        log_prefix=log_path,
                        num_gpus=num_servers,
                        qps_fac=qps_fac,
                        max_bs=max_bs,
                        max_queue_len=max_queue_len,
                        qps_polling_interval=qps_polling_interval)

    # Run workload.
    logger.info("Workload runner start")
    run_workload(frontend, workload_config, prep, scaling_factor=scaling_factor)
    frontend.shutdown()
