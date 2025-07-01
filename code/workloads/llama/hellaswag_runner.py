import time
import argparse
import os

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import torch

from workloads.llama.llama_model import *
from offline.gear_plan import load_gear_plan
from utils.helpers import WorkloadTrace, WorkloadConfig
from utils.logger_setup import setup_logger
from workloads.llama.llama_helpers import get_hellaswag
from online.frontend import Frontend

logger = setup_logger()


def run_workload(frontend, workload_config):
    # 1. Read in workload trace.
    trace = WorkloadTrace(workload_config.workload_trace, workload_config.scaling_factor)
    qps_history = trace.get_history(length=1200)

    # 2. Read in prompts.
    queries, _ = get_hellaswag(500)
    num_queries = len(queries)

    # 3. Warm up.
    warmup = 8
    for _ in range(warmup):
        warm_up_q = ("The question is ", ["live or not to live.", "a good one.", "wet rainy weather.", "option 4."])
        frontend.infer(warm_up_q)
    time.sleep(20)

    # 4. Run workload.
    q_ctr = 0
    for qps in qps_history:
        logger.debug(f"Issued QPS: {qps}")
        if qps > 0:
            sleep_interval = 1/qps # NOTE: - overhead, but cannot be negative

            for _ in range(qps):
                query = queries[q_ctr % num_queries]
                frontend.infer(query)
                q_ctr += 1

                time.sleep(sleep_interval)

        else:
            time.sleep(1)

    # 5. Done.
    logger.info("Workload runner done")
    time.sleep(10)


if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument("-g", "--gear-dir", type=str, required=True, help="The path to the gear directory")
    parser.add_argument("-l", "--log", type=str, required=True, help="Log prefix to producer, consumer logs")
    parser.add_argument("-n", "--num-nodes", type=str, required=True, help="Number of servers")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    gear_dir = args.gear_dir
    log_path = args.log
    debug = args.debug
    num_servers = int(args.num_nodes)

    # Llamas only work if GPUs available.
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0

    # Init ray.
    if debug:
        tmp_path = f"/state/partition1/user/{os.environ['USER']}/raytmp"
        os.makedirs(tmp_path, exist_ok=True)
        os.environ['TMPDIR'] = tmp_path # Create symlink to shorten tmp dir path for ray.

        # os.environ['TMPDIR'] = "/home/gridsan/fkossmann/ray_tmp" # Create symlink to shorten tmp dir path for ray.
        ray.init(num_cpus=1)
    else:
        ray.init(os.environ["ip_head"])

    # Get workload config, gear plan and models.
    models = get_model_dict("supercloud")
    workload_config = WorkloadConfig("hellaswag")
    gear_plan = load_gear_plan(gear_dir)

    # Init frontend.
    frontend = Frontend(model_dict=models, gear_plan=gear_plan, log_prefix=log_path, num_servers = num_servers)
    time.sleep(1)

    # Run workload.
    logger.info("Workload runner start")
    run_workload(frontend, workload_config)
    frontend.shutdown()
