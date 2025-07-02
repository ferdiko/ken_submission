"""Helper functions used in verification of tokens per forward pass."""
import json
from typing import List, Optional

from vllm_simulator.simulator.simulated_request_metrics_visualizer import SimulatedRequestMetricsVisualizer
from vllm_simulator.simulator.simulated_request import SimulatedRequest
from utils.simulator_helpers import get_logger_records
import utils.simulator_helpers as simulator_helpers

from utils import configure_logging
# logger = configure_logging(__name__)
logger = configure_logging(__name__, log_to_file=True, log_file_path="outputs/logs/vllm_simulator.log")


def generate_simulated_requests(in_tokens: List[int], out_tokens: List[int], query_arrival_timestamps: List[float], query_names: List[str]) -> List[SimulatedRequest]:
    """Converts request specifications into SimulatedRequest instances.
    
    Arguments:
        in_tokens: Number of in_tokens for each request.
        out_tokens: Number of out_tokens for each request.
        query_arrival_timestamps: Time when the query arrives to the model. This is typically taken from the vllm run.
        query_ids: List of strings that indicate request_id as assigned by vllm.
    """
    simulated_requests = []
    for i in range(len(in_tokens)):
        simulated_requests.append(
            SimulatedRequest(
                in_tokens=in_tokens[i],
                out_tokens=out_tokens[i],
                arrival_timestamp=query_arrival_timestamps[i],
                name=query_names[i])
        )
    return simulated_requests


def parse_event_tokens(json_file_path: str) -> List[dict[str, int]]:
    """Reads a json file of forward pass token logging and returns the matching Python object."""
    with open(json_file_path, "r") as file:
        event_tokens = json.load(file)
    return event_tokens


def verify_event_tokens(vllm_event_tokens: List[dict[str, int]], simulator_event_tokens: List[dict[str, int]], early_exit: bool = True) -> bool:
    """For the first forward pass with discrepancy, accumulate and log all differences.
    
    Returns true if no differences found."""
    i = -1
    events_differ = False
    for vllm_event, sim_event in zip(vllm_event_tokens, simulator_event_tokens):
        i += 1
        if vllm_event == sim_event:
            # Forward pass matches.
            continue

        events_differ = True
        logger.info(f"FORWARD PASS {i}")
        # Else, log differences.
        # Find extra requests in vllm or sim event.
        extra_sim_requests = sim_event.keys() - vllm_event.keys() 
        if extra_sim_requests:
            logger.info(f"\tEXTRA SIM REQUESTS")
        for request_id in extra_sim_requests:
            logger.info(f"\t\t* Request-{request_id}: {sim_event[request_id]} tokens")
        extra_vllm_requests = vllm_event.keys() - sim_event.keys()
        if extra_vllm_requests:
            logger.info(f"\tEXTRA VLLM REQUESTS")
        for request_id in extra_vllm_requests:
            logger.info(f"\t\t* Request-{request_id}: {vllm_event[request_id]} tokens")
        
        # Find changed values for common requests.
        common_requests = vllm_event.keys() & sim_event.keys()
        logger.info(f"\tCOMMON REQUESTS with different tokens")
        for request_id in common_requests:
            if vllm_event[request_id] != sim_event[request_id]:
                logger.info(f"\t\t* Request-{request_id}: {vllm_event[request_id]} != {sim_event[request_id]}")
        
        if early_exit:
            logger.info("Not checking for other mismatched events. Check log file for other discrepancies.")
            return False

    # Check if simulator has more or less events.
    if len(simulator_event_tokens) == len(vllm_event_tokens) and not events_differ:
        logger.info("Vllm and simulator events match! :D")
        # All events match!
        return True
    
    if len(simulator_event_tokens) > len(vllm_event_tokens):
        logger.info(f"Simulator has extra events.")
    elif len(simulator_event_tokens) < len(vllm_event_tokens):
        logger.info(f"Simulator has less events.")

    # Difference in events.
    return False


def get_event_tokens_forward_pass(json_file_path, i):
    """Logs the ith event in the specified event_tokens json."""
    event_tokens = parse_event_tokens(json_file_path)
    return event_tokens[i]


def print_event_tokens_forward_pass(json_file_path, i):
    """Logs the ith event in the specified event_tokens json."""
    event_tokens = parse_event_tokens(json_file_path)
    logger.info(event_tokens[i])


def clear_logs():
    VLLM_LOG_FILE_PATH = "outputs/logs/vllm.log"
    SIMULATOR_LOG_FILE_PATH = "outputs/logs/vllm_simulator.log"
    # Clear logs.
    with open(VLLM_LOG_FILE_PATH, 'w'):
        pass
    with open(SIMULATOR_LOG_FILE_PATH, 'w'):
        pass    


#######################################
## Helpers for metrics verification. ##
#######################################
def new_parse_vllm_metrics(json_file_path) -> dict:
    """Reads a json file of forward pass token logging and returns the matching Python object."""
    with open(json_file_path, "r") as file:
        vllm_metrics_list = json.load(file)
    # Reorganize list into a dictionary.
    vllm_metrics = {}
    for request_metric in vllm_metrics_list:
        vllm_metrics[request_metric["name"]] = {
            "arrival_time": request_metric["arrival_time"],
            "first_scheduled_time": request_metric["first_scheduled_time"],
            "time_in_queue": request_metric["time_in_queue"],
            "first_token_time": request_metric["first_token_time"],
            "finished_time": request_metric["finished_time"],
            "ttft": request_metric["ttft"],
            "tgt": request_metric["tgt"],
            "steps_in_queue": request_metric["steps_in_queue"],
        }
    return vllm_metrics


def get_vllm_metrics_from_logger() -> dict:
    """Retrieves list of forward pass logging from logger and returns the matching Python object."""
    vllm_metrics_list = get_logger_records(logger_name="vllm_metrics_verification_logger")
    # Reorganize list into a dictionary.
    vllm_metrics = {}
    for request_metric in vllm_metrics_list:
        vllm_metrics[request_metric["name"]] = {
            "arrival_time": request_metric["arrival_time"],
            "first_scheduled_time": request_metric["first_scheduled_time"],
            "time_in_queue": request_metric["time_in_queue"],
            "first_token_time": request_metric["first_token_time"],
            "finished_time": request_metric["finished_time"],
            "ttft": request_metric["ttft"],
            "tgt": request_metric["tgt"],
            "scheduler_time": request_metric["scheduler_time"],
            "steps_in_queue": request_metric["steps_in_queue"],
        }
    return vllm_metrics


def new_parse_simulator_metrics(request_metrics_visualizer, simulator_metrics_file_path=None) -> dict:
    """Reads a json file of forward pass token logging and returns the matching Python object."""
    # Reorganize list into a dictionary.
    sim_metrics = request_metrics_visualizer.to_request_dict()
    # Write metrics to file for easier debugging.
    SIMULATOR_LOG_FILE_PATH = "outputs/verification/simulator_metrics.json" if simulator_metrics_file_path is None else simulator_metrics_file_path
    simulator_helpers.maybe_create_dir(SIMULATOR_LOG_FILE_PATH)
    with open(SIMULATOR_LOG_FILE_PATH, 'w') as file:
        file.write(json.dumps(sim_metrics, indent=4))
    return sim_metrics


def calculate_relative_error(vllm_metric, sim_metric):
    '''Returns relative simulation error in percent'''
    if sim_metric == 0:
        sim_metric = 1e-6
    difference = (sim_metric - vllm_metric) / vllm_metric
    return difference * 100


def new_output_request_metric(file_path, request_metrics_visualizer, simulator_metrics_file_path=None, include_forward_passes=False):
    # Read vllm request metrics from file.
    vllm_metrics = get_vllm_metrics_from_logger()
    # Parses simulator_metrics (and writes simulator metrics to file for easier debugging).
    simulator_metrics = new_parse_simulator_metrics(request_metrics_visualizer, simulator_metrics_file_path)

    output = {}
    for request_name, sim_request_metric in simulator_metrics.items():
        vllm_request_metric = vllm_metrics[request_name]
        
        parsed_request_metric = {
            "in_tokens": sim_request_metric["in_tokens"],
            "out_tokens": sim_request_metric["out_tokens"],
            "time_in_queue": {
                "vllm": vllm_request_metric["time_in_queue"],
                "sim": sim_request_metric["time_in_queue"],
            },
            "steps_in_queue": {
                "vllm": vllm_request_metric["steps_in_queue"],
                # "sim": sim_request_metric["steps_in_queue"],
            },
            "ttft": {
                "vllm": vllm_request_metric["ttft"],
                "sim": sim_request_metric["ttft"],
                "relative_sim_error": calculate_relative_error(vllm_request_metric["ttft"], sim_request_metric["ttft"])
            },
            "tgt": {
                "vllm": vllm_request_metric["tgt"],
                "sim": sim_request_metric["tgt"],
                "relative_sim_error": calculate_relative_error(vllm_request_metric["tgt"], sim_request_metric["tgt"])
            }
        }

        if include_forward_passes:
            parsed_request_metric["forward_passes"] = {
                "vllm": "",
                "sim": sim_request_metric["forward_passes"]
            }

        output[request_name] = parsed_request_metric

    simulator_helpers.maybe_create_dir(file_path)
    with open(file_path, 'w') as file:
        file.write(json.dumps(output, indent=4))

    return output