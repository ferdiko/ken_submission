'''Runs metrics verifier for various workloads and produces plots.

Saves all raw files for future use.'''
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics
from typing import List

from utils import configure_logging
import utils.metrics_verifier as metrics_verifier
import utils.simulator_helpers as simulator_helpers

# Initialize logging.
logger = configure_logging()

# Workloads in 5:1 ratio of in to out tokens and AzurePublicDataset.
WORKLOADS = {
    "5_sequential_100_in": {
        "in_tokens": [100 for _ in range(5)],
        "out_tokens": [10 for _ in range(5)],
        "queries_per_iter": [1 for _ in range(5)],
    },
    "5_sequential_1000_in": {
        "in_tokens": [1000 for _ in range(5)],
        "out_tokens": [200 for _ in range(5)],
        "queries_per_iter": [1 for _ in range(5)],
    },
    "10_sequential_1000_in": {
        "in_tokens": [1000 for _ in range(10)],
        "out_tokens": [200 for _ in range(10)],
        "queries_per_iter": [1 for _ in range(10)],
    },
    "5_concurrent_100_in": {
        "in_tokens": [100 for _ in range(5)],
        "out_tokens": [10 for _ in range(5)],
        "queries_per_iter": [5],
    },
    "5_concurrent_1000_in": {
        "in_tokens": [1000 for _ in range(5)],
        "out_tokens": [200 for _ in range(5)],
        "queries_per_iter": [5],
    },
    "10_concurrent_1000_in": {
        "in_tokens": [1000 for _ in range(10)],
        "out_tokens": [200 for _ in range(10)],
        "queries_per_iter": [10],
    },
    # AzurePublicDataset first 30 requests from workloads. Link: https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md
    "30_concurrent_azure_code": {
        "in_tokens": [4808, 3180, 110, 7433, 34, 374, 6985, 34, 1145, 201, 137, 7427, 1555, 3893, 1827, 394, 675, 7436, 158, 6587, 763, 1556, 5108, 159, 458, 2464, 4009, 1632, 2567, 730],
        "out_tokens": [10, 8, 27, 14, 12, 14, 9, 23, 7, 24, 9, 8, 19, 19, 10, 17, 6, 9, 26, 18, 8, 18, 12, 127, 67, 30, 51, 9, 45, 36],
        "queries_per_iter": [30],
    },
    "30_concurrent_azure_conversation": {
    "in_tokens": [374, 396, 879, 91, 91, 381, 1313, 388, 242, 209, 394, 394, 1315, 2221, 389, 415, 120, 369, 206, 1353, 197, 181, 388, 4085, 2584, 203, 126, 389, 2548, 91],
    "out_tokens": [44, 109, 55, 16, 16, 84, 142, 84, 14, 152, 124, 59, 174, 15, 90, 106, 12, 74, 162, 142, 152, 154, 54, 62, 170, 147, 194, 87, 116, 16],
    "queries_per_iter": [30],
    },
}

# For running with models with max model len < 6650. Same as above, but removed all request pairs in + out > 6650.
# Note: sometimes generated prompts will vary in length, so left some buffer.
SMALL_WORKLOADS = {
    "5_sequential_100_in": {
        "in_tokens": [100 for _ in range(5)],
        "out_tokens": [10 for _ in range(5)],
        "queries_per_iter": [1 for _ in range(5)],
    },
    "5_sequential_1000_in": {
        "in_tokens": [1000 for _ in range(5)],
        "out_tokens": [200 for _ in range(5)],
        "queries_per_iter": [1 for _ in range(5)],
    },
    "10_sequential_1000_in": {
        "in_tokens": [1000 for _ in range(10)],
        "out_tokens": [200 for _ in range(10)],
        "queries_per_iter": [1 for _ in range(10)],
    },
    "5_concurrent_100_in": {
        "in_tokens": [100 for _ in range(5)],
        "out_tokens": [10 for _ in range(5)],
        "queries_per_iter": [5],
    },
    "5_concurrent_1000_in": {
        "in_tokens": [1000 for _ in range(5)],
        "out_tokens": [200 for _ in range(5)],
        "queries_per_iter": [5],
    },
    "10_concurrent_1000_in": {
        "in_tokens": [1000 for _ in range(10)],
        "out_tokens": [200 for _ in range(10)],
        "queries_per_iter": [10],
    },
    # Has 5 less requests
    "25_concurrent_azure_code": {
        "in_tokens": [
            4808, 3180, 110, 34, 374, 34, 1145, 201, 137, 
            1555, 3893, 1827, 394, 675, 158, 763, 
            1556, 5108, 159, 458, 2464, 4009, 1632, 2567, 730
        ],
        "out_tokens": [
            10, 8, 27, 12, 14, 23, 7, 24, 9, 19, 19, 10, 17, 6, 
            26, 8, 18, 12, 127, 67, 30, 51, 9, 45, 36
        ],
        "queries_per_iter": [25]
    },
    # Has same number of requests.
    "30_concurrent_azure_conversation": {
        "in_tokens": [
            374, 396, 879, 91, 91, 381, 1313, 388, 242, 
            209, 394, 394, 1315, 2221, 389, 415, 120, 369, 
            206, 1353, 197, 181, 388, 4085, 2584, 203, 
            126, 389, 2548, 91
        ],
        "out_tokens": [
            44, 109, 55, 16, 16, 84, 142, 84, 14, 152, 124, 
            59, 174, 15, 90, 106, 12, 74, 162, 142, 152, 
            154, 54, 62, 170, 147, 194, 87, 116, 16
        ],
        "queries_per_iter": [30]
    },
}


def generate_plots(
    run_verifier: bool,
    workloads,
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_num_batched_tokens=1024,
    max_model_len=8192,
    max_num_seqs=1024):
    logger.info(f"Generating plots for {model_name}")
    # Raw files and plots will be saved to this directory.
    results_dir = f"results/{model_name}_batch-{max_num_batched_tokens}_max-model-len-{max_model_len}_max-num-seqs-{max_num_seqs}"
    # Create this directory.
    simulator_helpers.maybe_create_dir(results_dir)
    # Plot data output for prosperity.
    plot_data_file_path = f"{results_dir}/plot_data.json"
    if run_verifier:
        # Initialize metrics verifier.
        verifier = metrics_verifier.MetricsVerifier(
            model_name=model_name,
            model=model,
            max_tokens=1, # This does not matter, is just for profiling.
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs)

        # Create the file if needed and save the complete profile.
        profile_file_path = f"{results_dir}/profile.json"
        verifier.dump_complete_profile(profile_file_path)

        # Saves all plot data for prosperity.
        plot_data = {}
        # Run various workloads and save all raw files.
        for workload_name, workload_specs in workloads.items():
            logger.info(f"Start workload {workload_name}")
            in_tokens = workload_specs["in_tokens"]
            out_tokens = workload_specs["out_tokens"]
            queries_per_iter = workload_specs["queries_per_iter"]
            request_metrics_comparison = verifier.compare_metrics(
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                queries_per_iter=queries_per_iter,
                out_file_path=f"{results_dir}/metrics/{workload_name}.json",
                trace_file_path=f"{results_dir}/traces/{workload_name}.json",
                include_forward_passes=True,
            )
            # Calculations for ttft.
            ttft_sim_error_values = [request_metrics["ttft"]["relative_sim_error"] for request_metrics in request_metrics_comparison.values()]
            ttft_average_sim_error = statistics.mean(ttft_sim_error_values)
            ttft_stdev_sim_error = statistics.stdev(ttft_sim_error_values)
            # Calculations for tgt.
            tgt_sim_error_values = [request_metrics["tgt"]["relative_sim_error"] for request_metrics in request_metrics_comparison.values()]
            tgt_average_sim_error = statistics.mean(tgt_sim_error_values)
            tgt_stdev_sim_error = statistics.stdev(tgt_sim_error_values)
            # Format plot_data for prosperity.
            plot_data[workload_name] = {
                "ttft": {
                    "mean": ttft_average_sim_error,
                    "stdev": ttft_stdev_sim_error,
                    "relative_sim_error": ttft_sim_error_values,
                },
                "tgt" : {
                    "mean": tgt_average_sim_error,
                    "stdev": tgt_stdev_sim_error,
                    "relative_sim_error": tgt_sim_error_values,
                }
            }

            # Clear file for plot data points.
            simulator_helpers.maybe_create_dir(plot_data_file_path)
            # Save plot data points.
            with open(plot_data_file_path, 'w') as file:
                file.write(json.dumps(plot_data, indent=4))
            # Save steps.log, vllm_metrics.json.
            # Note: These are hard to parse for each workload but can be reverse engineered to debug any workloads.
            steps_log = simulator_helpers.get_logger_records("steps_logger")
            steps_log_file_path = f"{results_dir}/steps.log"
            simulator_helpers.maybe_create_dir(steps_log_file_path)
            with open(steps_log_file_path, 'w') as file:
                file.write(json.dumps(steps_log, indent=4))
            metrics_log = simulator_helpers.get_logger_records("vllm_metrics_verification_logger")
            metrics_log_file_path = f"{results_dir}/vllm_metrics.log"
            simulator_helpers.maybe_create_dir(metrics_log_file_path)
            with open(metrics_log_file_path, 'w') as file:
                file.write(json.dumps(metrics_log, indent=4))
    else:
        # Read plot_data from json.
        with open(plot_data_file_path, 'r') as file:
            plot_data = json.load(file)

    # Reformats all means for ease of plotting.
    average_sim_error_ttft = []
    average_sim_error_tgt = []
    # Reformats all standard deviations for ease of plotting.
    stdev_sim_error_ttft = []
    stdev_sim_error_tgt = []
    # Parse data for plot.
    for workload_name, workload_data in plot_data.items():
        # Parse dictionary.
        ttft_average_sim_error = workload_data["ttft"]["mean"]
        ttft_stdev_sim_error = workload_data["ttft"]["stdev"]
        tgt_average_sim_error = workload_data["tgt"]["mean"]
        tgt_stdev_sim_error = workload_data["tgt"]["stdev"]
        # Append to lists
        average_sim_error_ttft.append(ttft_average_sim_error)
        stdev_sim_error_ttft.append(ttft_stdev_sim_error)
        average_sim_error_tgt.append(tgt_average_sim_error)        
        stdev_sim_error_tgt.append(tgt_stdev_sim_error)
        logger.info(f"Workload {workload_name} with ttft error: {ttft_average_sim_error}, tgt error: {tgt_average_sim_error}")

    # Generate and save TTFT plot.
    create_and_save_plot(
        title="Relative Simulation Error for TTFT",
        workloads=workloads,
        data=average_sim_error_ttft,
        stdev=stdev_sim_error_ttft,
        output_path=f"{results_dir}/ttft_plot.png")
    # Generate and save TGT plot.
    create_and_save_plot(
        title="Relative Simulation Error for TGT",
        workloads=workloads,
        data=average_sim_error_tgt,
        stdev=stdev_sim_error_tgt,
        output_path=f"{results_dir}/tgt_plot.png")


def create_and_save_plot(title: str, workloads, data: List[float], stdev: List[float], output_path: str):
    """
    Creates a bar plot with standard deviation error bars and workload names as x-axis labels.

    Args:
        title (str): Title for the y-axis.
        data (List[float]): List of values for each workload.
        stdev (List[float]): List of standard deviations for error bars.
        output_path (str): Path to save the output plot.
    """
    workload_names = [key for key in workloads.keys()]
    # HACK: Some workloads are buggy with 70b, so remove the name labels.
    workload_names = workload_names[:len(data)]
    # Ensure data consistency
    assert len(workload_names) == len(data) == len(stdev), \
        f"All input lists must have the same length; {len(workload_names)} != {len(data)} != {len(stdev)}"

    x = np.arange(len(workload_names))  # Use actual workload names for x-axis labels
    y = np.round(np.array(data), 2)  # Format y values to 2 decimal places

    fig, ax = plt.subplots(figsize=(10, 5))  # Increased figure size to prevent overlap

    # Create bar plot with error bars for standard deviation
    bars = ax.bar(x, y, yerr=stdev, capsize=5, color="steelblue", alpha=0.8)

    # Set y-axis limits to allow space for annotations  
    y_min = min(y - stdev) - 1  # Extend lower limit
    y_max = max(y + stdev) + 1  # Extend upper limit
    ax.set_ylim(y_min, y_max)

    # Standardized label distance from the end of the error bars
    label_offset = 0.3  # Fixed distance from the error bars

    # Adjust annotation position to prevent overlap with error bars
    for bar, value, err in zip(bars, y, stdev):
        text_y = (bar.get_height() + err + label_offset) if value > 0 else (bar.get_height() - err - label_offset)
        va_position = 'bottom' if value > 0 else 'top'  # Adjust text alignment
        ax.text(bar.get_x() + bar.get_width()/2, text_y, f"{value:.2f}%", 
                ha='center', va=va_position, fontsize=10)

    # Labels and formatting
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(workload_names, rotation=30, ha="right", fontsize=10)  # Rotated x-axis labels to prevent overlap
    ax.grid(True, linestyle="--", alpha=0.6)

    # Remove unnecessary chart borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to prevent display in interactive environments


if __name__ == "__main__":
    # Note: Only run 1 at a time because can not load multiple models onto the GPUs at once.
    # logger.info("Start generate_plots for 8b, batch 512")
    # generate_plots(
    #     run_verifier=True,
    #     workloads=workloads,
    #     model_name="llama8b_fp8",
    #     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    #     max_num_batched_tokens=512,
    #     max_model_len=8192,
    #     max_num_seqs=512)
    # logger.info("Finish generate_plots for 8b, batch 512")

    # logger.info("Start generate_plots for 8b, batch 1024")
    # generate_plots(
    #     run_verifier=False,
    #     workloads=WORKLOADS,
    #     model_name="llama8b_fp8",
    #     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    #     max_num_batched_tokens=1024,
    #     max_model_len=8192,
    #     max_num_seqs=1024)
    # logger.info("Finish generate_plots for 8b, batch 1024")

    # logger.info("Start generate_plots for 70b, batch 512")
    # generate_plots(
    #     run_verifier=True,
    #     workloads=SMALL_WORKLOADS,
    #     model_name="llama70b_fp8",
    #     model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
    #     max_num_batched_tokens=512,
    #     max_model_len=8192,
    #     max_num_seqs=512)
    # logger.info("Finish generate_plots for 70b, batch 512")

    # Note: KV cache has size 6720 after loading model weights, so decreased max_model_len
    #   (sometimes space will vary, so left some buffer room).
    # logger.info("Start generate_plots for 70b, batch 1024")
    # generate_plots(
    #     run_verifier=True,
    #     workloads=SMALL_WORKLOADS,
    #     model_name="llama70b_fp8",
    #     model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
    #     max_num_batched_tokens=1024,
    #     max_model_len=6700,
    #     max_num_seqs=1024)
    # logger.info("Finish generate_plots for 70b, batch 1024")
    logger.info("Start generate_plots for 70b, batch 1024")
    generate_plots(
        run_verifier=False,
        workloads=SMALL_WORKLOADS,
        model_name="llama70b_fp8",
        model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
        max_num_batched_tokens=1024,
        max_model_len=6700,
        max_num_seqs=1024)
    logger.info("Finish generate_plots for 70b, batch 1024")
    