import math
import json
import statistics
import os
import matplotlib.pyplot as plt
import numpy as np


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

def get_tpot(results_dir):
    '''Returns average time for a forward pass that invoves a decode token.
    
    Equal weighted average across approximately equally spread intervals of (0, max_batch_size).'''
    # Read in steps_coordinates of decode passes.
    with open(f"{results_dir}/steps_coordinates_cuda.json", "r") as file:
        steps_coords = json.load(file)

    # Calculate tpot per data point.
    cum_tpot = 0
    num_data_pts = 0
    for duration_seconds in steps_coords.values():
        num_data_pts += 1
        cum_tpot += duration_seconds

    # Average tpots.
    return cum_tpot / num_data_pts


def get_max_batch_runtime(results_dir, max_batch_size):
    '''Returns duration (seconds) of batch at max_batch_size tokens.'''
    with open(f"{results_dir}/steps_coordinates_no_cuda.json", "r") as file:
        steps_coords = json.load(file)

    return steps_coords[f"{max_batch_size}"]


def get_baseline_calc_ttft(max_batch_size, max_batch_runtime, in_tokens):
    '''For baseline_calc, return ttft of a request.'''
    num_batches = math.ceil(in_tokens/max_batch_size)
    return num_batches * max_batch_runtime


def get_baseline_decode_time(tpot, out_tokens):
    '''For both baselines, returns time to decode.'''
    return tpot * out_tokens


def calculate_relative_errors(sim_times, vllm_times):
    '''Returns relative simulation error in percent'''
    errors = []
    for sim_metric, vllm_metric in zip(sim_times, vllm_times):
        if sim_metric == 0:
            sim_metric = 1e-6
        difference = (sim_metric - vllm_metric) / vllm_metric
        errors.append(difference * 100)
    return errors


def run_baseline(
    workloads,
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_num_batched_tokens=1024,
    max_model_len=8192,
    max_num_seqs=1024):
    results_dir = f"results/{model_name}_batch-{max_num_batched_tokens}_max-model-len-{max_model_len}_max-num-seqs-{max_num_seqs}"
    metrics_dir = f"{results_dir}/{"metrics" if model_name=="llama8b_fp8" and max_num_batched_tokens==1024 else "predicted_metrics" }" 
    baseline_results_dir = f"results/baseline/{model_name}_{max_num_batched_tokens}"

    # Calculate variables needed for baseline.
    tpot = get_tpot(results_dir)
    max_batch_runtime = get_max_batch_runtime(results_dir, max_num_batched_tokens)

    baseline_metrics = {} # workload_name: {ttft, tgt, errors} for both baselines, sim, and vllm

    for workload_name in workloads.keys():
        with open(f"{metrics_dir}/{workload_name}.json", 'r') as file:
            workload_metrics = json.load(file)

        # Intermediate lists to store data.
        ttft_vllm = []
        ttft_sim = []
        ttft_baseline_calc = []
        ttft_relative_sim_error = []

        tgt_vllm = []
        tgt_sim = []
        tgt_baseline_calc = []
        tgt_baseline_sim = []
        tgt_relative_sim_error = []

        for request_id in workload_metrics.keys():
            calc_ttft = get_baseline_calc_ttft(max_num_batched_tokens, max_batch_runtime, workload_metrics[request_id]["in_tokens"])
            sim_ttft = workload_metrics[request_id]["ttft"]["sim"]
            baseline_decode_time = get_baseline_decode_time(tpot, workload_metrics[request_id]["out_tokens"])

            # Append to intermediate lists.
            ttft_vllm.append(workload_metrics[request_id]["ttft"]["vllm"])
            ttft_sim.append(sim_ttft)
            ttft_baseline_calc.append(calc_ttft)
            ttft_relative_sim_error.append(workload_metrics[request_id]["ttft"]["relative_sim_error"])

            tgt_vllm.append(workload_metrics[request_id]["tgt"]["vllm"])
            tgt_sim.append(workload_metrics[request_id]["tgt"]["sim"])
            tgt_baseline_calc.append(calc_ttft + baseline_decode_time)
            tgt_baseline_sim.append(sim_ttft + baseline_decode_time)
            tgt_relative_sim_error.append(workload_metrics[request_id]["tgt"]["relative_sim_error"])

        # Construct baseline_metrics.
        baseline_metrics[workload_name] = {
            "ttft": {
                "vllm": ttft_vllm,
                "sim": ttft_sim,
                "baseline_calc": ttft_baseline_calc,
                # "baseline_sim" = "sim" because the ttft used is the sim one.
                "relative_sim_error": ttft_relative_sim_error,
                "relative_baseline_calc_error": calculate_relative_errors(ttft_baseline_calc, ttft_vllm),
                # "relative_baseline_sim_error" = "relative_sim_error" because the ttft used is the sim one.          
            },
            "tgt": {
                "vllm": tgt_vllm,
                "sim": tgt_sim,
                "baseline_calc": tgt_baseline_calc,
                "baseline_sim": tgt_baseline_sim,
                "relative_sim_error": tgt_relative_sim_error,
                "relative_baseline_calc_error": calculate_relative_errors(tgt_baseline_calc, tgt_vllm),
                "relative_baseline_sim_error": calculate_relative_errors(tgt_baseline_sim, tgt_vllm), 
            },
        }

    metrics_file_path = f"{baseline_results_dir}/baseline_metrics.json"
    os.makedirs(os.path.dirname(f"{metrics_file_path}"), exist_ok=True)
    # Write metrics data to file.
    with open(metrics_file_path, 'w') as file:
        file.write(json.dumps(baseline_metrics, indent=4))

    return baseline_metrics


def get_plot_data(
    baseline_metrics,
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_num_batched_tokens=1024,
    max_model_len=8192,
    max_num_seqs=1024):
    '''Calculates avg, sd for TGT only.'''
    baseline_results_dir = f"results/baseline/{model_name}_{max_num_batched_tokens}"
    plot_data = {}

    avg_relative_sim_error_tgt = []
    avg_relative_baseline_calc_error_tgt = []
    avg_relative_baseline_sim_error_tgt = []

    # Note: Not needed within a workload.
    # sd_relative_sim_error_tgt = []
    # sd_relative_baseline_calc_error_tgt = []
    # sd_relative_baseline_sim_error_tgt = []

    for workload, metrics in baseline_metrics.items():
        # Calculate average for each workload.
        avg_relative_sim_error_tgt.append(statistics.mean(metrics["tgt"]["relative_sim_error"]))
        avg_relative_baseline_calc_error_tgt.append(statistics.mean(metrics["tgt"]["relative_baseline_calc_error"]))
        avg_relative_baseline_sim_error_tgt.append(statistics.mean(metrics["tgt"]["relative_baseline_sim_error"]))

        # sd_relative_sim_error_tgt.append(statistics.stdev(metrics["tgt"]["relative_sim_error"]))
        # sd_relative_baseline_calc_error_tgt.append(statistics.stdev(metrics["tgt"]["relative_baseline_calc_error"]))
        # sd_relative_baseline_sim_error_tgt.append(statistics.stdev(metrics["tgt"]["relative_baseline_sim_error"]))

    plot_data = {
        "relative_sim_error_tgt": avg_relative_sim_error_tgt,
        "relative_baseline_calc_error_tgt": avg_relative_baseline_calc_error_tgt,
        "relative_baseline_sim_error_tgt": avg_relative_baseline_sim_error_tgt,
    }

    # Write metrics data to file.
    plot_data_file_path = f"{baseline_results_dir}/plot_data.json"
    with open(plot_data_file_path, 'w') as file:
        file.write(json.dumps(plot_data, indent=4))

    return plot_data


def generate_bar_plot(
    plot_data, # takes dictionary or None if read in from json file.
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_num_batched_tokens=1024,
    max_model_len=8192,
    max_num_seqs=1024):
    '''Create a bar chart with 3 bars and standard deviation error bars.'''
    if plot_data is None:
        baseline_results_dir = f"results/baseline/{model_name}_{max_num_batched_tokens}"
        plot_data_file_path = f"{baseline_results_dir}/plot_data.json"
        with open(plot_data_file_path, 'r') as file:
            plot_data = json.load(file)

    # Extract means and standard deviations
    labels = list(plot_data.keys())
    means = [statistics.mean(plot_data[key]) for key in labels]
    stds = [statistics.stdev(plot_data[key]) for key in labels]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    # bars = ax.bar(x, means, yerr=stds, capsize=8)
    bars = ax.bar(x, means, capsize=8)

    # Labeling
    ax.set_ylabel("Relative SimulationError (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["vLLMSim", "Baseline with Calculated TTFT", "Baseline with our Simulated TTFT"], rotation=15, ha='right')

    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    # plt.show()
    baseline_results_dir = f"results/baseline/{model_name}_{max_num_batched_tokens}"
    plot_file_path =f"{baseline_results_dir}/{model_name}_{max_num_batched_tokens}"
    plt.savefig(f"{plot_file_path}.pdf")
    plt.savefig(f"{plot_file_path}.png")


def create_and_save_bar_plot(
    workloads,
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_num_batched_tokens=1024,
    max_model_len=8192,
    max_num_seqs=1024):
    # # Create baselines.
    # baseline_metrics = run_baseline(
    #     workloads,
    #     model_name,
    #     model,
    #     max_num_batched_tokens,
    #     max_model_len,
    #     max_num_seqs)

    # # Parse plot data.
    # plot_data = get_plot_data(
    #     baseline_metrics,
    #     model_name,
    #     model,
    #     max_num_batched_tokens,
    #     max_model_len,
    #     max_num_seqs)

    # # Create and save plot with plot_data
    # generate_bar_plot(
    #     plot_data,
    #     model_name,
    #     model,
    #     max_num_batched_tokens,
    #     max_model_len,
    #     max_num_seqs)

    generate_bar_plot(
        None,
        model_name,
        model,
        max_num_batched_tokens,
        max_model_len,
        max_num_seqs)


def create_combined_plot():
    root_dir = "results/baseline"
    model_dirs = ["llama8b_fp8_512", "llama8b_fp8_1024", "llama70b_fp8_512", "llama70b_fp8_1024"]
    bar_labels = ["vLLMSim", "Baseline with Calculated TTFT", "Baseline with vLLMSim TTFT"]
    bar_colors = plt.cm.tab10.colors[:3]  # consistent colors

    # Prep data
    num_groups = len(model_dirs)
    num_bars = len(bar_labels)
    group_means = []

    for model_dir in model_dirs:
        plot_path = os.path.join(root_dir, model_dir, "plot_data.json")
        with open(plot_path, "r") as f:
            plot_data = json.load(f)
        group_means.append([np.mean(plot_data[k]) for k in plot_data])

    # Plot
    fig, ax = plt.subplots(figsize=(11, 3))

    bar_width = 0.2
    x = np.arange(num_groups)

    for i in range(num_bars):
        values = [group[i] for group in group_means]
        bar_positions = x + i * bar_width
        bars = ax.bar(bar_positions, values, width=bar_width, color=bar_colors[i], label=bar_labels[i])

        # Label each bar with its value
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.1f}%" if height < 100 else f"{height:.0f}%",
                ha='center',
                va='bottom',
                fontsize=12
            )

    # Axis + legend formatting
    ax.set_ylabel("Relative Simulation Error", fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(x + bar_width)
    # ax.set_xticklabels([d.replace("_", "\n") for d in model_dirs], fontsize=14)
    ax.set_xticklabels([
        "Llama 8B\nmax_batch_size=512",
        "Llama 8B\nmax_batch_size=1024",
        "Llama 70B\nmax_batch_size=512",
        "Llama 70B\nmax_batch_size=1024"], fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=12)  # no title

    # Compute max height for dynamic y-limit
    max_height = max([max(group) for group in group_means])
    ax.set_ylim(top=max_height * 1.12)  # Add 25% headroom

    plt.tight_layout()
    plot_file_path = f"{root_dir}/baseline_plot"
    plt.savefig(f"{plot_file_path}.pdf")
    plt.savefig(f"{plot_file_path}.png")
    plt.show()


if __name__ == "__main__":
    create_combined_plot()
    # create_and_save_bar_plot(
    #     workloads=WORKLOADS,
    #     model_name="llama8b_fp8",
    #     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    #     max_num_batched_tokens=512,
    #     max_model_len=8192,
    #     max_num_seqs=512)

    # create_and_save_bar_plot(
    #     workloads=WORKLOADS,
    #     model_name="llama8b_fp8",
    #     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    #     max_num_batched_tokens=1024,
    #     max_model_len=8192,
    #     max_num_seqs=1024)

    # create_and_save_bar_plot(
    #     workloads=SMALL_WORKLOADS,
    #     model_name="llama70b_fp8",
    #     model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
    #     max_num_batched_tokens=512,
    #     max_model_len=6700,
    #     max_num_seqs=512)

    # create_and_save_bar_plot(
    #     workloads=SMALL_WORKLOADS,
    #     model_name="llama70b_fp8",
    #     model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
    #     max_num_batched_tokens=1024,
    #     max_model_len=6700,
    #     max_num_seqs=1024)