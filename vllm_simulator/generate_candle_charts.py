'''Used to generate candle charts to clearly show outliers from LLM metrics.

Specs:
- Individual request errors as dots.
- Candle (IQR box + lines) shows mean, 25/75 percentile, max, min.
- Can show 2 candles: relative error, abs error.
'''

import json
import matplotlib.pyplot as plt
import numpy as np
import os

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


ERROR_TYPE_TO_PLOT_TITLE = {
    "ttft_relative_errors": "Relative Simulation Error for TTFT Across Workloads",
    "tgt_relative_errors": "Relative Simulation Error for TGT Across Workloads",
}


def create_predicted_plot_data(
    workloads,
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_num_batched_tokens=1024,
    max_model_len=8192,
    max_num_seqs=1024):
    # Read in metrics data.
    results_dir = f"results/{model_name}_batch-{max_num_batched_tokens}_max-model-len-{max_model_len}_max-num-seqs-{max_num_seqs}"
    metrics_dir = f"{results_dir}/predicted_metrics"

    # Read old plot_data.
    with open(f"{results_dir}/plot_data.json", 'r') as file:
        plot_data = json.load(file)

    for workload_name in workloads.keys():
        with open(f"{metrics_dir}/{workload_name}.json", 'r') as file:
            workload_metrics = json.load(file)

        # Tracks absolute errors to add to plot_data.
        ttft_relative_sim_error = []

        for request_id in workload_metrics.keys():
            # Append to plot_data intermediate lists.
            ttft_relative_sim_error.append(workload_metrics[request_id]["ttft"]["relative_sim_error"])

        # Rewrite error to plot_data.
        plot_data[workload_name]["ttft"]["relative_sim_error"] = ttft_relative_sim_error

    plot_data_file_path = f"{results_dir}/predicted_plot_data.json"
    # Write plot data to file.
    with open(plot_data_file_path, 'w') as file:
        file.write(json.dumps(plot_data, indent=4))

def read_plot_data(    
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_num_batched_tokens=1024,
    max_model_len=8192,
    max_num_seqs=1024):
    # Get plot data.
    results_dir = f"results/{model_name}_batch-{max_num_batched_tokens}_max-model-len-{max_model_len}_max-num-seqs-{max_num_seqs}"
    plot_data_file_path = f"{results_dir}/predicted_plot_data.json"
    # Open file.
    with open(plot_data_file_path, 'r') as file:
        plot_data = json.load(file)
    # Reformats all stats for ease of plotting.
    ttft_relative_errors = {}
    tgt_relative_errors = {}
    # Parse data for plot.
    for workload_name, workload_data in plot_data.items():
        # Parse out lists of request errors.
        ttft_relative_errors[workload_name] = workload_data["ttft"]["relative_sim_error"]
        tgt_relative_errors[workload_name] = workload_data["tgt"]["relative_sim_error"]  
    return {
        "ttft_relative_errors": ttft_relative_errors,
        "tgt_relative_errors": tgt_relative_errors}


def create_combined_relative_error_plot(
    ttft_data: dict,
    tgt_data: dict,
    output_path: str
):
    """
    Creates a side-by-side figure showing relative simulation errors for TTFT and TGT.

    Args:
        ttft_data (dict): Workload to list of TTFT relative errors.
        tgt_data (dict): Workload to list of TGT relative errors.
        output_path (str): Path to save the output plot.
    """

    def add_single_candle(ax, data, x_center, color):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        mean = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)

        # IQR box
        ax.add_patch(plt.Rectangle((x_center - 0.3, q1), 0.6, iqr,
                                   facecolor=color, alpha=0.4, edgecolor='black'))
        # Whiskers
        ax.plot([x_center, x_center], [min_val, q1], color='black')
        ax.plot([x_center, x_center], [q3, max_val], color='black')
        # Mean line
        ax.plot([x_center - 0.3, x_center + 0.3], [mean, mean], color='red', linestyle='--')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 18, 6; 9, 3

    for ax, data, title in zip(
        axes,
        [ttft_data, tgt_data],
        ["TTFT", "TGT"]
    ):
        workload_names = list(data.keys())
        for idx, workload in enumerate(workload_names):
            errors = np.array(data[workload])
            # Scatter points with jitter
            x_vals = np.random.normal(loc=idx, scale=0.08, size=len(errors))
            ax.scatter(x_vals, errors, alpha=0.4, color='gray')
            # Candle overlay
            add_single_candle(ax, errors, x_center=idx, color='blue')

        # Formatting
        fontsize=12 # 14
        ax.set_xticks(range(len(workload_names)))
        ax.set_xticklabels(workload_names, rotation=45, ha='right', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Attach y-label to the left plot only so it aligns with the grid, not global fig
    axes[0].set_ylabel("Relative Simulation Error", fontsize=14)

    plt.tight_layout()
    # Save as both pdf and png for easy vscode viewing.
    plt.savefig(f"{output_path}.pdf")
    plt.savefig(f"{output_path}.png")


def generate_candle_charts(
    workloads,
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_num_batched_tokens=1024,
    max_model_len=8192,
    max_num_seqs=1024
):
    '''Used to generate candle charts to clearly show outliers from LLM metrics.

    Specs:
    - Individual request errors as dots.
    - Candle (IQR box + lines) shows mean, 25/75 percentile, max, min.
    - Can show 2 candles: relative error, abs error.
    '''
    create_predicted_plot_data(
        workloads,
        model_name,
        model,
        max_num_batched_tokens,
        max_model_len,
        max_num_seqs)

    plot_data = read_plot_data(
        model_name,
        model,
        max_num_batched_tokens,
        max_model_len,
        max_num_seqs)

    output_dir = f"predicted_candle_charts"
    os.makedirs(os.path.dirname(f"{output_dir}/dummy_file.txt"), exist_ok=True)

    create_combined_relative_error_plot(
        ttft_data=plot_data["ttft_relative_errors"],
        tgt_data=plot_data["tgt_relative_errors"],
        output_path=f"{output_dir}/{model_name}_{max_num_batched_tokens}",
    )


if __name__ == "__main__":
    ############################
    # Generate candle charts. ##
    ############################
    # generate_candle_charts(
    #     workloads=WORKLOADS,
    #     model_name="llama8b_fp8",
    #     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    #     max_num_batched_tokens=512,
    #     max_model_len=8192,
    #     max_num_seqs=512)

    # generate_candle_charts(
    #     workloads=WORKLOADS,
    #     model_name="llama8b_fp8",
    #     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    #     max_num_batched_tokens=1024,
    #     max_model_len=8192,
    #     max_num_seqs=1024)

    # generate_candle_charts(
    #     workloads=SMALL_WORKLOADS,
    #     model_name="llama70b_fp8",
    #     model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
    #     max_num_batched_tokens=512,
    #     max_model_len=6700,
    #     max_num_seqs=512)

    generate_candle_charts(
        workloads=SMALL_WORKLOADS,
        model_name="llama70b_fp8",
        model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
        max_num_batched_tokens=1024,
        max_model_len=6700,
        max_num_seqs=1024)
