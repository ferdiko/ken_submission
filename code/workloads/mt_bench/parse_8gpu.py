import os
from ast import literal_eval
import json
import matplotlib.pyplot as plt

def parse_time_log_file(filename):
    entries = []
    with open(filename, 'r') as f:
        current_entry = None
        for line in f:
            line = line.strip()
            if line.startswith('----------------- '):
                if current_entry is not None:
                    entries.append(current_entry)
                current_entry = {}
                current_entry['title'] = line.split('----------------- ')[1]
                current_entry['latencies'] = {}
            elif current_entry is not None and line:
                parts = line.split(': ')
                if len(parts) == 2:
                    key, value = parts
                    current_entry['latencies'][key.strip()] = float(value.strip())
        if current_entry is not None:
            entries.append(current_entry)

    return entries

def extract_dict_from_filename(filename):
    base_dir = 'batch_3011_8gpu'
    if not os.path.exists(filename):
        # Try adding the base directory if the file is not found
        filename = os.path.join(base_dir, os.path.basename(filename))
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return None
    with open(filename, "r") as f:
        data = json.load(f)
    final_dict = {}
    for k, v in data.items():
        final_dict[int(k)] = v
    return final_dict

def read_scores(filename):
    scores = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if 'mt_bench_single_model_samples' in line:
                parts = line.split(':', 1)
                filename_key = parts[0].split('/')[-1]
                rest = parts[1].strip()
                list_str_start = rest.find('[')
                if list_str_start != -1:
                    list_str = rest[list_str_start:]
                    try:
                        score_list = literal_eval(list_str)
                        scores[filename_key] = score_list
                    except:
                        pass
    return scores

def extract_models_from_title(title):
    # Remove directory path if any
    filename = os.path.basename(title)
    # Remove extension
    filename = os.path.splitext(filename)[0]
    # Now, split by '_'
    parts = filename.split('_')
    # Model names to look for
    model_names = {'1b', '3b', '8b', '70b'}
    models = [part for part in parts if part in model_names]
    return models

def compute_pareto_front(latencies, scores):
    # Combine latencies and scores into a list of tuples
    points = list(zip(latencies, scores))
    # Sort points by latency (ascending), then by score (descending)
    points.sort(key=lambda x: (x[0], -x[1]))
    pareto_front = []
    max_score = -float('inf')
    for latency, score in points:
        if True or score > max_score:
            pareto_front.append((latency, score))
            max_score = score
    return pareto_front

def main():
    time_logs = []
    filename = 'batch_3011_8gpu/time_log.txt'
    if os.path.exists(filename):
        entries = parse_time_log_file(filename)
        time_logs.extend(entries)
    else:
        print(f"File {filename} does not exist.")
        return

    scores_filename = 'pred_scores/scores.txt'
    if not os.path.exists(scores_filename):
        print(f"File {scores_filename} does not exist.")
        return
    scores_dict = read_scores(scores_filename)

    latency_metrics = [
        ('TTFT 95th Percentile', 'ttft_p95.png'),
        ('TTFT 50th Percentile', 'ttft_p50.png'),
        ('E2E 95th Percentile', 'e2e_p95.png'),
        ('E2E 50th Percentile', 'e2e_p50.png')
    ]

    # Initialize plot_data per metric and color
    plot_data = {}
    for metric, _ in latency_metrics:
        plot_data[metric] = {}
        for color in ['blue', 'red', 'black', 'green', 'orange']:
            plot_data[metric][color] = {'latencies': [], 'scores': [], 'titles': []}

    for entry in time_logs:
        title = entry['title']

        models = extract_models_from_title(title)
        if title.endswith('42s.json') and "qps0.35_" in title:
            dict_values = extract_dict_from_filename(title)
            if dict_values is None:
                continue  # Skip if the JSON file couldn't be read
            scores_list = []
            if len(models) == 1:
                model_name = models[0]
                scores_list_model = scores_dict.get(f'2x1600_{model_name}.txt', [])
                if not scores_list_model:
                    print(f"Warning: Scores for model {model_name} not found.")
                    continue
                for key, size in dict_values.items():
                    key_mod = key % 160
                    if key_mod < len(scores_list_model):
                        scores_list.append(scores_list_model[key_mod])
                    else:
                        print(f"Warning: Key {key} is out of range.")
            elif len(models) == 2:
                small_model, large_model = models
                scores_small = scores_dict.get(f'2x1600_{small_model}.txt', [])
                scores_large = scores_dict.get(f'2x1600_{large_model}.txt', [])
                if not scores_large:
                    print(f"Warning: Scores for model {large_model} not found.")
                    continue
                for key, size in dict_values.items():
                    key_mod = key % 160
                    if size == 'small':
                        if scores_small:
                            if key_mod < len(scores_small):
                                scores_list.append(scores_small[key_mod])
                            else:
                                print(f"Warning: Key {key} is out of range for scores_small.")
                        else:
                            print(f"Warning: Scores for model {small_model} not found.")
                    elif size == 'large':
                        if key_mod < len(scores_large):
                            scores_list.append(scores_large[key_mod])
                        else:
                            print(f"Warning: Key {key} is out of range for scores_large.")
                    else:
                        print(f"Warning: Unknown size '{size}' for key {key}.")
            else:
                print(f"Cannot determine models from title: {title}")
                continue

            if scores_list:
                final_score = sum(scores_list) / len(scores_list)
            else:
                final_score = None

            if final_score is not None:
                if 'ms_' in title:
                    color = 'orange'  # Entries with 'ms_'
                elif 'queue-1' in title:
                    color = 'red'  # Entries with 'queue-1'
                elif 'cert0.0' in title:
                    color = 'green'  # Entries with 'cert0.0'
                elif 'medusa' in title:
                    color = 'black'  # Entries with 'medusa'
                else:
                    color = 'blue'  # All other entries
                for metric, _ in latency_metrics:
                    latency_value = entry['latencies'].get(metric, None)
                    if latency_value is not None:
                        plot_data[metric][color]['latencies'].append(latency_value)
                        plot_data[metric][color]['scores'].append(final_score)
                        plot_data[metric][color]['titles'].append(title)
            else:
                print(f"Computed Score is None for title: {title}")
        else:
            print(f"Title does not end with '42s.json': {title}")
            continue

    # Dictionary to hold Pareto points per color and metric
    pareto_points = {}

    for metric, filename in latency_metrics:
        plt.figure()
        color_order = ['blue', 'red', 'black', 'green', 'orange']

        for color in color_order:
            data = plot_data[metric][color]
            if data['latencies']:
                # Compute Pareto front for this color and metric
                pareto_front = compute_pareto_front(data['latencies'], data['scores'])

                if pareto_front:
                    # Update data to include only Pareto optimal points
                    pareto_latencies, pareto_scores = zip(*pareto_front)
                    plot_data[metric][color]['latencies'] = pareto_latencies
                    plot_data[metric][color]['scores'] = pareto_scores

                    # Store Pareto points in pareto_points dictionary
                    metric_name = metric.lower().replace(' ', '_').replace('95th_percentile', 'p95').replace('50th_percentile', 'p50').replace('ttft', 'ttft').replace('e2e', 'e2e')
                    color_name_map = {
                        'blue': 'blue',
                        'red': 'red',
                        'black': 'black',
                        'green': 'green',
                        'orange': 'orange',
                    }
                    var_name_latency = f"{color_name_map[color]}_{metric_name}"
                    var_name_score = f"{color_name_map[color]}_{metric_name}_score"

                    pareto_points[var_name_latency] = list(pareto_latencies)
                    pareto_points[var_name_score] = list(pareto_scores)

                    # Plot only Pareto optimal points
                    plt.scatter(
                        pareto_latencies,
                        pareto_scores,
                        c=color,
                        label=color_name_map[color],
                        edgecolors='none'
                    )
                else:
                    print(f"No Pareto optimal points for color '{color}' and metric '{metric}'.")
            else:
                print(f"No data for color '{color}' and metric '{metric}'.")

        plt.xlabel(metric)
        plt.ylabel('Computed Score')
        plt.title(f'Computed Score vs {metric}')

        # Adjust legend with proper labels in the plotting order
        from matplotlib.lines import Line2D
        color_label_map = {
            'blue': 'Other',
            'red': 'queue-1',
            'black': 'medusa',
            'green': 'cert0.0',
            'orange': 'ms_',
        }
        legend_elements = []
        for color in color_order:
            if plot_data[metric][color]['latencies']:
                legend_elements.append(Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label=color_label_map[color],
                    markerfacecolor=color,
                    markersize=8
                ))
        if legend_elements:
            plt.legend(handles=legend_elements)
        else:
            print(f"No data to display for metric '{metric}'.")

        if filename == "ttft_p95.png":
            plt.xlim(0, 10000)
        elif filename == "ttft_p50.png":
            plt.xlim(0, 3000)

        plt.savefig(filename)
        plt.close()

    # Print Pareto optimal points as lists with latencies and scores
    for var_name in sorted(pareto_points.keys()):
        print(f"{var_name} = {pareto_points[var_name]}\n")

if __name__ == "__main__":
    main()
