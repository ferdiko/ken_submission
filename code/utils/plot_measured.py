import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np


log_file = '6servers-24875983.log'
prefix = "6gpu_fixed_"


# Get queries that don't match simulation.
samples_dict = {}

# Compile regex pattern to match the log entry and extract the sample number and timestamp
sample_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - DEBUG \[gpu_server\]:  Sample \[([0-9]+)\]')

# Open the log file and iterate through each line
with open(log_file, 'r') as file:
    for line in file:
        # Check for the specific log entry
        sample_match = sample_pattern.search(line)
        if sample_match:
            timestamp = sample_match.group(1)
            sample_number = int(sample_match.group(2))

            # If the sample number is already in the dict, append the timestamp to its list
            if sample_number in samples_dict:
                samples_dict[sample_number].append(timestamp)
            else:
                # Otherwise, create a new list with the timestamp for this sample number
                samples_dict[sample_number] = [timestamp]



print(samples_dict)
print()
print()


# Get queries with different pred
diff_timestamps = []
warmup = 8

with open(prefix+"consumer.txt", "r") as f:
    cons_lines = f.readlines()[warmup:]

with open("simulated_preds.csv", "r") as f:
    sim_lines_raw = f.readlines()
    sim_lines = []
    for l in sim_lines_raw:
        if l[0] != " ":
            sim_lines.append(l)

print("Qs in consumer:", len(cons_lines), "Qs in sim:", len(sim_lines))


for i, c in enumerate(cons_lines):
    c = c.split(",")

    # get acc
    pred_id = int(c[1])-warmup

    pred = int(c[2])
    sim_pred = int(sim_lines[pred_id].split(",")[0])

    if pred_id < 0:
        continue

    if pred_id not in samples_dict:
        print(pred_id, "not in dict")
        continue

    if pred != int(sim_pred):
        diff_timestamps += samples_dict[pred_id]


# Initialize lists to hold the parsed data
queue_sizes_values = []
queue_sizes_values_second = []
queue_sizes_values_third = []
queue_sizes_timestamps = []
measured_qps_values = []
measured_qps_timestamps = []
issued_qps_values = []
issued_qps_timestamps = []

# Compile regex patterns to match the lines and extract needed parts
queue_sizes_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - DEBUG \[gpu_server\]:  Queue sizes: \[([0-9]+), ([0-9]+), ([0-9]+)')
measured_qps_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - DEBUG \[producer\]:  Measured QPS: (\d+)')
issued_qps_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - DEBUG \[hellaswag_runner\]:  Issued QPS: (\d+)')

# Open the log file and iterate through each line
with open(log_file, 'r') as file:
    for line in file:
        # Check for 'Measured QPS' entries
        measured_match = measured_qps_pattern.search(line)
        if measured_match:
            measured_qps_timestamps.append(measured_match.group(1))
            measured_qps_values.append(int(measured_match.group(2)))

        # Check for 'Issued QPS' entries
        issued_match = issued_qps_pattern.search(line)
        if issued_match:
            issued_qps_timestamps.append(issued_match.group(1))
            issued_qps_values.append(int(issued_match.group(2)))

        queue_match = queue_sizes_pattern.search(line)
        if queue_match:
            queue_sizes_timestamps.append(queue_match.group(1))
            queue_sizes_values.append(int(queue_match.group(2)))
            queue_sizes_values_second.append(int(queue_match.group(3)))
            queue_sizes_values_third.append(int(queue_match.group(4)))



# You can now use the lists for further processing or analysis
# print("Measured QPS Values:", measured_qps_values)
# print("Measured QPS Timestamps:", measured_qps_timestamps)
# print("Issued QPS Values:", issued_qps_values)
# print("Issued QPS Timestamps:", issued_qps_timestamps)

# Convert string timestamps to datetime objects
diff_timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f') for ts in diff_timestamps]

measured_qps_timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f') for ts in measured_qps_timestamps]
issued_qps_timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f') for ts in issued_qps_timestamps]
queue_sizes_timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f') for ts in queue_sizes_timestamps]

# Create figure and plot
plt.figure(figsize=(10, 6))

# Plotting both series with step post to extend a line to the right of each data point
plt.step(measured_qps_timestamps, measured_qps_values, where='post', label='Measured QPS', linewidth=2)
plt.step(issued_qps_timestamps, issued_qps_values, where='post', label='Issued QPS', linewidth=2)
plt.step(queue_sizes_timestamps, queue_sizes_values, where='post', label='Queue sizes 1', linewidth=2)
plt.step(queue_sizes_timestamps, queue_sizes_values_second, where='post', label='Queue sizes 2', linewidth=2)
plt.step(queue_sizes_timestamps, queue_sizes_values_third, where='post', label='Queue sizes 3', linewidth=2)
plt.scatter(diff_timestamps, np.zeros(len(diff_timestamps)), s=50)


# Formatting the x-axis to display the timestamp properly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))  # Adjust the interval as needed
plt.gcf().autofmt_xdate()  # Auto formats the x-axis labels to fit them better

plt.xlabel('Timestamp')
plt.ylabel('QPS Values')
plt.title('Measured vs. Issued QPS Over Time')
plt.legend()
plt.grid(True)

plt.show()
