import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Function to calculate the time difference in milliseconds
def calculate_time_difference(enter_time, leave_time):
    return (leave_time - enter_time).total_seconds() * 1000


def get_cdf(log_file_path):

    # List to store time differences
    time_differences = []

    # Dictionary to store ENTER timestamps for each ID
    enter_timestamps = {}

    # Open the log file for reading
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            # Split the line into parts
            parts = line.strip().split(",")

            # Ensure the line has the expected format (ID, Timestamp, Event)
            if len(parts) != 3:
                continue

            id, timestamp, event = parts

            # Parse the timestamp
            try:
                parsed_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                continue

            if event == "ENTER":
                # Store the ENTER timestamp for the ID
                enter_timestamps[id] = parsed_timestamp
            elif event == "LEAVE":
                # Check if there is a corresponding ENTER timestamp
                if id in enter_timestamps:
                    enter_time = enter_timestamps.pop(id)
                    leave_time = parsed_timestamp
                    time_difference_ms = calculate_time_difference(enter_time, leave_time)
                    time_differences.append(time_difference_ms)

    # Sort the time differences
    time_differences.sort()

    # Calculate CDF
    num_samples = len(time_differences)
    cdf_values = np.arange(1, num_samples + 1) / num_samples

    return time_differences, cdf_values

old_td, old_cdf = get_cdf("logging.csv")
base_td, base_cdf = get_cdf("logging11.csv")
new_td, new_cdf = get_cdf("logging_after_dimfix.csv")



# Plot the latency CDF
plt.figure(figsize=(8, 6))
plt.plot(old_td, old_cdf, marker='.', label='Our system no ensemble overheads (e.g. thresholding)')
plt.plot(base_td, base_cdf, marker='.', label='Vanilla PyTorch implementation')
plt.plot(new_td, new_cdf, marker='.', label='Our system with ensemble overheads')
plt.legend()
plt.title("Latency Cumulative Distribution Function (CDF)")
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.grid(True)
plt.show()