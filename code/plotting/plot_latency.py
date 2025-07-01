import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# Function to calculate the time difference in milliseconds
def calculate_time_difference(enter_time, leave_time):
    return (leave_time - enter_time).total_seconds() * 1000


def plot_cdf(time_differences, p=0.95, label=""):
    # Sort the time differences
    time_differences.sort()

    print(f"p{int(p*100)} latency:", time_differences[int(p*len(time_differences))])

    # Calculate CDF
    num_samples = len(time_differences)
    cdf_values = np.arange(1, num_samples + 1) / num_samples

    # Plot the latency CDF
    plt.plot(time_differences, cdf_values, marker='.', label=label)


def plot_chronological(time_differences):
    raise NotImplementedError


def plot(producer_path, consumer_path, cdf=True):

    # List to store time differences
    time_differences = []

    # Dictionary to store ENTER timestamps for each ID
    enter_timestamps = {}

    # Read producer file to figure out when queries entered the system
    with open(producer_path, "r") as log_file:
        for line in log_file:
            # Split the line into parts
            parts = line.strip().split(",")

            # Ensure the line has the expected format (ID, Timestamp, Event)
            if len(parts) != 2:
                continue

            # Parse the timestamp
            timestamp, id = parts
            try:
                parsed_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                continue

            enter_timestamps[id] = parsed_timestamp

    # read consumer path to figure out when queries are done
    with open(consumer_path, "r") as log_file:
        for line in log_file:
            # Split the line into parts
            parts = line.strip().split(",")

            # Ensure the line has the expected format (ID, Timestamp, Event)
            if len(parts) != 2:
                continue

            # Parse the timestamp
            timestamp, id = parts
            try:
                parsed_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                continue

            # Check if there is a corresponding ENTER timestamp
            if id in enter_timestamps:
                enter_time = enter_timestamps.pop(id)
                leave_time = parsed_timestamp
                time_difference_ms = calculate_time_difference(enter_time, leave_time)
                time_differences.append(time_difference_ms)
            # else:
            #     assert False, f"ID {id} in consumer log is not in producer log"

    plt.figure(figsize=(8, 6))

    if cdf:
        plot_cdf(time_differences, label="unsimulated")

        sim = np.load("/Users/ferdi/Documents/ensemble-serve/offline/batch_size/tmp_meeting_sim_fig.npy")
        sim *= 1000
        plot_cdf(sim, label="simulated")

    else:
        plot_chronological(time_differences)


    plt.legend()
    plt.title("Latency Cumulative Distribution Function (CDF)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    consumer_path = "../online/log_consumer.txt"
    producer_path = "../online/log_producer.txt"
    plot(producer_path, consumer_path, cdf=True)