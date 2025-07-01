import os
from datetime import datetime


def parse_file(file_name):
    # Dictionary to store timestamps for each id
    timestamps = {}

    with open(file_name, 'r') as f:
        for line in f:
            # Splitting by comma to separate timestamp and id
            timestamp_str, id_str = line.strip().split(',')
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            id = int(id_str)
            timestamps[id] = timestamp
    return timestamps


def compute_throughput_diff(consumer_file, producer_file):
    consumer_timestamps = parse_file(consumer_file)
    producer_timestamps = parse_file(producer_file)

    # Get the largest id from consumer_timestamps
    max_id = max(consumer_timestamps.keys())

    # Calculate difference in seconds
    delta = consumer_timestamps[max_id] - producer_timestamps[0]
    return delta.total_seconds()

x = "01"  # replace this with appropriate number
consumer_file = f'../workloads/twitter/log_consumer.txt'
producer_file = f'../workloads/twitter/log_producer.txt'

time_difference = compute_throughput_diff(consumer_file, producer_file)
print(f"Time difference in seconds for x={x}: {time_difference}")
