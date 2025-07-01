import csv
from collections import defaultdict
from datetime import datetime
import numpy as np

def get_qps_freq(scale_factor=1, num_samples=100000):
    """
    Get how often different QPS occur
    :param scale_factor: scale factor of workload
    :return: number of occurences of qps bin, qps of bin
    """
    # Parse the CSV data and extract the date-time.
    timestamps_in_seconds = []

    # Date-time parsing format
    fmt = '%a %b %d %H:%M:%S'

    # Assuming the CSV data is in a file called 'data.csv'
    with open('twitter_sentiment.csv', 'r', encoding='latin1') as f:
        reader = csv.reader(f)
        counter = 0
        for row in reader:

            if counter % 1000 == 0:
                print(counter)
            if counter >= num_samples:
                break

            date_string = row[2][:-9]

            # Convert date-time string to a datetime object.
            dt = datetime.strptime(date_string, fmt)

            # Convert this datetime object into a timestamp.
            timestamp = int(dt.timestamp())
            for prog in range(scale_factor):
                noise = int(np.round(np.random.normal(0, 2)))
                timestamps_in_seconds.append(timestamp + noise)

            counter += 1
            # print(date_string, timestamp)

    # Compute the number of rows for each unique second.
    second_counts = defaultdict(int)
    for second in timestamps_in_seconds:
        second_counts[second] += 1

    # Get counts (i.e., number of rows per second).
    counts = list(second_counts.values())

    # Create a histogram of counts.
    qps_freq = np.bincount(counts)
    return qps_freq, np.linspace(0, qps_freq.shape[0]-1, qps_freq.shape[0])

if __name__ == "__main__":
    import json

    qps_freq, qps_bins = get_qps_freq(250)
    qps_freq_list = qps_freq.tolist()
    qps_bins_list = qps_bins.tolist()

    # Pack data into a dictionary
    data = {
        "qps_freq": qps_freq_list,
        "qps_bins": qps_bins_list
    }

    # Serialize the dictionary to JSON
    json_str = json.dumps(data, indent=4)

    # Write to a file
    with open("../../offline/profile/twitter_profile_250.json", "w") as json_file:
        json_file.write(json_str)
