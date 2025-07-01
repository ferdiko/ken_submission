import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

import numpy as np

# Parse the CSV data and extract the date-time.
timestamps_in_seconds = []

# Date-time parsing format
fmt = '%a %b %d %H:%M:%S'

# Assuming the CSV data is in a file called 'data.csv'
with open('../workloads/twitter/twitter_sentiment.csv', 'r', encoding='latin1') as f:
    reader = csv.reader(f)
    for row in reader:
        date_string = row[2][:-9]

        # Convert date-time string to a datetime object.
        dt = datetime.strptime(date_string, fmt)

        # Convert this datetime object into a timestamp.
        timestamp = int(dt.timestamp())
        timestamps_in_seconds.append(timestamp)

        # print(date_string, timestamp)

# Compute the number of rows for each unique second.
second_counts = defaultdict(int)
for second in timestamps_in_seconds:
    second_counts[second] += 1

# Get counts (i.e., number of rows per second).
counts = list(second_counts.values())

# Create a histogram of counts.
plt.hist(counts, bins=range(1, max(counts) + 2), align='left', density=True, rwidth=0.8)

print(np.bincount(counts))

plt.xlabel('Queries per Second')
plt.ylabel('Fraction of occurence')
plt.title('Distribution of Queries per Second')
plt.show()
