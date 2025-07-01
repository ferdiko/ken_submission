import matplotlib.pyplot as plt
import json


with open("../workloads/twitter/twitter_bert_prof_bs100.json", "r") as f:
    data = json.load(f)

# Preparing data for plotting with a more general approach
x_values = [int(key) for key in next(iter(data.values())).keys()]  # Assumes all dicts have the same keys
lines = {}

for key in data:
    lines[key] = [data[key][str(x)] / x for x in x_values]

# Plotting with the updated method
plt.figure(figsize=(10, 6))

for key, y_values in lines.items():
    plt.plot(x_values, y_values, label=key, marker='o')

plt.xlabel('X-axis (keys)')
plt.ylabel('Y-axis (values divided by key)')
plt.title('Graph of JSON data')
plt.legend()
plt.grid(True)
plt.show()