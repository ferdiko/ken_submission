import json
import matplotlib.pyplot as plt

# 1. Read the JSON file
with open('twitter_profile_250.json', 'r') as file:
    data = json.load(file)

# 2. Extract categories and values
categories = data["qps_bins"]
values = data["qps_freq"]

sum_below = 0
sum_above = 0
cutoff = 1325
for c, v in zip(categories, values):
    if c < cutoff:
        sum_below += c*v
    else:
        sum_above += c*v

print("PERCENTAGE BELOW:", sum_below/(sum_below+sum_above))


# 3. Plot the bar chart
plt.bar(categories, values)

# Add a title and labels
plt.title('QPS distribution for Twitter dataset')
plt.xlabel('QPS')
plt.ylabel('Occurences')
plt.axvline(x=1325, color='grey', linestyle='--')



# Display the chart
plt.show()
