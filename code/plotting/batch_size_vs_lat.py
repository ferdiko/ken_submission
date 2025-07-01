import matplotlib.pyplot as plt

x = [8, 16, 20, 30, 40]
y = [104, 30, 35, 52, 68]

x_mark, y_mark = 4, 150  # Replace with your specific coordinates

# Plotting the dashed line with dots at each datapoint
plt.plot(x, y, '--o', label='Line with Dots')

# Adding 'x' marker on specified coordinates with red color
plt.plot(x_mark, y_mark, 'rx', markersize=10, label='Mark')  # 'r' specifies red color for the 'x' marker


plt.ylabel('p95 latency (ms)')  # Add label for the x-axis
plt.xlabel('batch size')  # Add label for the y-axis
plt.title('BERT tiny @ 2000 QPS on one V100 GPU')  # Add a title

plt.show()  # Display the plot