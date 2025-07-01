import matplotlib.pyplot as plt

y = [
    0.850275,
    0.8439375,
    0.8439375,
    0.8439375,
    0.8439375,
    0.8439375,
    0.835875,
    0.8256875
]

x = [
    1,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3
]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, '-o',markersize=8)  # '-o' means line with circles

# Add titles and labels
plt.title("Accuracy degradation with decreasing memory size")
plt.xlabel("Memory (relative to 'everything fits')")
plt.ylabel("Accuracy")

# add baelines
line_y_value = 0.848625
line_x_end = 0.939233
plt.plot([1, line_x_end], [line_y_value, line_y_value], 'gray', linestyle='--')
plt.plot(line_x_end, line_y_value, color='gray', marker='|', markersize=12)
# plt.axhline(y=line_y_value, color='gray', linestyle='--')
plt.text(0.7, 0.8495, "BERT base, 1.7x more compute", va='center', ha='right', color='black')

# add baelines
line_y_value = 0.8365
line_x_end = 0.283486
plt.plot([1, line_x_end], [line_y_value, line_y_value], 'gray', linestyle='--')
plt.plot(line_x_end, line_y_value, color='gray', marker='|', markersize=12)
plt.text(0.67, 0.8373, "BERT medium, same amount of GPUs", va='center', ha='right', color='black')



# Display the plot
plt.grid(True)
plt.gca().invert_xaxis()  # Inverting the x-axis as the x values are in descending order
plt.show()