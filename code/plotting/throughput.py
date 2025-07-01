import matplotlib.pyplot as plt

# Data
batch_size = [2, 4, 8, 16, 32, 64, 128]
throughput = [561.4864793, 1198.715061, 1680.326105, 2152.442876, 2624.464285, 2609.158609, 2626.981289]

# Create plot
# plt.figure(figsize=(10, 6))
plt.plot(batch_size, throughput, marker='o', linestyle='-')
plt.xscale('log', base=2)  # Set x-axis to logarithmic scale with base 2
plt.xlabel('Batch Size (log scale)')
plt.ylabel('Throughput')
plt.title('Throughput vs Batch Size (BERT Tiny on V100 GPU)')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.xticks(batch_size, batch_size)  # Display actual batch sizes on the x-axis

# Display the plot
plt.tight_layout()
plt.show()