import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime




def plot():


    num_gpus_es = [16, 10, 5, 4, 3, 2, 1]
    num_gpus_baseline = [16, 6, 3, 1]
    acc_es = [0.851552565, 0.850846681, 0.849637396, 0.848940472, 0.847615679, 0.845617039, 0.842029842]
    acc_baseline = [0.848625, 0.8365, 0.8348125, 0.824]

    # Plot the latency CDF
    plt.figure(figsize=(8, 6))
    plt.plot(num_gpus_es, acc_es, marker='.', label="Ensemble serve")
    plt.plot(num_gpus_baseline, acc_baseline, marker='.', label="Single BERT model")

    # add arrow
    # arrow_y = 0.848625
    # plt.annotate("",
    #              xy=(16, arrow_y), xycoords='data',
    #              xytext=(1.6, arrow_y), textcoords='data',
    #              arrowprops=dict(arrowstyle="<->",
    #                              connectionstyle="arc3"))
    # plt.text(7, arrow_y - 0.001, "10x", horizontalalignment='center')  # Adjusts the text position as needed

    plt.legend()
    plt.title("Cost comparison to achieve throughput")
    plt.xlabel("Number of GPUs required")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()



def plot_11_16(slo):
    if slo == 0.03:
        num_gpus_es = [1]
        num_gpus_baseline = [4, 1]
        acc_es = [0.849723192513751]
        acc_baseline = [0.848625, 0.8365]

    elif slo == 0.05:
        num_gpus_es = [1]
        num_gpus_baseline = [3, 1]
        acc_es = [0.8515325397418753]
        acc_baseline = [0.848625, 0.8365]

    else:
        assert False

    # Plot the latency CDF
    plt.figure(figsize=(6, 4))
    marker_size = 100
    plt.grid(True)
    plt.scatter(num_gpus_es, acc_es, marker='.', label="Ensemble serve", s=[marker_size]*len(num_gpus_es))
    plt.scatter(num_gpus_baseline, acc_baseline, marker='.', label="Single BERT model", s=[marker_size]*len(num_gpus_es))

    # add arrow
    # arrow_y = 0.848625
    # plt.annotate("",
    #              xy=(16, arrow_y), xycoords='data',
    #              xytext=(1.6, arrow_y), textcoords='data',
    #              arrowprops=dict(arrowstyle="<->",
    #                              connectionstyle="arc3"))
    # plt.text(7, arrow_y - 0.001, "10x", horizontalalignment='center')  # Adjusts the text position as needed

    plt.legend(loc="lower right")
    # plt.title("")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Accuracy")
    x_ticks = np.arange(min(min(num_gpus_es), min(num_gpus_baseline)),
                        max(max(num_gpus_es), max(num_gpus_baseline)) + 1, dtype=int)
    plt.xticks(x_ticks)

    plt.show()

if __name__ == "__main__":
    plot_11_16(0.03)