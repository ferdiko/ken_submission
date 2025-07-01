"""
Single model, single batch size
"""
from matplotlib import pyplot as plt

import sys
sys.path.append("../")
from offline.batch_size.latency_simulator import LatencySimulator
from offline.helpers import Gear


def grid_search(models, batch_sizes, num_gpus, workload, cost, p_lat, slo_lat):
    # init simulator
    lso = LatencySimulator(batch_size_profiles=cost,
                           p_latency=p_lat,
                           pred_dir=workload)

    # gid search
    chosen_acc = []
    chosen_num_gpus = []
    chosen_lat = []
    chosen_model = []
    chosen_batch_size = []

    for m in models:
        for bs in batch_sizes:
            for n in num_gpus:
                g = Gear(qps=-1, models=[m], threshs=[0.5], num_gpus=1)
                g.gpu_models = [[m]]
                g.gpu_batch_sizes = [[bs]]
                lso.reset()
                # lat = lso.simulate_samples(g, gpu_idx=0, num_samples=5000)
                lat = lso.simulate_static(m, bs, n)
                acc = lso.get_acc()

                if lat < slo_lat:
                    print("lat {}, {}, bs {}, acc {}, gpus {}".format(lat, m, bs, acc, n))
                    chosen_acc.append(acc)
                    chosen_lat.append(lat)
                    chosen_num_gpus.append(n)
                    chosen_model.append(m)
                    chosen_batch_size.append(bs)
                    break

    # TODO: Return acc
    return chosen_acc, chosen_num_gpus


def grid_search_adapt(num_gpus, workload, cost, p_lat, slo_lat):
    # init simulator
    lso = LatencySimulator(batch_size_profiles=cost,
                           p_latency=p_lat,
                           pred_dir=workload)

    # gid search
    chosen_acc = []
    chosen_num_gpus = []
    chosen_lat = []

    for n in num_gpus:
        lso.reset()
        # lat = lso.simulate_samples(g, gpu_idx=0, num_samples=5000)
        lat = lso.simulate_adapt(n)
        acc = lso.get_acc()

        print("lat {}, acc {}, gpus {}".format(lat, acc, n))

        if lat < slo_lat:
            # print("lat {}, acc {}, gpus {}".format(lat, acc, n))
            chosen_acc.append(acc)
            chosen_lat.append(lat)
            chosen_num_gpus.append(n)

    # TODO: Return acc
    return chosen_acc, chosen_num_gpus


if __name__ == "__main__":
    # define paths
    cost_path = "../simulator/profiling/twitter/twitter_bert_prof_bs100.json"
    pred_dir = "../workloads/twitter/twitter_wlsf5000"

    # define search space
    num_gpus = [i+1 for i in range(8)]
    batch_sizes = [i+1 for i in range(100)]
    models = ["tiny", "mini", "small", "medium", "base"]

    # slo
    p_lat = 0.95
    slo_lat = 0.05

    # make grid search
    acc, num_gpus = grid_search(models=models,
                                batch_sizes=batch_sizes,
                                num_gpus=num_gpus,
                                workload=pred_dir,
                                cost=cost_path,
                                p_lat=p_lat,
                                slo_lat=slo_lat)

    # acc, num_gpus = grid_search_adapt(
    #                             num_gpus=num_gpus,
    #                             workload=pred_dir,
    #                             cost=cost_path,
    #                             p_lat=p_lat,
    #                             slo_lat=slo_lat)


    # visualize
    # TODO
    plt.scatter(num_gpus, acc, color='blue', marker='o')
    plt.title('Scatter plot of FLOPs vs. Accuracy')
    plt.xlabel('FLOPs')
    plt.ylabel('Accuracy')

    plt.show()
