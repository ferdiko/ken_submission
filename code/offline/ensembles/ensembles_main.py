import numpy as np
import json

from offline.ensembles.random_sample import Sampler
from offline.ensembles.train_rl import train_policy_gradients
from simulator.simulator_old import Simulation
import matplotlib.pyplot as plt


def sorted_pareto_front(ensembles, thresholds, flop, acc):

    # TODO: Rerun and get throughput (1/average inference time) and latency (sum of model inferences)

    pareto_acc = []
    pareto_flop = []
    pareto_models = []
    pareto_threshs = []

    for i in range(len(acc)):
        is_pareto_optimal = True  # assume the current point is Pareto optimal until proven otherwise
        for j in range(len(acc)):
            if i == j:  # don't compare the point to itself
                continue
            # if the other point has higher acc and lower or equal flop, then current point is not Pareto optimal
            if acc[j] > acc[i] and flop[j] <= flop[i]:
                is_pareto_optimal = False
                break
            # if the other point has equal acc and lower flop, then current point is not Pareto optimal
            elif acc[j] == acc[i] and flop[j] < flop[i]:
                is_pareto_optimal = False
                break

        if is_pareto_optimal:
            pareto_acc.append(acc[i])
            pareto_flop.append(flop[i])
            pareto_models.append(ensembles[i])
            pareto_threshs.append(thresholds[i])

    # plt.scatter(pareto_flop, pareto_acc)
    # plt.show()

    # Sort according to descending accuracy
    combined = list(zip(pareto_acc, pareto_flop, pareto_models, pareto_threshs))
    combined.sort(key=lambda x: x[0], reverse=True)
    pareto_acc, pareto_flop, pareto_models, pareto_threshs = zip(*combined)

    return pareto_models, pareto_threshs, pareto_flop, pareto_acc


def finalize_cascades(ensembles, threshs, runtime, accuracy):
    # check if it matters if we take models out:
    # NOTE not impl: we can for sure take those out where second thresh is higher than prev
    # also for sure take out if thresh is 0.0

    all_new_ensembles = []
    all_new_threshs = []
    all_new_runtimes = []
    all_new_accuracies = []
    unique_ensembles = set()

    for ensemble, t, r, a in zip(ensembles, threshs, runtime, accuracy):

        unique_models = set()
        new_ensemble = []
        new_threshs = []

        for idx, e in enumerate(ensemble):
            if e == -1 or t[idx] == 0.0:
                break

            if e not in unique_models:
                unique_models.add(e)
                new_ensemble.append(e)
                new_threshs.append(t[idx])

        if len(new_ensemble) > 0 and (tuple(new_ensemble), tuple(new_threshs)) not in unique_ensembles:
            all_new_ensembles.append(new_ensemble)
            all_new_threshs.append(new_threshs)
            all_new_runtimes.append(r)
            all_new_accuracies.append(a)
            unique_ensembles.add((tuple(new_ensemble), tuple(new_threshs)))

    # plot to check how if that's a good idea
    # sim = Simulation(1)
    # new_accs = []
    # new_flops = []
    # for e, t in zip(all_new_ensembles, all_new_threshs):
    #     sim.models = e
    #     sim.threshs = t
    #
    #     acc, cost = sim.get_goal()
    #     new_accs.append(acc)
    #     new_flops.append(cost)
    #
    #     sim.reset()
    #
    # plt.scatter(new_flops, new_accs, label="new")
    # plt.scatter(flops, accs, label="original")
    #
    # plt.legend()
    # plt.show()

    return all_new_ensembles, all_new_threshs, all_new_runtimes, all_new_accuracies


def model_idx_to_name(models):
    look_up = ["tiny1", "tiny2", "tiny3", "tiny4", "mini", "small", "medium", "base"]
    named_ensembles = []
    for ensemble in models:
        named_ensemble = []
        for e in ensemble:
            named_ensemble.append(look_up[e])
        named_ensembles.append(named_ensemble)

    return named_ensembles

def main(pred_dir, profiling_file, out_file, min_thresh, max_thresh, use_rl=True):
    if use_rl:
        _, acc, rt, models, threshs = train_policy_gradients(max_episodes=400,
                                                    batch_size=10,
                                                    flop_thresh=2.5 * 83920000,
                                                    learning_rate=0.001,
                                                    exploration_prob=0.5,
                                                    hidden_sizes=[128, 256, 128, 64, 32],
                                                    alpha=0.5)

    else:
        sampler = Sampler(pred_dir=pred_dir,
                          profiling_file=profiling_file)

        per_model_cost, acc, rt, models, threshs = sampler.search(num_samples=50000,
                                                     min_thresh=min_thresh,
                                                     max_thresh=max_thresh)

        # handcrafted ensembles
        # _, acc_hand, rt_hand, models_hand, threshs_hand = sampler.hand_crafted()

        # combine
        # models += models_hand
        # threshs += threshs_hand
        # acc += acc_hand
        # rt += rt_hand


    # Delete duplicate models / unreachable models etc
    models, threshs, rt, acc = finalize_cascades(models, threshs, rt, acc)

    # filter out non-pareto optimal
    models, threshs, rt, acc = sorted_pareto_front(models, threshs, rt, acc)

    # models = model_idx_to_name(models)

    # rt = [r / 16000 for r in rt]


    # Write to file
    data = {
        "accs": acc,
        "runtime": rt,
        "ensembles": models,
        "threshs": threshs
    }

    # dump found ensembles
    with open(out_file, 'w+') as f:
        json.dump(data, f, indent=2)

    # visualize RL results
    plt.scatter(rt, acc, color='blue', marker='o')

    # baseline twitter
    acc = [0.81025, 0.824, 0.8348125, 0.8365, 0.848625]
    flops = [0.001641607284546301, 0.0029012203216544287, 0.002983427047730353, 0.005438995361329737, 0.024385404586785012]

    # baseline -- single model
    # flops = np.array([0.1234750747680664, 0.21223187446594238, 0.3815484046936035, 0.918332576751709*2])
    # acc = [0.52, 0.528, 0.612, 0.644]
    #
    # # acc = [0.4808, 0.5166, 0.5676, 0.6186]

    plt.scatter(flops, acc, color='red', marker='o')


    plt.title('Scatter plot of FLOPs vs. Accuracy')
    plt.xlabel('FLOPs')
    plt.ylabel('Accuracy')

    plt.show()


if __name__ == "__main__":
    # Set workload
    workloads = ["hellaswag", "twitter", "wikitext"]
    workload = workloads[1]

    # set pred dir etc
    if workload == "hellaswag":
        # TODO: Use config obect here
        pred_dir = "../../profiling/predictions/hellaswag_5000"
        profiling_file = "../../profiling/models/llama.json"
        out_file = "../cached_runs/hellaswag/ensembles_500.json"
        min_thresh = 0
        max_thresh = 0.3
    elif workload == "twitter":
        # TODO: Use config obect here
        pred_dir = "../../profiling/predictions/twitter"
        profiling_file = "../../profiling/models/bert.json"
        out_file = "../cached_runs/twitter/ensembles_old_preds_new_code2.json"
        min_thresh = 0
        max_thresh = 0.4
    elif workload == "wikitext":
        # TODO: Use config obect here
        pred_dir = "../../profiling/predictions/wikitext5"
        profiling_file = "../../profiling/models/llama.json"
        out_file = "../cached_runs/wikitext/perlim.json"
        min_thresh = 0
        max_thresh = 1.0
    else:
        raise NotImplementedError

    # run search
    main(pred_dir, profiling_file, out_file, min_thresh, max_thresh, use_rl=False)

