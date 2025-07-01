import matplotlib.pyplot as plt
import numpy as np


def unsim_twitter_acc815():
    """
    The logs are in the acc_815 dirs in utils and baselines/cocktail.
    The dynamic batching are utils/acc_815/*mini.log, the MS are utils/acc_815/*ms.log.
    Mean QPS: 16'800, max QPS: 80'000, scaling factor 1000
    :return:
    """
    # CascadeServe.
    # utils/acc_815/sf1000_n1_qpsmul1_maxbs0_ql8_p0.05_mini15k_tiny.log
    our_acc = [0.8150462034739454]
    our_lat = [0.07627725601196289]
    our_gpus = [1]

    # Model switching.
    # NOTE: QPS < 15k: mini; Else: Tiny
    # utils/acc_815/sf1000_n1_qpsmul1_maxbs3000_ql8_p0.1_ms.log
    # utils/acc_815/sf1000_n2_qpsmul1_maxbs3000_ql8_p0.1_ms.log
    # utils/acc_815/sf1000_n3_qpsmul1_maxbs3000_ql8_p0.1_ms.log
    ms_acc = [0.815211754194381, 0.8152882382133995, 0.8152521091811414]
    ms_lat = [0.4707987308502197, 0.0740509033203125, 0.08017516136169434]
    ms_gpus = [1, 2, 3]

    # Cocktail.
    # baselines/cocktail/acc_815/max_20_10000.log -- 3.3 GPUs
    # baselines/cocktail/acc_815/max_5_17500.log -- 2 GPUs
    cocktail_acc = [0.8240621653389055]*2
    cocktail_lat = [0.08808732032775879, 18.782505989074707]
    cocktail_gpus = [3.3136556819923992, 2.015989473302566]

    # Dynamic batching.
    # utils/acc_815/sf1000_n1_qpsmul1_maxbs3000_ql8_p0.1_mini.log
    # utils/acc_815/sf1000_n2_qpsmul1_maxbs0_ql8_p0.1_mini.log
    # utils/acc_815/sf1000_n3_qpsmul1_maxbs0_ql8_p0.1_mini.log
    # utils/acc_815/sf1000_n4_qpsmul1_maxbs0_ql8_p0.1_mini.log
    static_acc = [0.8240019851116626]*4
    static_lat = [103.02046513557434, 0.10767006874084473, 0.08131003379821777, 0.08390355110168457]
    static_gpus = [1, 2, 3, 4]

    # For plotting
    title = "BERT (SLO: Accuracy $\geq$ 81.5%)"
    ylabel = "p95 Latency (s)"
    switch_ax = False

    # Set limits to plot so 6s baseline latency is cut off.
    plt.yticks(np.arange(0.1, 0.7, 0.1))
    plt.ylim([0.05, 0.65])

    return our_gpus, our_lat, cocktail_gpus, cocktail_lat, ms_gpus, ms_lat, static_gpus, static_lat, title, ylabel, switch_ax


def unsim_twitter_acc84():
    """
    Mean QPS: 1600, max QPS: 7600, scaling factor 95.
    :return:
    """
    # CascadeServe.
    # utils/03-14-runs/sf95_n1_qpsmul1.0_maxbs0_ql4_p0.1_acc84.log
    # utils/03-14-runs/sf95_n2_qpsmul1.0_maxbs0_ql4_p0.1_acc84.log
    our_acc = [0.8412740725192842, 0.8412347169019257]
    our_lat = [0.08636689186096191, 0.11727619171142578]
    our_gpus = [2, 1]

    # Cocktail.
    # baselines/cocktail/03-13-runs/max_5_900.log -- 3.2 GPUs
    # baselines/cocktail/03-13-runs/max_10_1100.log -- 2.9 GPUs
    # baselines/cocktail/03-13-runs/max_5_1100.log -- 2.4 GPUs
    cocktail_acc = [0.8486199296846303, 0.8486199296846303, 0.8486199296846303]
    cocktail_lat = [0.08348751068115234, 0.1824958324432373, 0.6379680633544922]
    cocktail_gpus = [3.1865312296802273, 2.8894905127341817, 2.432020535532354]

    # Model switching.
    # utils/ms/n2_sb100_sf1.2.log
    # utils/ms/n3_sb100_sf1.0.log
    # utils/ms/n4_sb100_sf1.3.log
    ms_acc = [0.8408488567401268, 0.8440373615994123, 0.840849031851813]
    ms_lat = [9.598909139633179, 0.13622689247131348, 0.1291821002960205]
    ms_gpus = [2, 3, 4]


    # Dynamic batching.
    # utils/03-12-runs/base_4gpu_sf95_maxbs0.log
    # utils/03-12-runs/base_3gpu_sf95_maxbs0.log
    # The 6s one I remember it was around there, not sure where log file (this is cut off in plot).
    static_acc = [0.8486199296846303]*3
    static_lat = [0.08551502227783203, 0.49612975120544434, 6]
    static_gpus = [4, 3, 2]

    # For plotting
    title = "BERT (SLO: Accuracy $\geq$ 84%)"
    ylabel = "p95 Latency (s)"
    switch_ax = False

    # Set limits to plot so 6s baseline latency is cut off.
    plt.yticks(np.arange(0.1, 0.7, 0.1))
    plt.ylim([0.05, 0.65])

    return our_gpus, our_lat, cocktail_gpus, cocktail_lat, ms_gpus, ms_lat, static_gpus, static_lat, title, ylabel, switch_ax


def unsim_twitter_slo250ms():
    # CascadeServe.
    # utils/03-13-runs/sf95_n2_qpsmul0.4_maxbs0_ql4_p0.1.log
    # utils/03-13-runs/sf95_n1_qpsmul0.8_maxbs0_ql0_p0.05.log
    our_acc = [0.8512069055989925, 0.8483963897780343]
    our_lat = [0.19145870208740234, 0.24292612075805664]
    our_gpus = [2, 1]

    # Cocktail.
    # baselines/cocktail/03-13-runs/max_10_1100.log
    # Couldn't get another good data point, so we take medium model on 1 GPU.
    cocktail_acc = [0.8486199296846303, 0.8364855876347161]
    cocktail_lat = [0.1824958324432373, -1]
    cocktail_gpus = [2.8894905127341817, 1]

    # Model switching.
    # First two: Couldn't get good data points so I take medium dynamic batching.
    # utils/ms/n3_sb100_sf0.4.log -- If I do slack factor 0.3: latency 1.854640245437622, acc 0.8485282595590135
    # utils/ms/n4_sb100_sf0.1.log
    ms_acc = [0.8364855876347161]*2 + [0.8481366427034686, 0.8486199296846303]
    ms_lat = [0.08808732032775879] *2 + [0.16296839714050293, 0.13570165634155273]
    ms_gpus = [1, 2, 3, 4]


    # Dynamic batching.
    # Medium: Puzzled together from different runs. Latency is not from actual measurement (it's not plotted).
    # utils/03-12-runs/base_4gpu_sf95_maxbs0.log
    # ablation/sf95_n1_qpsmul1_maxbs0_ql8_p0.1_gr0_medium.log
    # The latencies > 1 GPU are not from measurements but eye-balled (not plotted).
    static_acc = [0.8364855876347161]*3 + [0.8486350464442367]
    static_lat = [0.1398916244506836] + [0.071]*3
    static_gpus = [1,2,3,4]

    # For plotting
    title = "BERT (SLO: p95 Latency $\leq$ 250ms)"
    ylabel = "Accuracy"
    switch_ax = False

    return our_gpus, our_acc, cocktail_gpus, cocktail_acc, ms_gpus, ms_acc, static_gpus, static_acc, title, ylabel, switch_ax

def unsim_twitter_slo150ms():
    """
    This was an old implementation. Both us and the baseline were weaker
    because of some implementation detail. Wouldn't use that.
    :return:
    """
    # CascadeServe. TODO: I thought we have a better one for 1 gpu?
    # utils/03-13-runs/sf95_n2_qpsmul0.5_maxbs0_ql2_p0.05.log
    # utils/03-12-runs/sf95_n1_qpsmul0.5_maxbs0_ql8.log
    our_acc = [0.8507519546623288, 0.8454882720260272]
    our_lat = [0.13817477226257324, 0.14917397499084473]
    our_gpus = [2, 1]

    # Cocktail.
    # I couldn't get a better data point than medium 1 GPU.
    # baselines/cocktail/03-13-runs/max_5_700.log
    cocktail_acc = [0.8486199296846303, 0.8364855876347161]
    cocktail_lat = [0.08512496948242188, -1]
    cocktail_gpus = [3.185980669769923, 1]

    # Model switching.
    # 1 and 2 GPU, I couldn't get better data points than medium.
    # utils/ms/n3_sb100_sf0.5.log
    # utils/ms/ms/n4_sb100_sf0.1.log
    ms_acc = [0.8364855876347161]*2 + [0.8477220968672928, 0.8486199296846303]
    ms_lat = [0.08808732032775879] *2 + [0.1467750072479248, 0.13570165634155273]
    ms_gpus = [1, 2, 3, 4]

    # Dynamic batching.
    # Medium: Puzzled together from different runs. Latency is not from actual measurement (it's not plotted).
    # utils/03-12-runs/base_4gpu_sf95_maxbs0.log
    static_acc = [0.8364855876347161]*3 + [0.8486350464442367]
    static_lat = [0.057619333267211914]*5 # TODO: This is not from actual measurements.
    static_gpus = [1,2,3,4]

    # For plotting
    title = "BERT (SLO: p95 Latency $\leq$ 150ms)"
    ylabel = "Accuracy"
    switch_ax = False

    return our_gpus, our_acc, cocktail_gpus, cocktail_acc, ms_gpus, ms_acc, static_gpus, static_acc, title, ylabel, switch_ax


def unsim_hellaswag_acc54():
    # scaling factor 0.5
    # our_acc = [0.5383615084525357, 0.5344603381014305, 0.5383615084525357, 0.5331599479843954]
    # our_lat = [0.7173137664794922, 0.6690337657928467, 0.6693286895751953, 0.6737651824951172]

    # static_lat = [1.362476110458374, 0.5234737396240234]
    # static_acc = [0.6150845253576073]*len(static_lat)
    # static_gpus = [2, 4]

    # scaling factor 1
    # TODO: Those results are with eval script scale factor 0.5 still (so only first 789 samples)
    #  Need to rerun ours with SF 1 because after some time we get error bc QPS > max gear
    static_acc = [0.6150845253576073, 0.6150845253576073, 0.6150845253576073]
    static_lat = [3.766345500946045, 1.2873287200927734, 0.6814343929290771] #, 0.4874582290649414]
    static_gpus = [2, 4, 6] #, 8]

    our_acc = [0.5370611183355006, 0.5435630689206762, 0.5474642392717816, 0.5448634590377113]
    our_lat = [1.327214002609253, 0.6312580108642578, 0.6403822898864746] #, 0.650501012802124]
    our_gpus = [2, 4, 6] #, 8]

    # our_lat = [5.343698263168335]
    # our_gpus = [2]
    # our_acc = [0.6280884265279584]

    # For plotting
    title = "Llama-HellaSwag (SLO: Accuracy $\geq$ 0.54)"
    title = "SLO: Accuracy $\geq$ 0.54"

    ylabel = "p95 Latency (s)"
    switch_ax = True

    return our_gpus, our_lat, static_gpus, static_lat, title, ylabel, switch_ax


def unsim_hellaswag_acc625():
    our_acc = [0.6267880364109233, 0.6306892067620286, 0.6345903771131339] #, 0.62890625]
    our_lat = [6.21174955368042, 3.3702621459960938, 2.8345141410827637] #, 2.8716819286346436]
    our_gpus = [2, 4, 6] #, 8]

    # our_lat = [5.343698263168335]
    # our_gpus = [2]
    # our_acc = [0.6280884265279584]

    static_lat = [5.684344291687012, 4.370205402374268, 3.321794033050537, 3.0923218727111816, 2.8590247631073]
    static_acc = [0.6488946684005201]*len(static_lat)
    static_gpus = [8, 10, 12, 14, 16]

    # For plotting
    title = "Llama-HellaSwag (SLO: Accuracy $\geq$ 0.625)"
    title = "SLO: Accuracy $\geq$ 0.625"

    ylabel = "p95 Latency (s)"
    switch_ax = True

    return our_gpus, our_lat, static_gpus, static_lat, title, ylabel, switch_ax

def unsim_hellaswag_3s():
    our_acc = [0.6514954486345904, 0.635890767230169] # [0.6501950585175552,
    our_lat = [2.8699049949645996, 2.5917491912841797] # [3.0218024253845215,
    our_gpus = [4, 2 ] #, 6]

    static_acc = [0.6150845253576073]*7 + [0.6488946684005201]
    static_lat = [-1, -1, 2.8590247631073]
    static_gpus = [2, 4, 6, 8, 10, 12, 14, 16]

    # For plotting
    title = "Llama-HellaSwag (SLO: p95 Latency $\leq$ 3s)"
    title = "SLO: p95 Latency $\leq$ 3s"

    ylabel = "Accuracy"
    switch_ax = False
    return our_gpus, our_acc, static_gpus, static_acc, title, ylabel, switch_ax

def unsim_hellaswag_5s():
    our_acc = [0.6514954486345904, 0.6488946684005201]
    our_lat = [2.8699049949645996, 4.835216045379639]
    our_gpus = [4, 2 ]

    static_acc = [0.6150845253576073]*4 + [0.6488946684005201]
    static_lat = [-1.0]*4 + [4.370205402374268]
    static_gpus = [2, 4, 6, 8, 10]

    # For plotting
    title = "Llama-HellaSwag (SLO: p95 Latency $\leq$ 5s)"
    title = "SLO: p95 Latency $\leq$ 5s"

    ylabel = "Accuracy"
    switch_ax = False
    return our_gpus, our_acc, static_gpus, static_acc, title, ylabel, switch_ax


def sim_wikitext():
    our_acc = [0.9064430714916152, 0.9196822594880847, 0.9240953221535746, 0.9232127096204766]
    our_gpus = [2, 4, 6, 8]

    static_acc = [0.8305383936451898, 0.852603706972639, 0.9232127096204766]
    static_gpus = [3, 5, 10]
    return our_gpus, our_acc, static_gpus, static_acc


if __name__ == "__main__":

    plt.figure(figsize=(4,3))

    our_gpus, our_acc, cocktail_gpus, cocktail_acc, ms_gpus, ms_acc, static_gpus, static_acc, title, ylabel, switch_ax = unsim_twitter_acc84()

    accs = [our_acc, cocktail_acc, ms_acc, static_acc]
    gpus = [our_gpus, cocktail_gpus, ms_gpus, static_gpus]
    labels = ["CascadeServe", "Cocktail+", "MS+", "Dynamic batching"]

    # cnt = 0
    for a, g, l in zip(accs, gpus, labels):
        plt.scatter(g, a, marker='.', label=l, s=100)
        # if cnt > 0:
        #     plt.plot(g[:-1], a[:-1], linestyle='dashed', linewidth=2)
        # else:
        plt.plot(g, a, linestyle='dashed', linewidth=2)
        # cnt += 1

    plt.legend()

    plt.xticks(np.arange(1, 4, 1))
    plt.xlabel("Number of GPUs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if switch_ax:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
