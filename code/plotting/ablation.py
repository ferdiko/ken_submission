import matplotlib.pyplot as plt


def unsim_twitter_1gpu():
    """
    The logs are in the acc_815 dirs in utils and baselines/cocktail.
    The dynamic batching are utils/acc_815/*mini.log, the MS are utils/acc_815/*ms.log.
    Mean QPS: 16'800, max QPS: 80'000, scaling factor 1000
    :return:
    """
    # CascadeServe.
    # 03-12-runs/sf95_n1_qpsmul0.9_maxbs0.log
    # 03-14-runs/sf95_n1_qpsmul1.0_maxbs0_ql4_p0.1_acc84.log
    # 03-13-runs/sf95_n1_qpsmul1.2_maxbs0_ql8_p0.15.log
    # 03-12-runs/sf95_n1_qpsmul1.0_maxbs0_ql8_p01.log
    # # 03-13-runs/sf95_n1_qpsmul0.8_maxbs0_ql8_p0.05.log
    # # 03-13-runs/sf95_n1_qpsmul0.7_maxbs0_ql1_p0.1.log
    # utils/03-13-runs/sf95_n1_qpsmul0.8_maxbs0_ql0_p0.05.log
    our_acc = [0.8382048591068899, 0.8412347169019257, 0.8424599884556856, 0.8454919452169807, 0.8473642231201134, 0.8483963897780343] #0.8482095817809728, 0.849431980211196]
    our_lat = [0.1128225326538086, 0.11727619171142578, 0.12631845474243164, 0.14724469184875488,  0.17662906646728516, 0.24292612075805664] #0.22243094444274902, 10.148216247558594]

    # No Cascades.
    # ablation/sf95_n1_qpsmul7_maxbs0_ql8_p0.1_gr0_singlemodel.log
    # ablation/sf95_n1_qpsmul2_maxbs0_ql8_p0.1_gr0_singlemodel.log
    # ablation/sf95_n1_qpsmul1_maxbs0_ql8_p0.1_gr0_singlemodel.log
    # ablation/sf95_n1_qpsmul0.5_maxbs0_ql8_p0.1_gr0_singlemodel.log
    singlemodel_acc = [0.8365324267782427, 0.840606601248885, 0.8421429542673254, 0.8461418550272798]
    singlemodel_lat = [0.1270890235900879, 0.4041767120361328, 5.849385023117065, 11.928298234939575]

    # No switching.
    # ablation/sf95_n1_qpsmul1_maxbs0_ql8_p0.1_gr6500_statcasc.log
    # ablation/sf95_n1_qpsmul1_maxbs0_ql8_p0.1_gr4500_statcasc.log
    # ablation/sf95_n1_qpsmul1_maxbs0_ql8_p0.1_gr4000_statcasc.log
    # ablation/sf95_n1_qpsmul1_maxbs0_ql8_p0.1_gr3000_statcasc.log
    # ablation/sf95_n1_qpsmul1_maxbs0_ql8_p0.1_gr2000_statcasc.log
    # ablation/sf95_n1_qpsmul1_maxbs0_ql8_p0.1_gr1000_statcasc.log
    statcasc_acc = [0.8239339875111508, 0.8346833184656557, 0.8396300571968306, 0.8461216350947158, 0.8489352993650627, 0.8509152467019485]
    statcasc_lat = [0.08440613746643066, 0.13952422142028809, 0.23229551315307617, 0.3082563877105713, 0.9223766326904297, 16.41694450378418]

    # No cascade, no switching.
    # utils/acc_815/sf1000_n1_qpsmul1_maxbs3000_ql8_p0.1_mini.log
    # utils/acc_815/sf1000_n2_qpsmul1_maxbs0_ql8_p0.1_mini.log
    # 03-17-runs/sf130_n1_qpsmul1.6_maxbs0_ql8_p0.1_medium.log
    # 03-12-runs/base_3gpu_sf95_maxbs0.
    static_acc = [0.8367901234567902, 0.848622278672349]
    static_lat = [0.07129120826721191, 20] # The 20 is not measured (cut off in graph)

    # For plotting
    title = "BERT (1 GPU)"

    # Set limits to plot so 6s baseline latency is cut off.
    # plt.xticks(np.arange(0.1, 0.7, 0.1))
    plt.xlim([0.0, 1.0])

    return our_lat, our_acc, statcasc_lat, statcasc_acc, singlemodel_lat, singlemodel_acc, static_lat, static_acc, title


if __name__ == "__main__":

    plt.figure(figsize=(4,3))

    ours_lat, ours_acc, statcasc_lat, statcasc_acc, singlemodel_lat, singlemodel_acc, static_lat, static_acc, title = unsim_twitter_1gpu()

    accs = [ours_acc, statcasc_acc, singlemodel_acc] #, static_acc]
    lats = [ours_lat, statcasc_lat, singlemodel_lat] #, static_lat]
    labels = ["CascadeServe", "No switching", "No cascade", "No cascades. no switching"]

    for a, g, l in zip(accs, lats, labels):
        plt.scatter(g, a, marker='.', label=l, s=100)
        plt.plot(g, a, linestyle='dashed', linewidth=2)

    plt.legend()

    # plt.xticks(np.arange(1, 4, 1))
    plt.xlabel("p95 latency (s)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()
