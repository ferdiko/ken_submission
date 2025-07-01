import os
import time
from profile_worload import *


def scale_dataset(scale_factor, time_interv, in_dir, out_dir_pref):
    out_dir = out_dir_pref + f"sf{scale_factor}"
    start = time.time()
    qps_freq, qps_bins = get_qps_freq(scale_factor, num_samples=10000)
    with open(f'tmp_qps_freq_sf_{scale_factor}.npy', 'wb') as f:
        np.save(f, qps_freq)

    # with open('tmp_qps_bins.npy', 'wb') as f:
    #     np.save(f, qps_bins)

    print("time:", time.time() - start)

    with open(f'tmp_qps_freq_sf_{scale_factor}.npy', 'rb') as f:
        qps_freq = np.load(f)
        qps_freq = np.array(qps_freq, dtype=float)

    qps_freq /= np.sum(qps_freq)
    qps_samples = np.random.choice(np.linspace(0, qps_freq.shape[0]-1, qps_freq.shape[0], dtype=int), time_interv, p=qps_freq)
    print(qps_samples)


    # read tweets
    os.makedirs(out_dir, exist_ok=True)

    sim_files = ["casc_tiny.csv", "casc_tiny2.csv", "casc_tiny3.csv", "casc_tiny4.csv",
                 "casc_mini.csv", "casc_small.csv", "casc_medium.csv", "casc_base.csv", ]

    # read ground truth
    gt = []
    with open(os.path.join(in_dir, sim_files[0]), "r") as f:
        lines = f.readlines()
        for l in lines:
            gt.append(float(l.split(",")[1]))

    # write over
    for sim_file in sim_files:
        certs = []

        with open(os.path.join(in_dir, sim_file), "r") as f:
            lines = f.readlines()
            for l in lines:
                certs.append(float(l.split(",")[0]))

        # write
        out_file = open(os.path.join(out_dir, sim_file), "w")
        i_sample = 0
        sec = 0
        for qps in qps_samples:
            for q in range(int(qps)):
                timestamp = sec + q/qps
                out_file.write(f"{certs[i_sample]},{gt[i_sample]},{timestamp}\n")
                i_sample = (i_sample + 1) % 16000
            sec += 1
        out_file.close()


if __name__ == "__main__":
    scale_dataset(scale_factor=5000,
                  time_interv=60,
                  in_dir="../../simulator/nn_preds/twitter",
                  out_dir_pref="twitter_wl")