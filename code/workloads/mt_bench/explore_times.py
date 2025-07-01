import json

with open("batch_2111/qps0.1_queue1200.0_cert0.43_i3_3b_70b_2x600_42s_times.json", "r") as f:
    data = json.load(f)


for d, v in data.items():
    # if v[-1] - v[0] > 100:
    print(len(v))