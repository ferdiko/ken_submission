import json
import time
from torch.profiler import profile, record_function, ProfilerActivity
from twitter_models import *


def check_memory():
    print('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))

def profile_model_memory(model, prep):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load(device)

    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    print("MEMORY:", mem)

    # x, x_att = prep.prep(["sample one",
    #                       "sample two",
    #                       "sample three",
    #                       "sample four",
    #                       "sample five",
    #                       "sample six",
    #                       "sample seven",
    #                       "sample eight",
    #                       ])
    # if device == 'cuda':
    #     x = x.cuda()
    #     x_att = x_att.cuda()
    #     # with profile(activities=[ProfilerActivity.CUDA],
    #     #         profile_memory=True, record_shapes=True) as prof:
    #
    #     for _ in range(5):
    #         model.forward_direct(x, x_att)
    #         check_memory()
    #
    # else:
    #     with profile(activities=[ProfilerActivity.CPU],
    #             profile_memory=True, record_shapes=True) as prof:
    #         model.forward_direct(x, x_att)
    #
    #     print(m, "profile:")
    #     print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))


def profile_model_runtimes(models, prep, max_bs=64, num_runs=5):
    print(f"MEASURE: {models.keys()} | MAX BS {max_bs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    profile_dict = {}

    for name in models.keys():
        # init dict
        profile_dict[name] = {}

        # load model
        model = models[name]
        model.load(device)

        # run for different bs
        for bs in range(1, max_bs+1):
            x, x_att = prep.prep(["sample one, sample one, sample one, sample one"]*bs)

            if device == 'cuda':
                # tensors to GPU and warm up
                x = x.cuda()
                x_att = x_att.cuda()
                model.forward_direct(x, x_att)

                # run
                start = time.time()
                for _ in range(num_runs):
                    model.forward_direct(x, x_att)
                end = time.time()

                # output and insert into dict
                runtime = (end - start) / num_runs
                print("model {} bs {}: {}s | per sample: {}".format(name, bs, runtime, runtime/bs))
                profile_dict[name][bs] = runtime

        with open("model_profile_bs.json", "w+") as f:
            json.dump(profile_dict, f)





if __name__ == "__main__":
    machine = 'supercloud' if torch.cuda.is_available() else 'macbook'
    models, prep = get_model_dict(machine)

    profile_model_runtimes(models, prep, max_bs=4)

    #
    # for m in models.keys():
    #     print(m)
    #     profile_model_memory(models[m], prep)

    memory_usage = {
        "tiny1":    17552904,
        "tiny2": 17552904,
        "tiny3": 17552904,
        "tiny4": 17552904,
        "mini":    44692488,
        "small":  115066888,
        "medium": 165505032,
        "base":   437943304,
        "large": 1340583944
    }

    runtimes_bs8 = {
        "tiny1":    0.001615285873413086,
        "tiny2": 0.001615285873413086,
        "tiny3": 0.001615285873413086,
        "tiny4": 0.001615285873413086,
        "mini":    0.002736330032348633,
        "small":   0.002772808074951172,
        "medium":  0.005045175552368164,
        "base":    0.007277488708496094,
        "large":   0.013909578323364258
    }

    profile_dict = {
        "memory": memory_usage,
        "runtime_bs8": runtimes_bs8
    }

    # Serialize the dictionary to JSON
    # json_memory_usage = json.dumps(profile_dict, indent=4)
    #
    # # Write to a file
    # with open("../../offline/profile/bert_profile.json", "w") as json_file:
    #     json_file.write(json_memory_usage)
