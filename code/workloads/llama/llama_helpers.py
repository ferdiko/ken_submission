import glob
import os


def get_config(model_dir, gpu_split=None):
    """
    Get a default ExLlamaConfig given the model dict (and gpu split).
    :param model_dir: Directory to model
    :param gpu_split: Str list of VRAM (in GB) to use per GPU device for model layers, e.g. 20,7,7
    :return:
    """
    from model import ExLlamaConfig

    # Model paths.
    config_path = os.path.join(model_dir, "config.json")
    st_pattern = os.path.join(model_dir, "*.safetensors")
    st = glob.glob(st_pattern)
    assert len(st) > 0, f"No files matchin {st_pattern}"

    # Create config object.
    config = ExLlamaConfig(config_path)
    config.model_path = st

    # Set to default values.
    config.max_seq_len = 2048
    config.compress_pos_emb = 1.0
    config.set_auto_map(gpu_split)
    config.gpu_peer_fix = False # Disallow gpu-to-gpu communication
    config.alpha_value = 1.0
    config.calculate_rotary_embedding_base()

    # if args.flash_attn:
    #     config.use_flash_attn_2 = True
    #     try:
    #         config.max_input_len = int(args.flash_attn)
    #     except ValueError:
    #         pass

    config.matmul_recons_thd = 8
    config.fused_mlp_thd = 2
    config.sdp_thd = 8
    config.matmul_fused_remap = False
    config.fused_attn = True

    config.rmsnorm_no_half2 = False
    config.rope_no_half2 = False
    config.matmul_no_half2 = False
    config.silu_no_half2 = False
    config.concurrent_streams = False

    # if args.theta:
    #     config.rotary_embedding_base = args.theta

    return config


def get_hellaswag(num_lines=5):
    import json
    
    file_path = os.path.join(os.path.dirname(__file__), "hellaswag/data/hellaswag_train.jsonl")
    with open(file_path, "r") as f:
        lines = f.readlines()

    cnt = 0
    tasks = []
    labels = []
    for json_line in lines:
        # Parse JSON line
        parsed_json = json.loads(json_line)

        # Extract desired fields
        labels.append(parsed_json['label'])
        tasks.append((parsed_json['ctx'], parsed_json['endings']))

        cnt += 1
        if cnt >= num_lines:
            break

    return tasks, labels

if __name__ == "__main__":
    print(get_hellaswag(1))
