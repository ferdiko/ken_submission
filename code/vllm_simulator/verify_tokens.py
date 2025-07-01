"""Script to verify tokens are the same for each forward pass and request between vllm and simulator."""
import utils.token_verifier as token_verifier
import utils.verification_helpers as verification_helpers

import random
from utils import configure_logging

# Initialize logging.
print("Start configure_logging")
logger = configure_logging()
logger.info("End configure_logging.")

################################
## Testing simulator portion. ##
################################
# # Initialize a token verifier
# logger.info("Start initializing TokenVerifier.")
# vllm_profile_path = "outputs/profiles/llama-70b_L40S.json"
# verifier = token_verifier.TokenVerifier(vllm_profile_path=vllm_profile_path)
# logger.info("Finish initializing TokenVerifier.")

# # Define request specs.
# in_tokens = [10, 11, 12]
# out_tokens = [1, 1, 1]
# queries_per_iter = [1, 1, 1]
# query_arrival_timestamps = [0, 1, 2]
# query_ids = ['7', '8', '9']

# # Run the simulator. Outputs to outputs/verification/simulator_event_tokens.json.
# verifier._run_simulator(in_tokens, out_tokens, queries_per_iter, query_ids, query_arrival_timestamps, )

WORKLOADS = {
    "5_sequential_100_in": {
        "in_tokens": [100 for _ in range(5)],
        "out_tokens": [10 for _ in range(5)],
        "queries_per_iter": [1 for _ in range(5)],
    },
    "5_sequential_1000_in": {
        "in_tokens": [1000 for _ in range(5)],
        "out_tokens": [200 for _ in range(5)],
        "queries_per_iter": [1 for _ in range(5)],
    },
    "10_sequential_1000_in": {
        "in_tokens": [1000 for _ in range(10)],
        "out_tokens": [200 for _ in range(10)],
        "queries_per_iter": [1 for _ in range(10)],
    },
    "5_concurrent_100_in": {
        "in_tokens": [100 for _ in range(5)],
        "out_tokens": [10 for _ in range(5)],
        "queries_per_iter": [5],
    },
    "5_concurrent_1000_in": {
        "in_tokens": [1000 for _ in range(5)],
        "out_tokens": [200 for _ in range(5)],
        "queries_per_iter": [5],
    },
    "10_concurrent_1000_in": {
        "in_tokens": [1000 for _ in range(10)],
        "out_tokens": [200 for _ in range(10)],
        "queries_per_iter": [10],
    },
    # AzurePublicDataset first 30 requests from workloads. Link: https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md
    "30_concurrent_azure_code": {
        "in_tokens": [4808, 3180, 110, 7433, 34, 374, 6985, 34, 1145, 201, 137, 7427, 1555, 3893, 1827, 394, 675, 7436, 158, 6587, 763, 1556, 5108, 159, 458, 2464, 4009, 1632, 2567, 730],
        "out_tokens": [10, 8, 27, 14, 12, 14, 9, 23, 7, 24, 9, 8, 19, 19, 10, 17, 6, 9, 26, 18, 8, 18, 12, 127, 67, 30, 51, 9, 45, 36],
        "queries_per_iter": [30],
    },
    "30_concurrent_azure_conversation": {
    "in_tokens": [374, 396, 879, 91, 91, 381, 1313, 388, 242, 209, 394, 394, 1315, 2221, 389, 415, 120, 369, 206, 1353, 197, 181, 388, 4085, 2584, 203, 126, 389, 2548, 91],
    "out_tokens": [44, 109, 55, 16, 16, 84, 142, 84, 14, 152, 124, 59, 174, 15, 90, 106, 12, 74, 162, 142, 152, 154, 54, 62, 170, 147, 194, 87, 116, 16],
    "queries_per_iter": [30],
    },
}

##################################
## Basic testing of comparison. ##
##################################
# Initialize a token verifier
logger.info("Start initializing TokenVerifier.")
# Replicating vLLM run for results.
verifier = token_verifier.TokenVerifier(
    model_name="debug_8b",
    model= "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_tokens=2000,
    max_num_batched_tokens=512,
    max_model_len=8192,
    max_num_seqs= 512)
logger.info("Finish initializing TokenVerifier.")

# Define request specs.
for workload_name in WORKLOADS.keys():
    # workload_name = "5_sequential_1000_in" # passed.
    in_tokens = WORKLOADS[workload_name]["in_tokens"]
    out_tokens = WORKLOADS[workload_name]["out_tokens"]
    queries_per_iter = WORKLOADS[workload_name]["queries_per_iter"]
    # Run the requests and write to outputs/verification/{vllm, simulator}_event_tokens.json.
    assert verifier.verify_tokens_per_forward_pass(in_tokens, out_tokens, queries_per_iter), "Vllm and simulator tokens do not match!!"

# # ###################################
# # ## Simulataneous large requests. ##
# # ###################################
# # Initialize a token verifier
# logger.info("Start initializing TokenVerifier.")
# verifier = token_verifier.TokenVerifier(
#     max_tokens=100,
#     max_num_batched_tokens=5000,
#     max_model_len=2000)
# logger.info("Finish initializing TokenVerifier.")

# # Use a 5 to 1 ratio of in to out tokens.
# num_requests = 10
# queries_per_iter = [num_requests]
# in_tokens = []
# out_tokens = []
# for _ in range(num_requests):
#     in_tokens.append(500)
#     out_tokens.append(100)

# # Run the requests and write to outputs/verification/{vllm, simulator}_event_tokens.json.
# assert verifier.verify_tokens_per_forward_pass(in_tokens, out_tokens, queries_per_iter), "Vllm and simulator tokens do not match!!"

# # ####################################
# # ## Larger simultaneous requests. ##
# # ####################################
# # Initialize a token verifier
# logger.info("Start initializing TokenVerifier.")
# verifier = token_verifier.TokenVerifier(
#     max_tokens=200,
#     max_num_batched_tokens=5000,
#     max_model_len=2000)
# logger.info("Finish initializing TokenVerifier.")

# # Dump profile to json file for debugging purposes.
# verifier.dump_vllm_profile("outputs/profiles/large_batch.json")

# # Use a 5 to 1 ratio of in to out tokens.
# num_requests = 10
# queries_per_iter = [num_requests]
# in_tokens = []
# out_tokens = []
# for _ in range(num_requests):
#     in_tokens.append(1000)
#     out_tokens.append(200)

# # Run the requests and write to outputs/verification/{vllm, simulator}_event_tokens.json.
# assert verifier.verify_tokens_per_forward_pass(in_tokens, out_tokens, queries_per_iter), "Vllm and simulator tokens do not match!!"

#######################################################
## Simulataneous larger requests with simulator only. ##
#######################################################
# vllm_profile_path = "outputs/profiles/large_batch.json"
# verifier = token_verifier.TokenVerifier(vllm_profile_path=vllm_profile_path, max_num_batched_tokens=5000)

# # Define request specs.
# num_requests = 10
# in_tokens = [1007, 1001, 1009, 1005, 978 + 32, 1007, 1002, 1009, 995, 951 + 71]
# out_tokens = [200 for _ in range(num_requests)]
# queries_per_iter = [num_requests]
# query_arrival_timestamps = [0 for _ in range(num_requests)] # This does not matter.
# query_ids = []
# for i in range(20, 30):
#     query_ids.append(f"{i}")

# # Run the simulator. Outputs to outputs/verification/simulator_event_tokens.json.
# verifier._run_simulator(in_tokens, out_tokens, queries_per_iter, query_ids, query_arrival_timestamps)
# # Then, visually check similarity for previous discrepancy

# ####################################
# ## Mixed simultaneous requests. ##
# ####################################
# # Initialize a token verifier
# logger.info("Start initializing TokenVerifier.")
# verifier = token_verifier.TokenVerifier(
#     max_tokens=200,
#     max_num_batched_tokens=5000,
#     max_model_len=2000)
# logger.info("Finish initializing TokenVerifier.")

# # Dump profile to json file for debugging purposes.
# verifier.dump_vllm_profile("outputs/profiles/large_batch.json")

# # Use a 5 to 1 ratio of in to out tokens. Randomly add large and small requests with 1:1 ratio.
# num_requests = 10
# queries_per_iter = [num_requests]
# in_tokens = []
# out_tokens = []
# for _ in range(num_requests):
#     if random.random() < 0.5:
#         # Add a large request.
#         in_tokens.append(1000)
#         out_tokens.append(200)
#     else:
#         # Add a small request.
#         in_tokens.append(500)
#         out_tokens.append(100)


# # Run the requests and write to outputs/verification/{vllm, simulator}_event_tokens.json.
# assert verifier.verify_tokens_per_forward_pass(in_tokens, out_tokens, queries_per_iter), "Vllm and simulator tokens do not match!!"

# ###############################
# ## n random large requests  ##
# ###############################
# # Initialize a token verifier
# logger.info("Start initializing TokenVerifier.")
# max_model_len = 2000
# verifier = token_verifier.TokenVerifier(
#     max_tokens=200,
#     max_num_batched_tokens=5000,
#     max_model_len=max_model_len)
# logger.info("Finish initializing TokenVerifier.")

# # Dump profile to json file for debugging purposes.
# verifier.dump_vllm_profile("outputs/profiles/large_batch.json")

# # Use a 5 to 1 ratio of in to out tokens. Randomly add large and small requests with 1:1 ratio.
# num_requests = 40
# queries_per_iter = [num_requests]
# in_tokens = []
# out_tokens = []
# for _ in range(num_requests):
#     x = random.randint(500, int((max_model_len // 1.2) - 100))  # Generate a random in_token. Upper limit ensures in + out < max_model_len (even with randmoness of prompt generation).
#     in_tokens.append(x)
#     out_tokens.append(x // 5)

# # Run the requests and write to outputs/verification/{vllm, simulator}_event_tokens.json.
# assert verifier.verify_tokens_per_forward_pass(in_tokens, out_tokens, queries_per_iter), "Vllm and simulator tokens do not match!!"
