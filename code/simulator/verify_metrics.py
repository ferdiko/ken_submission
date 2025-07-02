"""Script to compare request metrics between vllm and simulator."""
import random

import utils.metrics_verifier as metrics_verifier
from utils import configure_logging

# Initialize logging.
print("Start configure_logging")
logger = configure_logging()
logger.info("End configure_logging.")

# #######################
# ## 1 request (small) ##
# #######################
# logger.info("Start initializing MetricsVerifier.")
# verifier = metrics_verifier.MetricsVerifier(
#     model_name="llama8b_fp8",
#     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
#     max_tokens=1,
#     max_model_len=2048,
#     max_num_seqs=100)
# logger.info("Finish initializing MetricsVerifier.")

# in_tokens = [200]
# out_tokens = [1]
# queries_per_iter = [1]

# verifier.compare_metrics([400], [100], queries_per_iter, "", "outputs/metrics/1_request_400_in.json")
# # verifier.compare_metrics(in_tokens, out_tokens, queries_per_iter, "outputs/metrics/1_request_2000_in_second.txt")

# #######################
# ## 1 request (large) ##
# #######################
# max_model_len = 2000
# verifier = metrics_verifier.MetricsVerifier(
#     max_tokens=200,
#     max_num_batched_tokens=5000,
#     max_model_len=max_model_len)

# verifier.compare_metrics([200], [50], [1], "outputs/metrics/1_request_1000.txt", "outputs/metrics/200.json")
# verifier.compare_metrics([400], [100], [1], "outputs/metrics/1_request_1000.txt", "outputs/metrics/400.json")
# verifier.compare_metrics([1000], [200], [1], "outputs/metrics/1_request_1000.txt", "outputs/metrics/1000.json")
# verifier.compare_metrics([1500], [250], [1], "outputs/metrics/1_request_1500.txt", "outputs/metrics/1500.json")

# def create_verifier_input(num_requests, num_in_tokens, num_out_tokens):
#     in_tokens, out_tokens = [], []
#     queries_per_iter = [num_requests]
#     for _ in range(num_requests):
#         in_tokens.append(num_in_tokens)
#         out_tokens.append(num_out_tokens)

#     return in_tokens, out_tokens, queries_per_iter

# # 5 large requests
# verifier.compare_metrics(
#     *create_verifier_input(5, 1500, 250),
#     "outputs/metrics/5_request_1500.txt", 
#     "outputs/metrics/5_1500.json")

# # 10 large requests
# verifier.compare_metrics(
#     *create_verifier_input(10, 1500, 250),
#     "outputs/metrics/10_request_1500.txt",
#     "outputs/metrics/10_1500.json")

# ######################
# # 3 small requests  ##
# ######################
# logger.info("Start initializing MetricsVerifier.")
# verifier = metrics_verifier.MetricsVerifier(
#     model_name="llama8b_fp8",
#     model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
#     max_tokens=1,
#     max_model_len=2048,
#     max_num_seqs=100)
# logger.info("Finish initializing MetricsVerifier.")

# in_tokens = [10, 11, 12]
# out_tokens = [1, 1, 1]
# queries_per_iter = [1, 1, 1]

# for i in range(5):
#     verifier.compare_metrics(in_tokens, out_tokens, queries_per_iter, f"outputs/metrics/3_request_10_in_iter_{i}.json")
# # verifier.dump_complete_profile("outputs/profiles/8b_verify_metrics.json")

####################################
## Larger simultaneous requests. ##
####################################
# Initialize a token verifier
verifier = metrics_verifier.MetricsVerifier(
    model_name="llama8b_fp8",
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    max_tokens=5,
    max_num_batched_tokens=1024,
    max_model_len=2048,
    max_num_seqs=1000)

verifier.dump_complete_profile("outputs/profiles/8b_verify_metrics-70b_L40S.json")

# Use a 5 to 1 ratio of in to out tokens.
num_requests = 10
queries_per_iter = [num_requests]
in_tokens = []
out_tokens = []
for _ in range(num_requests):
    in_tokens.append(1000)
    out_tokens.append(200)

verifier.compare_metrics(
    in_tokens,
    out_tokens,
    queries_per_iter,
    out_file_path=f"outputs/metrics/{num_requests}_request_1000_new.json",
    trace_file_path=f"outputs/traces/verifier-llama8b_fp8-{num_requests}_request_1000_new.json",
    include_forward_passes=False,
)

# for i in range(1):
#     verifier.compare_metrics(
#         in_tokens,
#         out_tokens,
#         queries_per_iter,
#         out_file_path=f"outputs/metrics/{num_requests}_request_1000_in_iter_{i}.json",
#         trace_file_path=f"outputs/traces/verifier-llama8b_fp8-{num_requests}_request_1000_in_iter_{i}.json"
#     )

###############################
## n random large requests  ##
###############################
# # Define request specs.
# logger.info("Start initializing TokenVerifier.")
# max_model_len = 2000
# verifier = metrics_verifier.MetricsVerifier(
#     max_tokens=200,
#     max_num_batched_tokens=5000,
#     max_model_len=max_model_len)
# logger.info("Finish initializing TokenVerifier.")
# num_requests = 40
# queries_per_iter = [num_requests]
# in_tokens = []
# out_tokens = []
# for _ in range(num_requests):
#     x = random.randint(500, int((max_model_len // 1.2) - 100))  # Generate a random in_token. Upper limit ensures in + out < max_model_len (even with randmoness of prompt generation).
#     in_tokens.append(x)
#     out_tokens.append(x // 5)

# verifier.compare_metrics(in_tokens, out_tokens, queries_per_iter)
