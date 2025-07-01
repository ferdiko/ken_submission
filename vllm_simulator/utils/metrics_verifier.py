from typing import List, Optional

import utils.verification_helpers as verification_helpers
import utils.verifer as verifier

from utils import configure_logging
logger = configure_logging(__name__)
VLLM_LOG_FILE_PATH = "outputs/verification/vllm_metrics.json"


class MetricsVerifier(verifier.Verifier):
    """Compares request metrics between vllm and VllmSimulator runs.

    Only considers Time To First Token (TTFT) and Total Generation Time (TGT).
    """
    def compare_metrics(self, in_tokens: List[int], out_tokens: List[int], queries_per_iter: List[int], out_file_path: str, trace_file_path=None, simulator_metrics_file_path=None, include_forward_passes=False, visualize_simulator_request_metrics=False) -> bool:
        """Compares request metrics between vllm and simulator, and logs the percent difference for TTFT, TGT.

        Arguments:
            in_tokens: List of how many prefill tokens for each request.
        out_tokens: List of how many decode tokens for each request.
            queries_per_iter: List of how many queries will be added to waiting queue on each forward pass.
        """
        # Validate inputs.
        self.validate_input(in_tokens, out_tokens, queries_per_iter)
        # Clear logs.
        self._clear_metrics_logs()
        verification_helpers.clear_logs()

        # Run vllm and simulator.
        # Note: Vllm uses a tokenizer to generate prompts, and they don't exactly match the specified in_tokens.
        vllm_in_tokens, query_arrival_timestamps, query_ids = self._run_vllm(in_tokens, out_tokens, queries_per_iter, trace_file_path)
        request_metrics_visualizer = self._run_simulator(vllm_in_tokens, out_tokens, queries_per_iter, query_ids, query_arrival_timestamps, visualize_metrics=visualize_simulator_request_metrics)

        return verification_helpers.new_output_request_metric(
            out_file_path,
            request_metrics_visualizer,
            simulator_metrics_file_path=simulator_metrics_file_path,
            include_forward_passes=include_forward_passes)

    def _clear_metrics_logs(self):
        '''Clears the event tokens log files.'''
        with open(VLLM_LOG_FILE_PATH, 'w'):
            pass
