from typing import List

import utils.verification_helpers as verification_helpers
import utils.verifer as verifier

from utils import configure_logging
logger = configure_logging(__name__)
VERIFICATION_DIRECTORY = "outputs/verification"
VLLM_LOG_FILE_PATH = f"{VERIFICATION_DIRECTORY}/vllm_event_tokens.json"
SIMULATOR_LOG_FILE_PATH = f"{VERIFICATION_DIRECTORY}/simulator_event_tokens.json"


class TokenVerifier(verifier.Verifier):
    """Verifies that request tokens are handled in the exact same way per forward pass
    between vllm and VllmSimulator runs.
    """
    def verify_tokens_per_forward_pass(self, in_tokens: List[int], out_tokens: List[int], queries_per_iter: List[int]) -> bool:
        """Compares token consumption between vllm and simulator. Returns true if: 1) same number of forward passes, 2) each
        request consumes the same number of tokens per forward pass.

        Arguments:
            in_tokens: List of how many prefill tokens for each request.
            out_tokens: List of how many decode tokens for each request.
            queries_per_iter: List of how many queries will be added to waiting queue on each forward pass.
        """
        # Validate inputs.
        self.validate_input(in_tokens, out_tokens, queries_per_iter)
        # Clear logs.
        self._clear_event_tokens_logs()
        verification_helpers.clear_logs()

        # Produce event organized logs.
        # Note: Vllm uses a tokenizer to generate prompts, and they don't exactly match the specified in_tokens.
        vllm_in_tokens, query_arrival_timestamps, query_ids = self._run_vllm(in_tokens, out_tokens, queries_per_iter)
        self._run_simulator(vllm_in_tokens, out_tokens, queries_per_iter, query_ids, query_arrival_timestamps)

        # Read event token logs for vllm and simulator.
        vllm_event_tokens = verification_helpers.parse_event_tokens(VLLM_LOG_FILE_PATH)
        simulator_event_tokens = verification_helpers.parse_event_tokens(SIMULATOR_LOG_FILE_PATH)

        # Verify event tokens per forward pass. Log any mismatches.
        return verification_helpers.verify_event_tokens(vllm_event_tokens, simulator_event_tokens)

    def _clear_event_tokens_logs(self):
        '''Clears the event tokens log files.'''
        with open(VLLM_LOG_FILE_PATH, 'w'):
            pass
        with open(SIMULATOR_LOG_FILE_PATH, 'w'):
            pass
