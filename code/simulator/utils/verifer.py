from typing import List

from vllm import LLM, SamplingParams, RequestOutput, SequenceGroup
from torch.profiler import ProfilerActivity, profile

import utils.generate_vllm_input as generate_vllm_input
import utils.simulator_helpers as simulator_helpers
import utils.verification_helpers as verification_helpers

from vllm_simulator.profiler.vllm_profile import VllmProfile
from vllm_simulator.profiler.vllm_profiler import VllmProfiler
from vllm_simulator.simulator.vllm_simulator import VllmSimulator

from utils import configure_logging
logger = configure_logging(__name__)


class Verifier:
    """Verifier parent class that is inherited by TokenVerifier or MetricsVerifier.
    
    Follows the following steps:
    1. Initializes vllm entrypoint, llm. 
    2. Uses the llm to profile and create a log.
    3. Initializes VllmSimulator to create a synonymous log.
    4. Verifies tokens per forward pass or metrics."""
    def __init__(self,
                 model_name="llama70b_fp8",
                 model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
                 max_tokens=5, # Output sequence length.
                 max_num_batched_tokens=700,
                 max_model_len=600,
                 max_num_seqs=300,
                 vllm_profile_path=None):
        self.max_model_len = max_model_len
        if vllm_profile_path is not None:
            # When vllm_profile is provided, do not reprofile. This is used to debug the simulator token verification only.
            vllm_profile = VllmProfile.load(vllm_profile_path)
            self.vllm_simulator = VllmSimulator(vllm_profile, max_num_batched_tokens)
            logger.info(f"Vllm profile path initialized from {vllm_profile_path}. Will not initialize vllm or re-profile.")
            return
        
        # Initialize vllm profiler.
        self.vllm_profiler = VllmProfiler(
                 model_name=model_name,
                 model=model,
                 max_tokens=max_tokens,
                 max_num_batched_tokens=max_num_batched_tokens,
                 max_model_len=max_model_len,
                 max_num_seqs=max_num_seqs,
                 debug_analyze=False, 
                 print_generated_text=False)
        logger.info("VllmProfiler finished initializing.")

        # Extract vllm resources needed for vllm runs.
        self.llm = self.vllm_profiler.get_llm()

        # Initialize simulator.
        vllm_complete_profile = self.vllm_profiler.get_vllm_complete_profile()
        self.vllm_simulator = VllmSimulator(vllm_complete_profile)

    def validate_input(self, in_tokens: List[int], out_tokens: List[int], queries_per_iter: List[int], trace_file_path=None):
        """Validate the number of queries defined and that {in, out}_tokens > 0."""
        simulator_helpers.validate_simulated_request_input(in_tokens, out_tokens, queries_per_iter)

    def _run_vllm(self, in_tokens: List[int], out_tokens: List[int], queries_per_iter: List[int], trace_file_path=None) -> tuple[List[int], List[float], List[str]]:
        '''Runs vllm, saves log, and returns query_arrival_timestamps.
        
        Closely follows https://github.com/ferdiko/vllm/blob/main/examples/run_single_server.py'''
        # Generate prompts.
        logger.info("Start generating prompts.")
        seq_groups = self._generate_vllm_prompts(in_tokens, out_tokens)
        logger.info("Prompts generated.")

        # Determine the vllm_in_tokens for the vllm tokenizer generated prompts.
        vllm_in_tokens = []
        for seq_group in seq_groups:
            # Note: Each seq_group is a request and only holds one sequence for our purposes. Thus,
            #   only need to inspect the first sequence in the seq_group.
            seq = seq_group.seqs[0]
            vllm_in_tokens.append(seq.get_prompt_len())

        logger.info("Start running workload trace.")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            # Run workload. This outputs tokens per event log, query_arrival_timestamps, and query_ids
            _, query_arrival_timestamps, query_ids = self.llm.run_workload_trace(queries_per_iter=queries_per_iter,
                                                                                requests=seq_groups,
                                                                                log_token_verification=True)
        
        logger.info("Finished running workload trace.")
        
        trace_file_path = f"outputs/traces/verifier-{self.vllm_profiler.model_name}.json" if trace_file_path is None else trace_file_path
        prof.export_chrome_trace(trace_file_path)

        return vllm_in_tokens, query_arrival_timestamps, query_ids

    def _generate_vllm_prompts(self, in_tokens, out_tokens) -> List[SequenceGroup]:
        """Generates random vllm prompts."""
        # Use tokenizer to generate prompts
        tokenizer = self.llm.get_tokenizer()
        prompts, prompt_token_ids = generate_vllm_input.generate_random_prompts(tokenizer, in_tokens)
        
        # Generate requests as sequence groups.
        sampling_params = []
        for out in out_tokens:
            sampling_params.append(SamplingParams(best_of=1,
                                            temperature=0.0,
                                            top_p=1,
                                            top_k=-1,
                                            max_tokens=out,
                                            presence_penalty=0,
                                            frequency_penalty=0,
                                            min_tokens=out))

        seq_groups = self.llm.produce_request(prompts, sampling_params, prompt_token_ids=prompt_token_ids)

        # Combine sequence groups and number of out_tokens.
        for s, out in zip(seq_groups, out_tokens):
            # Validate that the prompt does not exceed max_model_len.
            if len(s.prompt_token_ids) + out >= self.max_model_len:
                raise Exception(f"Request with in={len(s.prompt_token_ids)}, out={out} exceeds max_model_len={self.max_model_len}.")
            s.seqs[0].out_tokens = out

        return seq_groups

    def _run_simulator(self,
                       in_tokens: List[int],
                       out_tokens: List[int],
                       queries_per_iter: List[int],
                       query_names: List[str],
                       query_arrival_timestamps: List[float],
                       visualize_metrics: bool=True):
        '''Runs simulator and outputs event organized token log.'''
        # TODO: Deprecate query_arrival_timestamps.
        # Generate request schedule. Each request is given an arrival timestamp that will be rewritten/ignored in simulation.
        request_schedule = verification_helpers.generate_simulated_requests(in_tokens, out_tokens, query_arrival_timestamps, query_names)

        # Run simulator and output event token log.
        request_metrics_visualizer = self.vllm_simulator.simulate_requests_per_iter(request_schedule, queries_per_iter)
        
        # Visualize metrics.
        if visualize_metrics:
            request_metrics_visualizer.log_metrics()
        
        return request_metrics_visualizer

    def dump_complete_profile(self, profile_file_path: str) -> None:
        '''Writes VllmProfile instance in JSON format to file.'''
        self.vllm_profiler.dump_complete_profile(profile_file_path)