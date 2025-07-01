from vllm import LLM, SamplingParams, SequenceGroup
from torch.profiler import ProfilerActivity, profile
from typing import List
import json

from vllm_simulator.profiler.vllm_profile import VllmProfile
from vllm_simulator.profiler.vllm_complete_profile import VllmCompleteProfile
from utils.simulator_helpers import get_logger_records

from utils import configure_logging
import utils.parse_vllm_output as parse_vllm_output
import utils.generate_vllm_input as generate_vllm_input
import utils.simulator_helpers as simulator_helpers

logger = configure_logging(__name__, log_to_file=True, log_file_path="outputs/logs/vllm_simulator.log")


TRACES_DIRECTORY = "outputs/traces"
CACHE_LOG_FILE_PATH = "outputs/cache.log"
STEPS_LOG_FILE_PATH = "outputs/steps.log"


class VllmProfiler:
    '''
    # vLLM specific instance variables.
    vllm_specifications: dict[str, str] # Configuration details of vLLM.
    llm: LLM # vLLM entry point.

    max_input_sequence_tokens: int # Maximum input sequence length (in tokens).
    token_to_runtime_data: dict[int, list[int]] # Keys are number of tokens in the forward pass. Values are lists of forward pass runtimes. This is an intermediate data storage while vLLM is being run multiple times.
    vllm_profile: VllmProfile # Stores VllmProfile instance created by this VllmProfiler.
    '''
    def __init__(self,
                 model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
                 model_name="llama-70b_L40S",
                 max_tokens=1,
                 max_num_batched_tokens=512,
                 max_model_len=2000,
                 max_num_seqs=100,
                 debug_analyze=False, 
                 print_generated_text=False):
        '''Calls _setup_vllm and _characterize_vllm.'''
        logger.info("Initializing VllmProfiler.")
        self.model = model
        self.model_name = model_name

        # vLLM configuration settings for SamplingParams.
        self.max_tokens = max_tokens # Maximum number of tokens to generate per output sequence.

        # vLLM configuration settings for LLM entrypoint.
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs

        # Upper bound of number of tokens that can be passed in.
        self.input_token_upper_limit = min(max_num_batched_tokens, self.max_model_len)

        # TODO: Deprecate.
        # If true, eliminates vllm setup and running overhead when focusing on analysis functions (ie analyze_vllm_output).
        self.debug_analyze = debug_analyze

        # If true, prints the prompt and generated output after each run of vLLM
        self.print_generated_text = print_generated_text

        # Used for generating baselines.
        self.step_coords_results_path = f"results/{model_name}_batch-{max_num_batched_tokens}_max-model-len-{max_model_len}_max-num-seqs-{max_num_seqs}"

        if not self.debug_analyze:
            # Initialize vLLM setup.
            self._setup_vllm()

        # Profile this vllm instance.
        self._characterize_vllm()


    def _setup_vllm(self):
        '''Initializes vLLM instance.'''
        # Clear cache.log.
        with open(CACHE_LOG_FILE_PATH, 'w'):
            pass

        self.sampling_params = SamplingParams(best_of=1,
            temperature=0.0,
            top_p=1,
            top_k=1,
            max_tokens=self.max_tokens,
            presence_penalty=0,
            frequency_penalty=0)
        
        # Create an LLM.
        self.llm = LLM(
            model=self.model,
            quantization="fp8", # Comment out for 1B, 3B.
            tensor_parallel_size=2,
            # Extra arguments for EngineArgs as input to the LLMEngine go here.
            max_num_batched_tokens=self.max_num_batched_tokens, # Must be >= max_model_len, max_num_seqs. Essentially no limit, allows modification of batch size. Defaults to 8192 if enable_chunked_prefill==False.
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enable_chunked_prefill=True, # Toggling chunked_prefill.
        )

        logger.info("vLLM setup finished.")

    def _characterize_vllm(self):
        '''Characterizes vLLM fully and how long each forward pass takes depending on number of tokens.'''
        # Parse KV cache characteristics in all modes (debug or not).
        self._analyze_vllm_cache_output()
        # Send a small warmup request before measuring times.
        self._process_warmup_request()
        # Create no_cuda_profile.
        self._characterize_vllm_no_cuda()
        # Create cuda_profile.
        self._characterize_vllm_cuda()

        # Combine the above profiles into complete profile.
        self.complete_profile = VllmCompleteProfile(
            self.cuda_profile,
            self.no_cuda_profile,
            block_size=self.block_size,
            num_gpu_blocks=self.num_gpu_blocks,
            num_watermark_blocks=self.num_watermark_blocks,
            model=self.model,
            model_name=self.model_name,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            input_token_upper_limit=self.input_token_upper_limit)

    def _process_warmup_request(self):
        '''Completes a small warmup request to eliminate initial overheads.
        
        Usually see big time jump in vLLM's first prefill and first decode.'''
        logger.info(f"Running warmup request.")
        self._run_vllm_new(in_tokens=[2], out_tokens=[3], queries_per_iter=[1], is_warmup=True)
        logger.info(f"Finished running warmup request.")

    ################################
    # No CUDA profiling functions. #
    ################################
    def _characterize_vllm_no_cuda(self):
        '''Runs vLLM with varying input sequence lengths from 1 token to the max_input_sequence_tokens limit. Analyzes log and trace output.
        
        Divides the number of input tokens from 0 to max_num_batched_tokens into chunks of size 100.
        Uses linear interpolation to determine the slope of each 100.'''
        # For each 100 token interval, generate tokens and run_vllm.
        curr_num_input_tokens = 1
        while curr_num_input_tokens <= self.input_token_upper_limit:
            self._run_vllm_no_cuda(curr_num_input_tokens)

            # Update number of input tokens for the next pass.
            curr_num_input_tokens += 50

        # Run a request with max_batch_size.
        self._run_vllm_no_cuda(self.max_num_batched_tokens - 1) # Subtract 2 for buffer.

        # Analyze output to steps.log.
        tokens_to_runtime = self._analyze_vllm_output_no_cuda()

        # Use linear interpolation to determine interval specific equations for number of tokens to event duration.
        intervals = parse_vllm_output.linear_interpolate_tokens_to_event_duration(logger, tokens_to_runtime, self.input_token_upper_limit)
        self.no_cuda_profile = VllmProfile(
            intervals=intervals,
            uses_cuda=False,
            block_size=self.block_size,
            num_gpu_blocks=self.num_gpu_blocks,
            num_watermark_blocks=self.num_watermark_blocks,
            model=self.model,
            model_name=self.model_name,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            input_token_upper_limit=self.input_token_upper_limit)

    def _run_vllm_no_cuda(self, num_input_tokens):
        '''Runs vllm with generated prompts and dumps trace to file.'''
        # TODO: Change this to use run_vllm_new (ie llm.run_workload_trace).
        # Generate tokens if needed.
        input_token_file_name = generate_vllm_input.generate_input_tokens(num_input_tokens)
        prompts = generate_vllm_input.read_input_tokens_from_file(input_token_file_name)

        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        logger.info(f"vLLM running with {num_input_tokens} tokens.")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            outputs = self.llm.generate(prompts, self.sampling_params)

        # Print the prompt and generated output if desired.
        if self.print_generated_text:
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        # Dump trace.
        trace_file_path = f"outputs/traces/{self.model_name}_{num_input_tokens}_{self.get_cuda_label(uses_cuda=False)}.json"
        prof.export_chrome_trace(trace_file_path)
        logger.info(f"Trace dumped to {trace_file_path}")

    def _analyze_vllm_output_no_cuda(self):
        '''Analyze the vLLM output for tokens, runtime, and KV cache characterization.'''
        # # Note: this currently will go through the output steps from _process_warmup_request. It correctly ignores these metrics.
        # Access the records.
        steps_info = get_logger_records("steps_logger")
        tokens_to_runtime = {}
        for step_info in steps_info:
            # If small warmup request exists, skip its data.
            if step_info["is_warmup"]:
                continue

            num_batched_tokens = step_info["num_batched_tokens"]
            duration = step_info["duration"]

            # TODO: Currently tokens_to_runtime can only store 1 duration per num_tokens data point-- should this be an average later?
            tokens_to_runtime[num_batched_tokens] = duration

        # Output steps.info in an easy to plot format.
        with open(f"{self.step_coords_results_path}/steps_coordinates_no_cuda.json", "w") as file:
            # Num_tokens: duration in seconds.
            file.write(json.dumps(tokens_to_runtime, indent=4))
            # file.write("({num_tokens}, {duration in seconds})\n")
            # for num_tokens, duration in tokens_to_runtime.items():
            #     file.write(f"({num_tokens}, {duration})\n")
        
        return tokens_to_runtime

    #############################
    # CUDA profiling functions. #
    #############################
    def _characterize_vllm_cuda(self):
        '''Runs vLLM with n sequences and measures the time for a forward pass with only decode tokens.'''
        # For each 100 token interval, generate tokens and run_vllm.
        decode_tokens_in_forward_pass = 1
        # Note: A decode pass can only have at maximum max_num_seqs tokens.
        while decode_tokens_in_forward_pass <= self.max_num_seqs:
        # Shorten the loop to decrease runtime.
        # while decode_tokens_in_forward_pass <= 200:
            in_tokens = [1 for _ in range(decode_tokens_in_forward_pass)]
            out_tokens = [10 for _ in range(decode_tokens_in_forward_pass)]
            queries_per_iter = [decode_tokens_in_forward_pass]
            self._run_vllm_new(in_tokens, out_tokens, queries_per_iter)

            # Update number of input tokens for the next pass.
            decode_tokens_in_forward_pass += 100

        # Analyze output to steps.log.
        tokens_to_runtime = self._analyze_vllm_output_cuda()

        # Use linear interpolation to determine interval specific equations for number of tokens to event duration.
        input_token_upper_limit = min(self.max_num_batched_tokens, self.max_num_seqs)
        intervals = parse_vllm_output.linear_interpolate_tokens_to_event_duration(logger, tokens_to_runtime, input_token_upper_limit)
        self.cuda_profile = VllmProfile(
            intervals=intervals,
            uses_cuda=True,
            block_size=self.block_size,
            num_gpu_blocks=self.num_gpu_blocks,
            num_watermark_blocks=self.num_watermark_blocks,
            model=self.model,
            model_name=self.model_name,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            input_token_upper_limit=input_token_upper_limit)

    def _run_vllm_new(self, in_tokens: List[int], out_tokens: List[int], queries_per_iter: List[int], is_warmup=False):
        '''Runs vllm with n prompts. Measures time of decode pass. Dumps trace to file.'''
        logger.info("Start generating prompts.")
        seq_groups = self._generate_vllm_prompts(in_tokens, out_tokens)
        logger.info("Prompts generated.")

        vllm_in_tokens = []
        for seq_group in seq_groups:
            # Note: Each seq_group is a request and only holds one sequence for our purposes. Thus,
            #   only need to inspect the first sequence in the seq_group.
            seq = seq_group.seqs[0]
            vllm_in_tokens.append(seq.get_prompt_len())

        logger.info("Start running workload trace.")
        # Issue all requests at once
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            # Run workload. This outputs tokens per event log, query_arrival_timestamps, and query_ids
            _, query_arrival_timestamps, query_ids = self.llm.run_workload_trace(queries_per_iter=queries_per_iter,
                                                                                requests=seq_groups,
                                                                                log_token_verification=False,
                                                                                is_warmup=is_warmup)
        logger.info("Finished running workload trace.")
        
        # Dump trace.
        num_requests = len(in_tokens)
        trace_file_path = f"outputs/traces/{self.model_name}_{num_requests}_requests_{in_tokens[0]}_in_{self.get_cuda_label(uses_cuda=True)}.json"
        prof.export_chrome_trace(trace_file_path)
        logger.info(f"Trace dumped to {trace_file_path}")

    def _analyze_vllm_output_cuda(self):
        '''Analyze the vLLM output for pure decode forward pass runtimes.'''
        # Note: this currently will go through the output steps from _process_warmup_request and _characterize_vllm_no_cuda.
        #  It correctly includes or ignores these metrics.
        # Parse steps log into steps_info, a list of dictionaries.
        steps_info = get_logger_records("steps_logger")
        tokens_to_runtime = {}
        for step_info in steps_info:
            # If small warmup request exists, skip its data.
            if step_info["is_warmup"]:
                continue
            # Only take times of decode passes which is when CUDA is used.
            if step_info["only_contains_decode"]:
                num_batched_tokens = step_info["num_batched_tokens"]
                duration = step_info["duration"]

                # TODO: Currently tokens_to_runtime can only store 1 duration per num_tokens data point-- should this be an average later?
                tokens_to_runtime[num_batched_tokens] = duration

        # Output steps.info in an easy to plot format.
        with open(f"{self.step_coords_results_path}/steps_coordinates_cuda.json", "w") as file:
            # Num_tokens: duration in seconds.
            file.write(json.dumps(tokens_to_runtime, indent=4))
            # file.write("({num_tokens}, {duration in seconds})\n")
            # for num_tokens, duration in tokens_to_runtime.items():
            #     file.write(f"({num_tokens}, {duration})\n")

        return tokens_to_runtime

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
                raise Exception(f"Request with in={len(s.prompt_token_ids())}, out={out} exceeds max_model_len={self.max_model_len}.")
            s.seqs[0].out_tokens = out

        return seq_groups

    ####################
    # Other functions. #
    ####################
    def _analyze_vllm_cache_output(self):
        '''Parses cache.log and sets instance variables on KV cache size.'''
        # Parse the cache.log to retrieve block_size, num_gpu_blocks.
        block_size, num_gpu_blocks, num_watermark_blocks = parse_vllm_output.parse_cache_log(logger)

        # Set instance variables on block size and num_gpu_blocks.
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_watermark_blocks = num_watermark_blocks

    def get_vllm_profile(self, uses_cuda: bool) -> VllmProfile:
        '''Returns a VllmProfile instance.'''
        if uses_cuda:
            return self.cuda_profile
        else:
            return self.no_cuda_profile

    def get_vllm_complete_profile(self) -> VllmCompleteProfile:
        '''Returns a VllmCompleteProfile instance.'''
        return self.complete_profile

    def dump_vllm_profile(self, profile_file_path: str, uses_cuda: bool) -> None:
        '''Writes VllmProfile instance in JSON format to file.'''
        if uses_cuda:
            assert self.cuda_profile is not None, "vllm_profile is None, can not dump to file"
            simulator_helpers.maybe_create_dir(profile_file_path)
            self.cuda_profile.dump(profile_file_path)
        else:
            assert self.no_cuda_profile is not None, "vllm_profile is None, can not dump to file"
            simulator_helpers.maybe_create_dir(profile_file_path)
            self.no_cuda_profile.dump(profile_file_path)

    def dump_complete_profile(self, profile_file_path: str) -> None:
        '''Writes VllmCompleteProfile instance in JSON format to file.'''
        assert self.complete_profile is not None, "complete_profile is None, can not dump to file"
        simulator_helpers.maybe_create_dir(profile_file_path)
        self.complete_profile.dump(profile_file_path)

    def get_llm(self):
        return self.llm
    
    def get_sampling_params(self):
        return self.sampling_params

    def get_cuda_label(self, uses_cuda: bool):
        return "no_cuda" if uses_cuda else "cuda"