import json

from utils import configure_logging
from vllm_simulator.profiler.vllm_profile import VllmProfile

logger = configure_logging(__name__)


class VllmCompleteProfile:
    '''
    Characterization of a vLLM instance that can predict forward pass runtimes (units=seconds) using profiles with CUDA graph enabled or disabled.
    '''
    
    def __init__(self,
                 cuda_profile: VllmProfile,
                 no_cuda_profile: VllmProfile,
                 block_size: int,
                 num_gpu_blocks: int,
                 num_watermark_blocks: int,
                 model: str,
                 model_name: str,
                 max_num_batched_tokens: int,
                 max_model_len: int,
                 max_num_seqs: int,
                 input_token_upper_limit: int) -> None:
        '''Initializes VllmCompleteProfile instance.'''
        # Profile of forward pass times that use CUDA graph.
        self.cuda_profile = cuda_profile
        # Profile of forward pass times that do not use CUDA graph.
        self.no_cuda_profile = no_cuda_profile

        # KV cache information.
        # Determines size in number of tokens in each block of the KV cache.
        self.block_size = block_size
        # Determines number of GPU blocks allocated for the KV cache.
        self.num_gpu_blocks = num_gpu_blocks
        self.num_watermark_blocks = num_watermark_blocks

        # Other information from profiler to be stored so profile can be identified.
        self.model = model
        self.model_name = model_name
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.input_token_upper_limit = input_token_upper_limit

        logger.info(f"VllmCompleteProfile initialized with block_size={self.block_size}, num_gpu_blocks={self.num_gpu_blocks}.")

    def dump(self, profile_file_path: str) -> None:
        '''Writes VllmCompleteProfile instance in JSON format to file.'''
        logger.info(f"Dumping VllmCompleteProfile to JSON at {profile_file_path}.")
        
        # Convert intervals and additional attributes into a JSON-serializable format
        serializable_data = {
            "cuda_profile": self.cuda_profile.to_json_serializable_dict(),
            "no_cuda_profile": self.no_cuda_profile.to_json_serializable_dict(),

            "block_size": self.block_size,
            "num_gpu_blocks": self.num_gpu_blocks,
            "num_watermark_blocks": self.num_watermark_blocks,

            "model": self.model,
            "model_name": self.model_name,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "input_token_upper_limit": self.input_token_upper_limit,
        }
        
        with open(profile_file_path, 'w') as file:
            json.dump(serializable_data, file, sort_keys=True, indent=4)
        
        logger.info(f"Finished dumping VllmCompleteProfile JSON to {profile_file_path}.")

    @classmethod
    def load(cls, profile_file_path: str):
        '''Loads VllmCompleteProfile instance from JSON file.'''
        logger.info(f"Loading VllmProfile from JSON at {profile_file_path}.")
        
        with open(profile_file_path, 'r') as file:
            data = json.load(file)
        
        # Convert the loaded data back to the original VllmProfile format.
        cuda_profile_data = data.get("cuda_profile", {})
        cuda_profile = VllmProfile.load_from_dict(cuda_profile_data)
        no_cuda_profile_data = data.get("no_cuda_profile", {})
        no_cuda_profile = VllmProfile.load_from_dict(no_cuda_profile_data)

        block_size = data["block_size"]
        num_gpu_blocks = data["num_gpu_blocks"]
        num_watermark_blocks = data["num_watermark_blocks"]

        model = data["model"]
        model_name = data["model_name"]
        max_num_batched_tokens = data["max_num_batched_tokens"]
        max_model_len = data["max_model_len"]
        max_num_seqs = data["max_num_seqs"]
        input_token_upper_limit = data["input_token_upper_limit"]

        logger.info("VllmCompleteProfile loaded successfully.")
        return cls(
            cuda_profile,
            no_cuda_profile,
            block_size,
            num_gpu_blocks,
            num_watermark_blocks,
            model,
            model_name,
            max_num_batched_tokens,
            max_model_len,
            max_num_seqs,
            input_token_upper_limit)
    
    @classmethod
    def combine_profiles(cls, cuda_profile_path, no_cuda_profile_path):
        cuda_profile = VllmProfile.load(cuda_profile_path)
        no_cuda_profile = VllmProfile.load(no_cuda_profile_path)

        # Check that the profiles are from the same instance of vLLM.
        assert cuda_profile.block_size == no_cuda_profile.block_size, \
            f"block_size must match but {cuda_profile.block_size} != {no_cuda_profile.block_size}"
        assert cuda_profile.num_gpu_blocks == no_cuda_profile.num_gpu_blocks, \
            f"num_gpu_blocks must match but {cuda_profile.num_gpu_blocks} != {no_cuda_profile.num_gpu_blocks}"
        assert cuda_profile.num_watermark_blocks == no_cuda_profile.num_watermark_blocks, \
            f"num_watermark_blocks must match but {cuda_profile.num_watermark_blocks} != {no_cuda_profile.num_watermark_blocks}"
        assert cuda_profile.model == no_cuda_profile.model, \
            f"model must match but {cuda_profile.model} != {no_cuda_profile.model}"
        assert cuda_profile.model_name == no_cuda_profile.model_name, \
            f"model_name must match but {cuda_profile.model_name} != {no_cuda_profile.model_name}"
        assert cuda_profile.max_num_batched_tokens == no_cuda_profile.max_num_batched_tokens, \
            f"max_num_batched_tokens must match but {cuda_profile.max_num_batched_tokens} != {no_cuda_profile.max_num_batched_tokens}"
        assert cuda_profile.max_model_len == no_cuda_profile.max_model_len, \
            f"max_model_len must match but {cuda_profile.max_model_len} != {no_cuda_profile.max_model_len}"
        assert cuda_profile.max_num_seqs == no_cuda_profile.max_num_seqs, \
            f"max_num_seqs must match but {cuda_profile.max_num_seqs} != {no_cuda_profile.max_num_seqs}"
        assert cuda_profile.input_token_upper_limit == no_cuda_profile.input_token_upper_limit, \
            f"input_token_upper_limit must match but {cuda_profile.input_token_upper_limit} != {no_cuda_profile.input_token_upper_limit}"
        
        return cls(
            cuda_profile,
            no_cuda_profile,
            cuda_profile.block_size,
            cuda_profile.num_gpu_blocks,
            cuda_profile.num_watermark_blocks,
            cuda_profile.model,
            cuda_profile.model_name,
            cuda_profile.max_num_batched_tokens,
            cuda_profile.max_model_len,
            cuda_profile.max_num_seqs,
            cuda_profile.input_token_upper_limit)
    
    def predict_forward_pass_runtime(self, num_tokens: int, use_cuda_graph_profile: bool) -> int:
        '''Returns forward pass runtime in ms.'''
        assert num_tokens > 0, "Number of tokens to process must be greater than zero."

        if use_cuda_graph_profile:
            logger.info("Using cuda profile")
            return self.cuda_profile.predict_forward_pass_runtime(num_tokens)
        else:
            logger.info("Using no cuda profile")
            return self.no_cuda_profile.predict_forward_pass_runtime(num_tokens)
