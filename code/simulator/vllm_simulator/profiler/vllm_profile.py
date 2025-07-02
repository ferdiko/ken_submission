import json

from utils import configure_logging


logger = configure_logging(__name__)


class VllmProfile:
    '''
    Characterization of a vLLM instance that can predict forward pass runtimes (units=seconds).
    '''

    def __init__(self,
                 intervals: dict,
                 uses_cuda: bool,
                 block_size: int,
                 num_gpu_blocks: int,
                 num_watermark_blocks: int,
                 model: str,
                 model_name: str,
                 max_num_batched_tokens: int,
                 max_model_len: int,
                 max_num_seqs: int,
                 input_token_upper_limit: int) -> None:
        '''Initializes VllmProfile instance.'''
        # If enforce_eager = False from LLM initialization
        self.uses_cuda = uses_cuda

        # The keys in the dictionary are inclusive interval endpoints (i.e. (interval_start, interval_end)), and the value is a TokensToRuntime instance.
        self.intervals = intervals

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

        logger.info(f"VllmProfile initialized with block_size={self.block_size}, num_gpu_blocks={self.num_gpu_blocks}.")

    def dump(self, profile_file_path: str) -> None:
        '''Writes VllmProfile instance in JSON format to file.'''
        logger.info(f"Dumping VllmProfile to JSON at {profile_file_path}.")
        
        # Convert intervals and additional attributes into a JSON-serializable format
        serializable_data = self.to_json_serializable_dict()
        
        with open(profile_file_path, 'w') as file:
            json.dump(serializable_data, file, sort_keys=True, indent=4)
        
        logger.info(f"Finished dumping VllmProfile JSON to {profile_file_path}.")

    def to_json_serializable_dict(self) -> dict:
        '''Returns a json serializable dictionary.'''
        serializable_data = {
            "intervals": {
                str(interval): tokens_to_runtime.to_dict()
                for interval, tokens_to_runtime in self.intervals.items()
            },

            "uses_cuda": self.uses_cuda,

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
        return serializable_data

    @classmethod
    def load(cls, profile_file_path: str):
        '''Loads VllmProfile instance from JSON file.'''
        logger.info(f"Loading VllmProfile from JSON at {profile_file_path}.")
        
        with open(profile_file_path, 'r') as file:
            data = json.load(file)
        
        return cls.load_from_dict(data)
    
    @classmethod
    def load_from_dict(cls, data: str):
        '''Loads VllmProfile from json serializable dict.'''
        # Convert the loaded data back to the original intervals format
        interval_data = data.get("intervals", {})
        intervals = {
            eval(interval): TokensToRuntime.from_dict(tokens_to_runtime)
            for interval, tokens_to_runtime in interval_data.items()
        }
        
        uses_cuda = data["uses_cuda"]

        block_size = data["block_size"]
        num_gpu_blocks = data["num_gpu_blocks"]
        num_watermark_blocks = data["num_watermark_blocks"]

        model = data["model"]
        model_name = data["model_name"]
        max_num_batched_tokens = data["max_num_batched_tokens"]
        max_model_len = data["max_model_len"]
        max_num_seqs = data["max_num_seqs"]
        input_token_upper_limit = data["input_token_upper_limit"]

        logger.info("VllmProfile loaded successfully.")
        return cls(
            intervals=intervals,
            uses_cuda=uses_cuda,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_watermark_blocks=num_watermark_blocks,
            model=model,
            model_name=model_name,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            input_token_upper_limit=input_token_upper_limit)

    def predict_forward_pass_runtime(self, num_tokens: int) -> int:
        '''Returns forward pass runtime in ms.'''
        assert num_tokens > 0, "Number of tokens to process must be greater than zero."

        # Loop through self.intervals until the interval contains num_tokens_to_process.
        for interval, tokens_to_runtime in self.intervals.items():
            interval_start, interval_end = interval
            if interval_start <= num_tokens <= interval_end:
                runtime = tokens_to_runtime.calculate(num_tokens)
                logger.info(f"predict_cuda_graph_launch_runtime({num_tokens}) finds interval {interval} and returns {runtime} ms")
                return runtime
            
        raise Exception(f"{num_tokens} num_tokens_to_process was not included in any intervals.")


class TokensToRuntime:
    '''Represents an interval with a linearly interpolated equation.
    
    Units are seconds.'''
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
        
    def calculate(self, num_tokens):
        ''''''
        return self.slope * num_tokens + self.intercept
    
    def to_dict(self):
        '''Converts the instance to a dictionary format for JSON serialization.'''
        return {
            'slope': self.slope,
            'intercept': self.intercept
        }

    @classmethod
    def from_dict(cls, data):
        '''Creates an instance of TokensToRuntime from a dictionary.'''
        return cls(slope=data['slope'], intercept=data['intercept'])