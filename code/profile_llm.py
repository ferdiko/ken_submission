from vllm import LLM, SamplingParams
from torch.profiler import ProfilerActivity, profile


# Sample prompts.
prompts = [
    "Hello, my name is The president of the United States is The capital of France is The future of AI is",
    "hehehe"
]

# Create a sampling params object.
sampling_params = SamplingParams(best_of=1,
    temperature=0.0,
    top_p=1,
    top_k=1,
    max_tokens=28,
    min_tokens=28,
    presence_penalty=0,
    frequency_penalty=0)


# Create an LLM.
llm = LLM(#model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8", 
            # model="meta-llama/Meta-Llama-3-8B",
            model="meta-llama/Llama-3.2-1B-Instruct",
        #   model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        # model="meta-llama/Llama-3.1-70B-Instruct",
        quantization="fp8",
        enforce_eager=True,
          tensor_parallel_size=1,
          max_model_len=1000,
          enable_chunked_prefill=False,
        max_num_batched_tokens=1024,
        # max_num_seqs=1024,
        max_num_seqs=1)
        
        #  )
          #tensor_parallel_size=2, execute_eager=True) #meta-llama/Meta-Llama-3-8B")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

outputs = llm.generate(prompts, sampling_params)
