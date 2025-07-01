from vllm import LLM, SamplingParams
import time

# Sample prompts.
prompts = [
    "Hello, my name is"*100
]*500
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=200, max_tokens=200)

# Create an LLM.
llm = LLM(model="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16",
          gpu_memory_utilization = 0.8,
          tensor_parallel_size=4,
          max_num_seqs=1024,
          max_num_batched_tokens=1024,
          enforce_eager=False,
          max_model_len=1024)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

start = time.time()
outputs = llm.generate(prompts, sampling_params)
print(time.time() - start)

fac = [0.6, 0.7, 0.8, 0.9]
mem = [52, 84, 116, 148]
time = [119.28597688674927, 102.60819840431213, 97.70034646987915, 83.39708590507507]

# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")