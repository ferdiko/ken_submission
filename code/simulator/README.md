# VllmSimulator
A request latency simulator for vLLM that can be run without GPUs. It consists of two components:
1. A profiler that measures runtimes of different LLM-hardware configurations and generates a specific profile.
2. A simulator that leverages the runtime profiles to give an accurate latency estimate for the specified LLM and hardware combination.

Use private [vLLM repository](https://github.com/plantbud/new_vllm) along with the simulator.

## How to use
### Generate a profile
Modify the following file and run it using the command:
```python generate_example_profile.py```
### Simulate a workload
Once a profile is generated, you can use the python object or json to simulate a workload. An example script with workloads is in `run.py`.
### Run vLLM with the same framework that simulator uses
In case you want to debug the LLM and not run the whole simulator, use `run_workload_trace.py`. It uses the newly added LLM function, run_workload_trace.
### Verify metrics
This creates a profile and compares the per request metrics between vLLM and my simulator. Use `verify_metrics.py`.
### Verify tokens
If you want to debug per forward pass number of tokens, use `verify_tokens.py`. I’m not sure what state this in right now, I may have disabled the loggers needed for this…

## Known limitations
- TTFT and TGT are almost always underestimated. The absolute difference in real and simulated metrics scales linearly with number of forward passes included in the TGT, so a request with less forward passes will have less of a difference with tgt. and a request with more than 200 forward passes will have more than .04 s difference in TGT.

## Further Work
- [ ] Implement other restrictions to starting requests (these should throw errors)
  - [ ] Max_model_len
  - [ ] Max_num_seqs
  - [ ] Max_tokens (maximum number of output tokens per sequence)
  - [ ] A request prefill that is too large to ever fit in the cache (should be handled by 1 of the above, but currently will just throw a 0 tokens in a forward pass error)
- [ ] Fix cache_simulator and request_gpu_block_tracker to not differentiate between gpu and watermark blocks
- [ ] Deprecate vllm simulator api that allows input with query arrival timestamps (will only use queries_per_iter)
