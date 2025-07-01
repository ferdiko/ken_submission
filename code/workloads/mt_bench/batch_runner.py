import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
import signal
import time


def shutdown_vllm():
    """
    vLLM doesn't shut down properly. This is a hack to free up all memory
    """
    processes = [p for p in psutil.process_iter(['pid', 'name']) if p.info['name'] == 'pt_main_thread']
    processes_sorted = sorted(processes, key=lambda p: p.info['pid'], reverse=True)

    # Kill each process
    for process in processes_sorted:
        pid = process.info['pid']
        try:
            os.kill(pid, signal.SIGTERM)  # or signal.SIGKILL for a forced kill
        except Exception as e:
            print(f"Could not kill process with PID {pid}: {e}")


# Function to execute a command
def run_command(command):
    process = subprocess.Popen(command)
    return process


# cert_thresh_options = [0.2, 0.4, 0.6, 0.8]
# model1_options = ["3b", "8b"]
# model2_options = ["70b"]
# queued_thresh_mult_options = [2, 3, 4]
# queued_thresh_options = [400, 2000, 4000]
# qps_mult_options = [0.55]
# tokens_per_turn_options = [1600] #, 1600] # 800 did worse based on 3 runs
# num_secs_options = [42]
# small_iters_options = [1, 3]
# trigger2_options = [2, 3]


cert_thresh_options = [0.0]
model1_options = ["3b"]
model2_options = ["70b"]
queued_thresh_mult_options = [1]
queued_thresh_options = [100, 200, 800, 1200, 2400, 4000, 8000, 12000]
qps_mult_options = [0.15]
tokens_per_turn_options = [1600] #, 1600] # 800 did worse based on 3 runs
num_secs_options = [42]
small_iters_options = [1]
trigger2_options = [1]


# Create a ThreadPoolExecutor with a maximum of 2 workers
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit commands to the executor
    processes = []
    toggle = 0  # To alternate between commands
    for trigger2, model1, model2, num_secs, tokens_per_turn, queued_thresh_mult, queued_thresh, small_iters, qps_mult, cert_thresh in itertools.product(
                                                                                                          trigger2_options,
                                                                                                          model1_options,
                                                                                                          model2_options,
                                                                                                          num_secs_options,
                                                                                                          tokens_per_turn_options, 
                                                                                                          queued_thresh_mult_options,
                                                                                                          queued_thresh_options,
                                                                                                          small_iters_options,
                                                                                                          qps_mult_options,
                                                                                                          cert_thresh_options):
        if queued_thresh == -1:
            queued_thresh_mult = 1
            trigger2 = 1
            small_iters = 1


        # if int(model1[:-1]) >= int(model2[:-1]):
        #     continue

        # Apply vLLM-specific filtering
        # if cert_thresh == 0:
        #     if queued_thresh != queued_thresh_options[0] or small_iters != small_iters_options[0] or queued_thresh_mult != queued_thresh_mult_options[0] or trigger2 != trigger2_options[0]:
        #         continue
        #     else:
        #         queued_thresh = 9999999

        # if trigger2 == 1 and queued_thresh_mult != queued_thresh_mult_options[0]:
        #     continue

        # Base command
        command = [
            "python",
            "sample_qps_runner.py",
            f"--model1={model1}",
            f"--model2={model2}",
            f"--cert-thresh={cert_thresh}",
            f"--queued-thresh={queued_thresh}",
            f"--qps-mult={qps_mult}",
            f"--tokens-per-turn={tokens_per_turn}",
            f"--num-secs={num_secs}",
            f"--small-iters={small_iters}",
            f"--run-no={toggle%4}"
        ]
        toggle += 1

        # # MS runner.
        # command = [
        #     "python",
        #     "ms_qps_runner.py",
        #     f"--model1={model1}",
        #     f"--model2={model2}",
        #     f"--cert-thresh={cert_thresh}",
        #     f"--queued-thresh={queued_thresh}",
        #     f"--qps-mult={qps_mult}",
        #     f"--tokens-per-turn={tokens_per_turn}",
        #     f"--num-secs={num_secs}",
        #     f"--small-iters={small_iters}"
        # ]

        # # MS runner.
        # command = [
        #     "python",
        #     "alpaserve_qps_runner.py",
        #     f"--model1={model1}",
        #     f"--model2={model2}",
        #     f"--cert-thresh={cert_thresh}",
        #     f"--queued-thresh={queued_thresh}",
        #     f"--qps-mult={qps_mult}",
        #     f"--tokens-per-turn={tokens_per_turn}",
        #     f"--num-secs={num_secs}",
        #     f"--small-iters={small_iters}"
        # ]

        # if toggle:
        #     command.append("--second")
        # toggle = not toggle


        #################### 8 GPUs
        # Base command
        # command = [
        #     "python",
        #     "multi_sample_qps_runner.py",
        #     f"--model1={model1}",
        #     f"--model2={model2}",
        #     f"--cert-thresh={cert_thresh}",
        #     f"--queued-thresh={queued_thresh}",
        #     f"--queued-thresh-mult={queued_thresh_mult}",
        #     f"--qps-mult={qps_mult}",
        #     f"--tokens-per-turn={tokens_per_turn}",
        #     f"--num-secs={num_secs}",
        #     f"--small-iters={small_iters}",
        #     f"--trigger2={trigger2}"
        # ]



        # Launch the command in parallel
        processes.append(run_command(command))

        # If 2 tasks are running, wait for them to complete
        if len(processes) == 4:
            for process in processes:
                process.wait()  # Ensure all processes finish
            processes = []  # Reset the process list
            shutdown_vllm()  # Safely cleanup resources after completion


    # Wait for any remaining tasks to finish
    for process in processes:
        process.wait()
    shutdown_vllm()  # Final cleanup
