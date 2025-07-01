import math
import os
import subprocess
import tempfile

# Define the SLURM script
slurm_script_tmplt = """#!/bin/bash

#SBATCH -o ms/n{{NUM_GPUS}}_sb{{START_BS}}_sf{{SLACK_FAC}}.log
#SBATCH --job-name=n{{NUM_GPUS}}_sb{{START_BS}}_sf{{SLACK_FAC}}
#SBATCH -N {{NUM_NODES}} #Number of Nodes
#SBATCH -n 2 #Number of processes
#SBATCH -c 4 #Number of cores per process
#SBATCH --gres=gpu:volta:2

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=/usr/local/pkg/cuda/cuda-11.8
export PYTHONPATH=$PATHONPATH:/home/gridsan/fkossmann/ensemble_serve/

# Load anaconda module, this includes ray
source /etc/profile
module load anaconda/2023a

# Setting up lists of nodes
((worker_num=$SLURM_NNODES-1))
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
node1=${nodes_array[0]}

# Set IP address/port of head node
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address)
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
ip_head=$ip_prefix':'$port
export ip_head

# Temporary directory for logging
tmpdir='/state/partition1/user/'$USER'/raytmp'
mkdircmd='mkdir -p '$tmpdir

# Start the leader
echo "starting ray leader on "$node1
srun --nodes=1 --ntasks=1 -w $node1 $mkdircmd
srun --nodes=1 --ntasks=1 -w $node1 ray start --temp-dir=$tmpdir --block --head --port=$port &
sleep 5

# Start workers
echo "adding workers"
srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=$node1 ray start -v --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &
sleep 5

cd /home/gridsan/fkossmann/ensemble_serve/workloads/twitter
python twitter_runner.py -l /home/gridsan/fkossmann/ensemble_serve/utils/ms/n{{NUM_GPUS}}_sb{{START_BS}}_sf{{SLACK_FAC}} -n {{NUM_GPUS}} -s 95 -b 0 -pi {{PLAN_INTER}} -pf {{PAC_FAC}} && python print_lat.py /home/gridsan/fkossmann/ensemble_serve/utils/ms/n{{NUM_GPUS}}
"""

START_BS = "{{START_BS}}"
SLACK_FAC = "{{SLACK_FAC}}"
NUM_GPUS = "{{NUM_GPUS}}"
NUM_NODES = "{{NUM_NODES}}"

# Grid search.
slack_factor = [1.0, 0.9, 0.8]
start_bs = [3000]
num_gpus = [2,3]

for n in num_gpus:
    for sf in slack_factor:
        for sb in start_bs:

            slurm_script = slurm_script_tmplt.replace(START_BS, str(sb))
            slurm_script = slurm_script.replace(SLACK_FAC, str(sf))
            slurm_script = slurm_script.replace(NUM_GPUS, str(n))
            slurm_script = slurm_script.replace(NUM_NODES, str(math.ceil(n/2)))

            print(slurm_script)

            # Create a temporary file to store the SLURM script
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
                script_path = tmpfile.name
                tmpfile.write(slurm_script)

            # Submit the script to SLURM using sbatch
            submit_command = ['LLsub', script_path]
            result = subprocess.run(submit_command, capture_output=True, text=True)

            # Print the result of the sbatch command
            print("SLURM submission result:", result.stdout)

            # Optionally, delete the temporary script file if desired
            os.remove(script_path)
