import os
import subprocess
import tempfile

# Define the SLURM script
slurm_script_tmplt = """#!/bin/bash

#SBATCH -o 03-13-runs/sf95_n4_maxbs0_pf{{PAC_FAC}}_pi{{PLAN_INTER}}.log
#SBATCH --job-name=sf95_n4_maxbs0_pf{{PAC_FAC}}_pi{{PLAN_INTER}}
#SBATCH -N 2 #Number of Nodes
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

cd /home/gridsan/fkossmann/ensemble_serve/baselines/cocktail
python twitter_runner.py -l /home/gridsan/fkossmann/ensemble_serve/baselines/cocktail/03-13-runs/sf95_n4_maxbs0_pf{{PAC_FAC}}_pi{{PLAN_INTER}} -n 4 -s 95 -b 0 -pi {{PLAN_INTER}} -pf {{PAC_FAC}} && python print_lat.py /home/gridsan/fkossmann/ensemble_serve/baselines/cocktail/03-13-runs/sf95_n4_maxbs0_pf{{PAC_FAC}}_pi{{PLAN_INTER}}
"""

PLAN_INTER = "{{PLAN_INTER}}"
PAC_FAC = "{{PAC_FAC}}"

# Grid search.
planning_interval = [10, 60, 300]
packing_factor = [900, 1000, 1200]

for pi in planning_interval:
    for pf in packing_factor:

        slurm_script = slurm_script_tmplt.replace(PLAN_INTER, str(pi))
        slurm_script = slurm_script.replace(PAC_FAC, str(pf))

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
