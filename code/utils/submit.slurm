#!/bin/bash

#SBATCH -o run-%j.log
#SBATCH --job-name=ray_4servers
#SBATCH -N 3 #Number of Nodes
#SBATCH -c 2 #Number of cores per process
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

sleep 5

# Start workers
echo "adding workers"
srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=$node1 ray start -v --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &
sleep 5

cd /home/gridsan/fkossmann/ensemble_serve/workloads/llama/ 
python hellaswag_runner.py -g /home/gridsan/fkossmann/ensemble_serve/offline/cached_runs/hellaswag/6gpu_slack08 -l /home/gridsan/fkossmann/ensemble_serve/utils/6gpu_slack08_thresh8_new
