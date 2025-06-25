#!/bin/bash

#SBATCH --job-name=test_ft
#SBATCH --output=logs/test_ft_%j.out
#SBATCH --error=logs/test_ft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=IscrB_medit


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # Set OpenMP threads per task
export NCCL_DEBUG=INFO

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MACHINE_RANK=$SLURM_NODEID
export MASTER_PORT=32456


export VLLM_TORCH_COMPILE_CACHE_DIR=$SCRATCH/vllm_cache
export XDG_CACHE_HOME=$SCRATCH/.cache  # opzionale ma consigliato


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

module purge
module load cuda/11.8
module load python/3.11.6--gcc--8.5.0
module load openmpi

#source PRM_FT_env/bin/activate
#accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 finetuneqwen.py

#srun --mpi=pmi2 accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 finetuneqwen.py

# srun --ntasks=2 --nodes=2 bash -c '
# source PRM_FT_env/bin/activate
# accelerate launch \
#   --multi_gpu \
#   --num_processes 8 \
#   --num_machines 2 \
#   --machine_rank $SLURM_PROCID \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port 32456 \
#   --mixed_precision fp16 \
#   test_process_bench_base.py
# '
source PRM_FT_env/bin/activate
python test_process_bench.py

# torchrun \
#   --nproc_per_node=4 \
#   --nnodes=1 \
#   test_process_bench_base.py


#srun --mpi=pmi2 accelerate launch \
#--multi_gpu \
#--num_processes 8 \
#--num_machines 2 \
#--machine_rank $SLURM_PROCID \
#--main_process_ip $MASTER_ADDR \
#--main_process_port 32456 \
#--mixed_precision fp16 \
#finetuneqwen.py
