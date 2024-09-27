#!/bin/bash
#SBATCH --job-name=vmc_test
#SBATCH --output=log_vmc
#SBATCH --error=log_vmc
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -p pgi-8-gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1 
# #SBATCH --array=1-3
# #SBATCH -d afterok:16387_3


export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo 'job started'
srun source ~/.bashrc; conda activate jvmc2; python src/main.py input.txt
echo 'job ended'
