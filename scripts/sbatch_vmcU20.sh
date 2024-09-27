#!/bin/bash
#SBATCH --job-name=vmc_Ising20U
#SBATCH --output=log_vmcIsing20U
#SBATCH --error=log_vmcIsing20U
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
srun source ~/.bashrc; conda activate jvmc2; $CONDA_PREFIX/bin/python src/mainUnif.py input20.txt
echo 'job ended'
