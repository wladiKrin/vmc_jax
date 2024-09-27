#!/bin/bash
#SBATCH --job-name=vmc_Ising20MLowerInt
#SBATCH --output=log_vmcIsing20MLowerInt
#SBATCH --error=log_vmcIsing20MLowerInt
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
srun $CONDA_PREFIX/bin/python main.py input.txt
echo 'job ended'
