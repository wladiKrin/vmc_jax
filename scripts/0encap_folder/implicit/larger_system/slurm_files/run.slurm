#!/bin/bash
#SBATCH --job-name=implicit/larger_system
#SBATCH --output=0encap_folder/implicit/larger_system/slurm_files/log.slurm
#SBATCH --error=0encap_folder/implicit/larger_system/slurm_files/log.slurm
#SBATCH --partition=pgi-8-gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun bash 0encap_folder/implicit/larger_system/slurm_files/run.sh

