#!/bin/bash
#SBATCH --job-name=scripts/implicit/compare_to_explicit
#SBATCH --output=scripts/0encap_folder/implicit/compare_to_explicit/slurm_files/log.slurm
#SBATCH --error=scripts/0encap_folder/implicit/compare_to_explicit/slurm_files/log.slurm
#SBATCH --partition=pgi-8-gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun bash scripts/0encap_folder/implicit/compare_to_explicit/slurm_files/run.sh

