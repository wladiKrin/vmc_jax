#!/bin/bash -l
#SBATCH -J fidelity_analysis
#SBATCH -o log_fidelity_analysis
#SBATCH -e log_fidelity_analysis
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:a40:1
# #SBATCH -C a100_80
# #SBATCH -d afterany:2054879

module load python/3.9-anaconda
module load cuda/12.6.1
module load nvhpc/23.7
module load openmpi/4.1.6-nvhpc23.7-cuda
conda activate /home/atuin/b245da/b245da12/jvmc

srun python fidelity_analysis.py
