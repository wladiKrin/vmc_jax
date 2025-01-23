#!/bin/bash -l
#SBATCH -J SR_eval
#SBATCH -o log_SR_eval
#SBATCH -e log_SR_eval
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:a40:1
##SBATCH -C a100_80
# #SBATCH -d afterany:2054879

module load python/3.9-anaconda
module load cuda/12.6.1
module load nvhpc/23.7
module load openmpi/4.1.6-nvhpc23.7-cuda
conda activate /home/atuin/b245da/b245da12/jvmc

srun python SR_evaluation_update.py
