#!/bin/bash -l
#SBATCH -J TDVP_RBMCNN16Ex
#SBATCH -o log_TDVP_RBMCNN16Ex
#SBATCH -e log_TDVP_RBMCNN16Ex
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80
# #SBATCH -d afterany:2054879

module load python/3.9-anaconda
module load cuda/12.6.1
module load nvhpc/23.7
module load openmpi/4.1.6-nvhpc23.7-cuda
conda activate /home/atuin/b245da/b245da12/jvmc

# srun python mainCNN.py inputRBMCNN.txt

srun python mainCNNEx.py --lattice 16 -g -1 --numSamples 100000 --exactRenorm False --numHidden 20 --filterSize 10 --tmax 1.0 --dt 1e-4 --integratorTol 1e-6 --invCutoff 1e-8
