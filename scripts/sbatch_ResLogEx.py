#!/bin/bash -l
#SBATCH -J TDVP_ResNet20ExLog
#SBATCH -o log_TDVP_ResNet20ExLog
#SBATCH -e log_TDVP_ResNet20ExLog
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:a100:4
# #SBATCH -C a100_80
# #SBATCH -d afterany:2054879

module load python/3.9-anaconda
module load cuda/12.6.1
module load nvhpc/23.7
module load openmpi/4.1.6-nvhpc23.7-cuda
conda activate /home/atuin/b245da/b245da12/jvmc

srun mpirun python mainResExLog.py --lattice 20 -g -1 --numSamples 40000 --exactRenorm False --numHidden 12 --filterSize 3 --tmax 2.0 --dt 1e-4 --integratorTol 1e-4 --invCutoff 1e-8
