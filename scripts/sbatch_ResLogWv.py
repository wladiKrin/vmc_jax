#!/bin/bash
#SBATCH     -J TDVP_ResNet2DWvLognh
#SBATCH -o log_TDVP_ResNet2DWvLognh%a
#SBATCH -e log_TDVP_ResNet2DWvLognh%a
#SBATCH --ntasks-per-node=2
#SBATCH -p pgi-8-gpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --gres=gpu:a100:2
#SBATCH --array=0-0
# #SBATCH -d afterok:58903_1

echo 'job started'
conda activate jvmc
srun python mainRes2DWvLog.py --lattice 16 -g ${SLURM_ARRAY_TASK_ID} --numSamples 40000 --exactRenorm False --numHidden 10 --filterSize 2 --tmax ${SLURM_ARRAY_TASK_ID} --dt 1e-4 --integratorTol 1e-5 --invCutoff 1e-8
# srun source ~/.bashrc; conda activate jvmc; python mainRes2DWvLog.py --lattice 3 -g ${SLURM_ARRAY_TASK_ID} --numSamples 2000 --exactRenorm False --numHidden 6 --filterSize 2 --tmax 2.0 --dt 1e-4 --integratorTol 1e-4 --invCutoff 1e-8
echo 'job ended'
