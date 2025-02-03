#!/bin/bash
#SBATCH     -J TDVP_RBMCNN2DWvLog
#SBATCH -o log_TDVP_RBMCNN2DWvLog%a
#SBATCH -e log_TDVP_RBMCNN2DWvLog%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -p pgi-8-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres=gpu:a100:1 
#SBATCH --array=1-3%1
# #SBATCH -d afterok:58903_1

echo 'job started'
srun source ~/.bashrc; conda activate jvmc; python mainCNN2DWvLog.py --lattice 8 -g ${SLURM_ARRAY_TASK_ID} --numSamples 40000 --exactRenorm False --numHidden 10 --filterSize 8 --tmax 2.0 --dt 1e-4 --integratorTol 1e-4 --invCutoff 1e-8
echo 'job ended'
