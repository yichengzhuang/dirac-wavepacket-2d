#!/bin/bash
#SBATCH --job-name=scar_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
source ~/.bashrc
conda activate myenv
module load cuda/12.4.1-fasrc01
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_scar_gpu
python -u scripts/ldos_stadium_scar.py > Results/slurm_scar_gpu/run.out 2> Results/slurm_scar_gpu/run.err
