#!/bin/bash
#SBATCH --job-name=stad_hi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
source ~/.bashrc
conda activate myenv
module load cuda/12.4.1-fasrc01
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_stadium_hires
python -u scripts/ldos_stadium_hires.py > Results/slurm_stadium_hires/run.out 2> Results/slurm_stadium_hires/run.err
