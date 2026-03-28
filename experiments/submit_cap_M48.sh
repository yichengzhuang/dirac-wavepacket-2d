#!/bin/bash
#SBATCH --job-name=cap_M48
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=Results/slurm_cap_M48/run.out
#SBATCH --error=Results/slurm_cap_M48/run.err

mkdir -p Results/slurm_cap_M48

source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u scripts/cap_convergence_M48.py
