#!/bin/bash
#SBATCH --job-name=cap_study
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=Results/slurm_cap_study/run.out
#SBATCH --error=Results/slurm_cap_study/run.err

mkdir -p Results/slurm_cap_study

source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u scripts/cap_systematic_study.py
