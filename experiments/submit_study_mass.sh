#!/bin/bash
#SBATCH --job-name=study_mass
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=Results/slurm_study_mass/run.out
#SBATCH --error=Results/slurm_study_mass/run.err

mkdir -p Results/slurm_study_mass

source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u scripts/study_mass.py
