#!/bin/bash
#SBATCH --job-name=study_grid
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=Results/slurm_study_grid/run.out
#SBATCH --error=Results/slurm_study_grid/run.err

mkdir -p Results/slurm_study_grid
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u scripts/study_grid_resolution.py
