#!/bin/bash
#SBATCH --job-name=study_sw
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=Results/slurm_study_softwall/run.out
#SBATCH --error=Results/slurm_study_softwall/run.err

mkdir -p Results/slurm_study_softwall
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u scripts/study_softwall_params.py
