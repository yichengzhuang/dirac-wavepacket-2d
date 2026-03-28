#!/bin/bash
#SBATCH --job-name=narrow_sym
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --partition=serial_requeue

source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_narrow_sym
python -u scripts/narrow_sym_plot.py > Results/slurm_narrow_sym/run.out 2> Results/slurm_narrow_sym/run.err
