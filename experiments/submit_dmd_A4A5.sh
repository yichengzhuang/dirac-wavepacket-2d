#!/bin/bash
#SBATCH --job-name=dmd_A4A5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=serial_requeue
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_dmd_A4A5
python -u scripts/dmd_phase_A4A5.py > Results/slurm_dmd_A4A5/run.out 2> Results/slurm_dmd_A4A5/run.err
