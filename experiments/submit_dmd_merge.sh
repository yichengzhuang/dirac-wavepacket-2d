#!/bin/bash
#SBATCH --job-name=dmd_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=serial_requeue
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_dmd_merge
python -u scripts/dmd_independent_merge.py > Results/slurm_dmd_merge/run.out 2> Results/slurm_dmd_merge/run.err
