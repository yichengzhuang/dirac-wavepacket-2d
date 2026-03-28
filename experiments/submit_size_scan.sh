#!/bin/bash
#SBATCH --job-name=size_scan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=serial_requeue
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_size_scan figures/size_scan
python -u scripts/size_scan.py > Results/slurm_size_scan/run.out 2> Results/slurm_size_scan/run.err
