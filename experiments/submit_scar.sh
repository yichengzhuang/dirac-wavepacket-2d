#!/bin/bash
#SBATCH --job-name=scar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=serial_requeue
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_scar
python -u scripts/ldos_stadium_scar.py > Results/slurm_scar/run.out 2> Results/slurm_scar/run.err
