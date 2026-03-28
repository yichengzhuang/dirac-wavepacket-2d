#!/bin/bash
#SBATCH --job-name=ldos_shapes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=serial_requeue
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_ldos_shapes
python -u scripts/ldos_shapes.py > Results/slurm_ldos_shapes/run.out 2> Results/slurm_ldos_shapes/run.err
