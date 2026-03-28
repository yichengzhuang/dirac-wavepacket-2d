#!/bin/bash
#SBATCH --job-name=stad_ext
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=serial_requeue
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_stadium_ext
python -u /tmp/stadium_hires.py > Results/slurm_stadium_ext/run.out 2> Results/slurm_stadium_ext/run.err
