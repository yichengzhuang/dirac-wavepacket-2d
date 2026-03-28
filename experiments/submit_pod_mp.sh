#!/bin/bash
#SBATCH --job-name=pod_mp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=serial_requeue
source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
mkdir -p Results/slurm_pod_mp
python -u scripts/pod_mp_extract.py > Results/slurm_pod_mp/run.out 2> Results/slurm_pod_mp/run.err
