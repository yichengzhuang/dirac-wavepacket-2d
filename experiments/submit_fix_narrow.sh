#!/bin/bash
#SBATCH --job-name=fdm_fix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=serial_requeue

OUTDIR="Results/slurm_fix_narrow"
mkdir -p $OUTDIR

source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
python -u scripts/fdm_fix_and_narrow.py > $OUTDIR/run.out 2> $OUTDIR/run.err
