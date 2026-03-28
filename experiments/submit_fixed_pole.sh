#!/bin/bash
#SBATCH --job-name=fixed_pole
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=serial_requeue

# =========================
# Log files
# =========================
OUTDIR="Results/slurm_fixed_pole"
mkdir -p $OUTDIR
RUN_OUT="$OUTDIR/run.out"
RUN_ERR="$OUTDIR/run.err"

# =========================
# Load environment
# =========================
source ~/.bashrc
conda activate myenv

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# =========================
# Run
# =========================
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation

python -u scripts/fixed_pole_recon.py \
    > $RUN_OUT 2> $RUN_ERR
