#!/bin/bash
#SBATCH --job-name=dirac_ldos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --partition=serial_requeue

# =========================
# Log files
# =========================
OUTDIR="Results/slurm_ldos"
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

python -u scripts/ldos_average.py \
    > $RUN_OUT 2> $RUN_ERR
