#!/bin/bash
#SBATCH --job-name=dirac_fdm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# =========================
# Log files
# =========================
OUTDIR="Results/slurm_fdm"
mkdir -p $OUTDIR
RUN_OUT="$OUTDIR/run.out"
RUN_ERR="$OUTDIR/run.err"

# =========================
# Load environment
# =========================
source ~/.bashrc
conda activate myenv
module load cuda/12.4.1-fasrc01

# =========================
# Run
# =========================
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation

python -u scripts/fdm_extract.py \
    > $RUN_OUT 2> $RUN_ERR
