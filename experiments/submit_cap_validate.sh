#!/bin/bash
#SBATCH --job-name=cap_validate
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=Results/slurm_cap_validate/run.out
#SBATCH --error=Results/slurm_cap_validate/run.err

mkdir -p Results/slurm_cap_validate

source ~/.bashrc
conda activate myenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Step A: Reflection test ==="
python -u scripts/cap_reflection_test.py

echo ""
echo "=== Step B: Zone density verification ==="
python -u scripts/cap_zone_verify.py
