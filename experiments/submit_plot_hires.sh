#!/bin/bash
#SBATCH --job-name=plot_hi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --partition=serial_requeue
source ~/.bashrc
conda activate myenv
cd /n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation
python -u scripts/plot_stadium_hires.py
