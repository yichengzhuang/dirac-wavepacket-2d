#!/bin/bash
# High-resolution run on GPU: 48 directions, 512 grid, 500 energy points.
python run.py \
    --backend torch_cuda \
    --n 512 \
    --n_steps 30000 \
    --n_prop 48 \
    --n_E 500 \
    --outdir output/highres
