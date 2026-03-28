#!/bin/bash
# Soft-wall Bunimovich stadium — chaotic billiard.
python run.py \
    --potential soft_stadium \
    --a_half 1.0 --b 1.0 \
    --V0 70 --lam 25 --mu 5 \
    --mass 0.0 \
    --k_mag 3.0 \
    --n_prop 8 \
    --outdir output/stadium
