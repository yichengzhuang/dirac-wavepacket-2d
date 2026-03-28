#!/bin/bash
# Massive Dirac (m=1) — well-separated resonances, sharp LDOS peaks.
python run.py \
    --potential harmonic_ellipse \
    --mass 1.0 \
    --n_prop 8 \
    --outdir output/massive_ellipse
