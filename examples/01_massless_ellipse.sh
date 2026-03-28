#!/bin/bash
# Massless Dirac in a harmonic ellipse (default parameters).
# ~17 min on CPU, ~1 min on GPU.
python run.py \
    --potential harmonic_ellipse \
    --k_harm 3.0 --a 2.0 --b 1.2 \
    --mass 0.0 \
    --n_prop 8 \
    --outdir output/massless_ellipse
