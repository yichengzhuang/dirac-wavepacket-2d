"""Return signal C(r, t) of a wavepacket frame on the Form-P potential hill, along its two axes.

Writes formp_signal.npz:  t (fs), x, y (nm), C_x, C_y (1 / nm^2, one valley).
"""

import numpy as np
import torch

from dirac import grid, form_p, absorber, Propagator
from ldos import frame, frame_signal

# grid and absorber
n, L = 129, 320.0                       # points per side, box side (nm)
x_on, y_on, W0 = 130.0, 90.0, 472.86    # absorber onsets (nm), strength (meV)

# time stepping (fs)
dt, T, every = 0.16, 2400.0, 10

# wavepacket frame
sigma = 8.0                     # wavepacket width (nm)
x_half, y_half = 120.0, 78.0    # launch rectangle (nm)
k_max = 0.6                     # launch momenta |k0| < k_max (1/nm)
n_points = 16384                # phase-space points, two band spinors each

device = "cuda" if torch.cuda.is_available() else "cpu"

x, X, Y, KX, KY = grid(n, L)
V = form_p(X, Y)
W = absorber(X, Y, x_on, y_on, W0)
prop = Propagator(V, W, KX, KY, dt, device)

# probe pixels: the major (x) axis, then the minor (y) axis; x is also the y grid
c = n // 2
ix = np.flatnonzero(abs(x) <= 100)
iy = np.flatnonzero(abs(x) <= 65)
rows = np.r_[np.full(ix.size, c), iy]
cols = np.r_[ix, np.full(iy.size, c)]
probes = (rows, cols)

points, w = frame(n_points, x_half, y_half, k_max)
n_steps = round(T / dt)
C = frame_signal(prop, X, Y, points, w, sigma, probes, n_steps, every)
t = every * dt * np.arange(len(C))

np.savez("formp_signal.npz", t=t,
         x=x[ix], C_x=C[:, :ix.size],
         y=x[iy], C_y=C[:, ix.size:])
