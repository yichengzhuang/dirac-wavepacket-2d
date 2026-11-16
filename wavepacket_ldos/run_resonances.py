"""Two wavepacket launches on the Form-P hill, for its resonances.

Writes formp_launches.npz:  t (fs), x, y (nm), launches (x0, y0, k, angle), and the
wavefunctions psi0 (launch, sublattice, y, x) and psi (launch, time, sublattice, y, x)
over the recorded region, in 1 / nm.
"""

import numpy as np
import torch

from dirac import grid, form_p, absorber, Propagator
from ldos import wavepackets

# grid and absorber
n, L = 129, 320.0                       # points per side, box side (nm)
x_on, y_on, W0 = 130.0, 90.0, 472.86    # absorber onsets (nm), strength (meV)

# time stepping (fs)
dt, T, every = 0.16, 2400.0, 10

# recorded region (nm)
x_max, y_max = 105.0, 70.0

# launches: centre x0, y0 (nm), momentum k (1/nm) at an angle (deg), hole-band spinor
launches = np.array([[25.0, 12.0, 0.15, 30.0],
                     [-40.0, 20.0, 0.15, 160.0]])
sigma = 4.0                             # wavepacket width (nm)

device = "cuda" if torch.cuda.is_available() else "cpu"

x, X, Y, KX, KY = grid(n, L)
V = form_p(X, Y)
W = absorber(X, Y, x_on, y_on, W0)
prop = Propagator(V, W, KX, KY, dt, device)

x0, y0, k, angle = launches.T
phi = np.deg2rad(angle)
points = np.stack([x0, y0, k * np.cos(phi), k * np.sin(phi)], axis=1)
a0, b0 = wavepackets(X, Y, points, -1, sigma)

iy = np.flatnonzero(abs(x) <= y_max)
ix = np.flatnonzero(abs(x) <= x_max)
ys, xs = slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)

a = torch.as_tensor(a0, device=device).to(prop.dtype)
b = torch.as_tensor(b0, device=device).to(prop.dtype)
psi = [torch.stack([a[:, ys, xs], b[:, ys, xs]], dim=1)]
for step in range(1, round(T / dt) + 1):
    a, b = prop.step(a, b)
    if step % every == 0:
        psi.append(torch.stack([a[:, ys, xs], b[:, ys, xs]], dim=1))

psi = torch.stack(psi, dim=1).cpu().numpy()
psi0 = np.stack([a0, b0], axis=1)[:, :, ys, xs]
t = every * dt * np.arange(psi.shape[1])

np.savez("formp_launches.npz", t=t, x=x[ix], y=x[iy], launches=launches,
         psi0=psi0, psi=psi)
