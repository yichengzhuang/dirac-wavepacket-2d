"""Local density of states from a frame of Dirac wavepackets.

Gaussian wavepackets |r0, k0, s>, with s = +1, -1 the two band spinors at k0,
resolve the identity:

    sum_j w |j><j| = 1,    w = (phase-space area) / (2 pi)^2 / (number of points r0, k0).

Inserted into the local spectral function this gives

    LDOS(r, E) = (1 / pi hbar) Re int_0^T dt exp(i E t / hbar) f(t) C(r, t),
    C(r, t)    = w sum_j sum_sublattice psi_j(r, t) psi_j(r, 0)^*,

with psi_j(t) = exp(-i H t / hbar) |j>.  The window f(t) = exp(-eta t / hbar) gives
-Im G(r, r; E + i eta) / pi, a Lorentzian broadening; f(t) = exp(-(sigma t / hbar)^2 / 2)
gives a Gaussian broadening of standard deviation sigma.
"""

import numpy as np
import torch
from scipy.stats import qmc

from dirac import HBAR


def frame(n_points, x_half, y_half, k_max, seed=0):
    """Quasi-random phase-space points (x0, y0, kx0, ky0) and the frame weight w.

    r0 is uniform in |x| < x_half, |y| < y_half; k0 is uniform in the disk |k| < k_max.
    """
    u = qmc.Sobol(4, seed=seed).random(n_points)
    x0 = x_half * (2 * u[:, 0] - 1)
    y0 = y_half * (2 * u[:, 1] - 1)
    k0 = k_max * np.sqrt(u[:, 2])
    phi = 2 * np.pi * u[:, 3]
    kx0 = k0 * np.cos(phi)
    ky0 = k0 * np.sin(phi)
    points = np.stack([x0, y0, kx0, ky0], axis=1)

    area = (4 * x_half * y_half) * (np.pi * k_max**2)
    w = area / (2 * np.pi)**2 / n_points
    return points, w


def wavepackets(X, Y, points, s, sigma):
    """Normalized spinors g(r) (1, s e^{i phi_k}) / sqrt(2) for a batch of phase-space points,

    g(r) = exp(i k0.r - |r - r0|^2 / 2 sigma^2).
    """
    x0, y0, kx, ky = (p[:, None, None] for p in points.T)
    envelope = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    g = envelope * np.exp(1j * (kx * X + ky * Y))

    dx = X[0, 1] - X[0, 0]
    g /= np.sqrt(2 * np.sum(abs(g)**2, axis=(1, 2), keepdims=True) * dx**2)
    return g, s * np.exp(1j * np.arctan2(ky, kx)) * g


def return_signal(prop, a, b, probes, n_steps, every):
    """sum over the batch of psi(r, t) psi(r, 0)^* at the probe pixels, recorded every `every` steps."""
    iy, ix = probes
    a0 = a[:, iy, ix].conj()
    b0 = b[:, iy, ix].conj()

    def overlap(a, b):
        return (a[:, iy, ix] * a0 + b[:, iy, ix] * b0).sum(0)

    C = [overlap(a, b)]
    for step in range(1, n_steps + 1):
        a, b = prop.step(a, b)
        if step % every == 0:
            C.append(overlap(a, b))
    return torch.stack(C).cpu().numpy().astype(complex)


def frame_signal(prop, X, Y, points, w, sigma, probes, n_steps, every, batch=256):
    """C(r, t) of the full frame: both band spinors at every phase-space point."""
    C = 0
    for start in range(0, len(points), batch):
        p = points[start:start + batch]
        for s in (1, -1):
            a, b = wavepackets(X, Y, p, s, sigma)
            a = torch.as_tensor(a, device=prop.device).to(prop.dtype)
            b = torch.as_tensor(b, device=prop.device).to(prop.dtype)
            C = C + return_signal(prop, a, b, probes, n_steps, every)
        print(f"{start + len(p)} / {len(points)} phase-space points", flush=True)
    return w * C


def ldos(t, C, E, width, kernel="lorentz"):
    """LDOS(r, E) in 1 / (nm^2 meV) from C(r, t) of shape (time, probe), uniform t (trapezoidal rule).

    kernel = "lorentz": -Im G(r, r; E + i width) / pi.
    kernel = "gauss":   Gaussian broadening of standard deviation `width`.
    """
    if kernel == "lorentz":
        f = np.exp(-width * t / HBAR)
    else:
        f = np.exp(-(width * t / HBAR)**2 / 2)

    dt = t[1] - t[0]
    K = np.exp(1j * np.outer(E, t) / HBAR) * f * dt
    K[:, [0, -1]] /= 2
    return (K @ C).real / (np.pi * HBAR)
