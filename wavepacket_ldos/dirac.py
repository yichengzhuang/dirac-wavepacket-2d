"""Massless 2D Dirac equation on a periodic FFT grid.

    H = hbar v (sigma_x p_x + sigma_y p_y) + V(r) - i W(r)

Units: nm, meV, fs.  W is an absorbing potential near the box edge.
"""

import numpy as np
import torch

HBAR = 658.2119569      # meV fs
HBAR_V = 788.096        # meV nm, hbar v_F of the reference tight-binding model


def grid(n, L):
    """Square n x n grid of side L; odd n puts x = 0 on the grid."""
    dx = L / n
    x = (np.arange(n) - n // 2) * dx
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    X, Y = np.meshgrid(x, x)
    KX, KY = np.meshgrid(k, k)
    return x, X, Y, KX, KY


def form_p(X, Y, V0=146.0, cx=0.0288, cy=0.0677, floor=-600.0, width=15.0):
    """Anisotropic potential hill V0 - cx x^2 - cy y^2, smoothly floored far outside."""
    V = V0 - cx * X**2 - cy * Y**2
    return floor + width * np.logaddexp(0.0, (V - floor) / width)


def absorber(X, Y, x_on, y_on, W0):
    """W0 (s_x^2 + s_y^2), with s rising linearly from 0 at the onset to 1 at the box edge."""
    edge = X.max() + (X[0, 1] - X[0, 0]) / 2
    sx = np.clip((abs(X) - x_on) / (edge - x_on), 0, 1)
    sy = np.clip((abs(Y) - y_on) / (edge - y_on), 0, 1)
    return W0 * (sx**2 + sy**2)


class Propagator:
    """Split-operator step exp(-i H dt / hbar) for a batch of spinors (a, b) of shape (batch, n, n).

    Half step in V - iW, exact kinetic step in k space, half step in V - iW.
    The kinetic factor is cos(theta) - i sin(theta) sigma.k/|k| with theta = v |k| dt.
    """

    def __init__(self, V, W, KX, KY, dt, device, dtype=torch.complex64):
        K = np.hypot(KX, KY)
        K1 = np.where(K > 0, K, 1.0)
        theta = HBAR_V * K * dt / HBAR

        def put(f):
            return torch.as_tensor(f, device=device).to(dtype)

        self.half = put(np.exp(-(1j * V + W) * dt / (2 * HBAR)))
        self.cos = put(np.cos(theta))
        self.ab = put(-1j * np.sin(theta) * (KX - 1j * KY) / K1)
        self.ba = put(-1j * np.sin(theta) * (KX + 1j * KY) / K1)
        self.device, self.dtype = device, dtype

    def step(self, a, b):
        fa = torch.fft.fft2(self.half * a)
        fb = torch.fft.fft2(self.half * b)
        a = torch.fft.ifft2(self.cos * fa + self.ab * fb)
        b = torch.fft.ifft2(self.cos * fb + self.ba * fa)
        return self.half * a, self.half * b
