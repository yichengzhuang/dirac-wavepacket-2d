"""Resonances of an open system from the wavefunctions of a few launches.

Once the direct, non-resonant part has left, the wavefunction of a launch psi_s is a sum of
decaying resonances with complex energies z_n = E_n - i Gamma_n / 2,

    psi_s(r, t) = sum_n  P_n psi_s(r) exp(-i z_n t / hbar),

where P_n projects onto resonance n.  Since H^T = sigma_y H sigma_y, every resonance is a
doublet and its left eigenvectors are sigma_y times its right ones (transposed, not conjugated).
Two launches with independent parts in each doublet therefore fix P_n.  Its local trace
rho_n(r) = Tr <r|P_n|r> is the share of resonance n in the local density of states,
-Im[rho_n(r) / (E - z_n)] / pi.

    band      keep one energy window and sample it coarsely in time
    poles     the z_n shared by all channels, from one matrix pencil
    weights   P_n psi_s(r) by least squares with the poles held fixed
    density   rho_n(r) from the weights of the launches
"""

import numpy as np
from scipy import signal

from dirac import HBAR


def band(t, f, E_lo, E_hi, transition=60.0, attenuation=100.0):
    """Keep E_lo < E < E_hi (meV) in f(t, ...) and sample it as coarsely as the band allows.

    A Kaiser-window FIR filter removes all other energies, by `attenuation` dB beyond
    `transition` meV from the band edges; energies inside the transition are only damped.
    The first and last half-length of the filter are dropped.
    An exponential exp(-i z t / hbar) keeps its z; only its weight is multiplied by gain(z).
    Returns the new times, the filtered signal and gain.
    """
    dt = t[1] - t[0]
    centre = (E_lo + E_hi) / 2
    half = (E_hi - E_lo) / 2
    nyquist = np.pi * HBAR / dt

    taps, beta = signal.kaiserord(attenuation, transition / nyquist)
    taps |= 1
    lowpass = signal.firwin(taps, half + transition / 2, window=("kaiser", beta), fs=2 * nyquist)
    tau = (np.arange(taps) - taps // 2) * dt
    h = lowpass * np.exp(-1j * centre * tau / HBAR)

    def gain(z):
        return h @ np.exp(1j * np.outer(tau, z) / HBAR)

    shape = (taps,) + (1,) * (f.ndim - 1)
    ff = signal.fftconvolve(f, h.reshape(shape), mode="same", axes=0)

    step = int(nyquist / (half + transition))
    keep = np.arange(taps // 2, len(t) - taps // 2, step)
    return t[keep], ff[keep], gain


def poles(t, f, E_lo, E_hi, tol=1e-5, depth=1):
    """Complex energies z_n = E_n - i Gamma_n / 2 with E_lo < E_n < E_hi, shared by all channels.

    One time step multiplies the part of f(t, ...) that belongs to resonance n by
    exp(-i z_n dt / hbar).  The poles are the eigenvalues of this step, taken in the
    singular directions of f above tol times the largest; tol should lie near the leakage
    of the filter, 10^(-attenuation / 20).  With fewer channels than resonances, stack
    `depth` successive samples of every channel into each column.
    """
    dt = t[1] - t[0]
    f = f.reshape(len(t), -1)
    cols = len(t) - depth + 1
    M = np.concatenate([f[k:k + cols] for k in range(depth)], axis=1).T

    U, s, Vh = np.linalg.svd(M[:, :-1], full_matrices=False)
    r = np.sum(s > tol * s[0])
    step = U[:, :r].conj().T @ M[:, 1:] @ Vh[:r].conj().T / s[:r]
    z = 1j * HBAR * np.log(np.linalg.eigvals(step)) / dt

    # energies are known modulo 2 pi hbar / dt: take the copy nearest the window
    period = 2 * np.pi * HBAR / dt
    z += period * np.round(((E_lo + E_hi) / 2 - z.real) / period)

    inside = (z.real > E_lo) & (z.real < E_hi) & (z.imag < 0)
    return np.sort_complex(z[inside])


def weights(t, f, z, gain):
    """Least-squares weights of exp(-i z_n t / hbar) in the filtered f(t, ...), one per channel."""
    A = np.exp(-1j * np.outer(t, z) / HBAR)
    a = np.linalg.lstsq(A, f.reshape(len(t), -1), rcond=None)[0]
    a = a / gain(z)[:, None]
    return a.reshape((len(z),) + f.shape[1:])


def density(F, psi0, dx):
    """rho_n(r) = Tr <r|P_n|r> of one doublet from the weights F_s = P_n psi0_s of the launches.

    F and psi0 have shape (launch, sublattice, y, x), and the launches lie inside the region.
    With R an orthonormal basis of the F_s, F_s = R B_s, and left vectors l = sigma_y R,
    the overlaps S = l^T R follow from l^T psi0_s = S B_s; then P_n = R S^-1 l^T.
    """
    m = len(F)
    M = F.reshape(m, -1).T
    R = np.linalg.svd(M, full_matrices=False)[0][:, :2]
    B = R.conj().T @ M

    R = R.T.reshape((2,) + F.shape[1:])
    L = np.stack([-1j * R[:, 1], 1j * R[:, 0]], axis=1)
    G = np.einsum("jayx,sayx->js", L, psi0) * dx**2
    S = G @ np.linalg.pinv(B)
    return np.einsum("ij,jayx,iayx->yx", np.linalg.inv(S), L, R)
