#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral analysis module: matrix-pencil harmonic inversion, resonance
extraction, complex-frequency time-filtering, and initial-state energy
envelope from Dirac band projection.
"""

import numpy as np


# ═══════════════════════════════════════════════
# Matrix-pencil harmonic inversion
# ═══════════════════════════════════════════════

def matrix_pencil(C, L):
    """
    Matrix-pencil (Hankel) method.

    Build L×L Hankel blocks Y0, Y1 from signal C[0..K] (requires 2L ≤ K),
    solve the generalised eigenvalue problem  Y1 v = z Y0 v.

    Parameters
    ----------
    C : 1-D complex array, length K+1
    L : int, pencil size

    Returns
    -------
    z : 1-D complex array, eigenvalues
    W : 2-D complex array, eigenvectors
    """
    C = np.asarray(C)
    K = len(C) - 1
    assert 2 * L <= K, f"matrix_pencil: need 2L <= K  (L={L}, K={K})"

    Y0 = np.empty((L, L), dtype=complex)
    Y1 = np.empty((L, L), dtype=complex)
    for i in range(L):
        for j in range(L):
            Y0[i, j] = C[i + j]
            Y1[i, j] = C[i + j + 1]

    # SVD-regularised pseudo-inverse of Y0
    U, S, Vh = np.linalg.svd(Y0, full_matrices=False)
    tol = 1e-10 * S[0]
    r = int(np.sum(S > tol))
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
    Y0_inv = Vh_r.conj().T @ np.diag(1.0 / S_r) @ U_r.conj().T

    z, W = np.linalg.eig(Y0_inv @ Y1)
    return z, W


def wrap_to_nyquist(omega, dt):
    """Wrap Re(ω) into (-π/dt, π/dt] to resolve log-branch ambiguity."""
    ny = np.pi / dt
    om_re = np.real(omega)
    om_re_wrapped = ((om_re + ny) % (2 * ny)) - ny
    return om_re_wrapped + 1j * np.imag(omega)


# ═══════════════════════════════════════════════
# Full resonance extraction pipeline
# ═══════════════════════════════════════════════

def extract_resonances(C_R, corr_times, dt_corr,
                       Gamma_cut=8.0, pencil_fraction=3, L_pencil_max=400):
    """
    Run the complete harmonic-inversion pipeline on C_R(t).

    Parameters
    ----------
    C_R : 1-D complex array
        Regional autocorrelation (decimated).
    corr_times : 1-D array
        Time values corresponding to C_R.
    dt_corr : float
        Sampling interval of C_R  (= dt * corr_every).
    Gamma_cut : float
        Discard poles with Γ > Gamma_cut.
    pencil_fraction : int
        L_pencil = len(C_R) // pencil_fraction.
    L_pencil_max : int
        Hard cap on pencil size to keep SVD tractable.

    Returns
    -------
    dict with keys: omega, E, Gamma, a
    """
    Ksig = len(C_R) - 1
    L_pencil = min(L_pencil_max, max(50, Ksig // pencil_fraction))
    L_pencil = min(L_pencil, Ksig // 2)

    z_raw, _ = matrix_pencil(C_R, L=L_pencil)

    # z → ω  (use dt_corr, the actual sampling interval!)
    omega_raw = (1j / dt_corr) * np.log(z_raw)
    omega_raw = wrap_to_nyquist(omega_raw, dt_corr)

    # keep decaying poles  |z| < 1
    ok = np.abs(z_raw) < 1.0
    omega = omega_raw[ok]

    E = np.real(omega)
    Gamma = -2.0 * np.imag(omega)

    # physical width filter
    ok2 = (Gamma > 0) & (Gamma < Gamma_cut)
    omega = omega[ok2]
    E = E[ok2]
    Gamma = Gamma[ok2]

    if len(omega) == 0:
        return {"omega": omega, "E": E, "Gamma": Gamma, "a": np.array([])}

    # least-squares amplitude fit
    Vls = np.vstack([np.exp(-1j * om * corr_times) for om in omega]).T
    a, *_ = np.linalg.lstsq(Vls, C_R, rcond=None)

    # sort by |a| descending
    order = np.argsort(-np.abs(a))
    omega = omega[order]
    E     = E[order]
    Gamma = Gamma[order]
    a     = a[order]

    return {"omega": omega, "E": E, "Gamma": Gamma, "a": a}


# ═══════════════════════════════════════════════
# Spinor normalization helper
# ═══════════════════════════════════════════════

def _normalize(phi):
    norm = np.sqrt(np.sum(np.abs(phi[0])**2 + np.abs(phi[1])**2))
    return phi / norm if norm > 0 else phi


# ═══════════════════════════════════════════════
# Complex-frequency time-filtering (mode reconstruction)
# ═══════════════════════════════════════════════

def time_filter_mode(snapshots, snap_times, omega_target, chi_R, window="hann"):
    """
    Reconstruct a quasi-eigenstate by Fourier-filtering stored snapshots.

    φ(r) ∝ Σ_k  w(t_k) e^{+iω t_k} χ_R(r) ψ(r, t_k)

    Parameters
    ----------
    snapshots : ndarray (Ns, 2, ny, nx)
    snap_times : 1-D array (Ns,)
    omega_target : complex
    chi_R : ndarray (ny, nx)  — analysis mask
    window : str  ("hann" or "rect")

    Returns
    -------
    phi : ndarray (2, ny, nx), normalised
    """
    Ns = len(snap_times)
    if window == "hann" and Ns > 1:
        w = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(Ns) / (Ns - 1)))
    else:
        w = np.ones(Ns)

    acc_up = np.zeros_like(snapshots[0, 0], dtype=complex)
    acc_dn = np.zeros_like(snapshots[0, 1], dtype=complex)

    for k in range(Ns):
        phase = np.exp(1j * omega_target * snap_times[k])
        acc_up += w[k] * phase * (snapshots[k, 0] * chi_R)
        acc_dn += w[k] * phase * (snapshots[k, 1] * chi_R)

    return _normalize(np.stack([acc_up, acc_dn], axis=0))


def filter_modes(snapshots, snap_times, omega_arr, chi_R, n_modes=3, window="hann"):
    """Filter the first *n_modes* strongest modes."""
    n = min(n_modes, len(omega_arr))
    return [
        time_filter_mode(snapshots, snap_times, omega_arr[j], chi_R, window=window)
        for j in range(n)
    ]


def compute_ldos(snapshots, snap_times, E_values, chi_R=None):
    """
    Compute the local density of states (LDOS) at real energies.

    LDOS(E, r) ∝ |Σ_k e^{iEt_k} ψ_A(r, t_k)|² + |Σ_k e^{iEt_k} ψ_B(r, t_k)|²

    This is a real-frequency Fourier transform — no matrix pencil, no complex
    frequencies, no mode separation.  Overlapping Lorentzians naturally sum.
    Each contributing mode's density has C₂v symmetry, so the LDOS inherits it.

    Parameters
    ----------
    snapshots : ndarray (Ns, 2, ny, nx)
    snap_times : 1-D array (Ns,)
    E_values : 1-D array of real energies at which to evaluate
    chi_R : ndarray (ny, nx) or None — optional analysis mask

    Returns
    -------
    ldos : ndarray (n_E, ny, nx)
    """
    Ns = len(snap_times)
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    n_E = len(E_values)
    N_sp = ny * nx

    # Vectorised: replace inner Python loop with matrix multiplication.
    # phases: (n_E, Ns)  —  Fourier kernel e^{iE_j t_k}
    phases = np.exp(1j * np.outer(E_values, snap_times))  # (n_E, Ns)

    # Flatten spatial dims and apply mask once
    snap_flat = snapshots.reshape(Ns, 2, N_sp)             # (Ns, 2, N_sp)
    if chi_R is not None:
        chi_flat = chi_R.ravel()
        snap_A = snap_flat[:, 0, :] * chi_flat[None, :]    # (Ns, N_sp)
        snap_B = snap_flat[:, 1, :] * chi_flat[None, :]
    else:
        snap_A = snap_flat[:, 0, :]
        snap_B = snap_flat[:, 1, :]

    # Matrix multiply: (n_E, Ns) @ (Ns, N_sp) → (n_E, N_sp)
    ft_A = phases @ snap_A
    ft_B = phases @ snap_B

    ldos = (np.abs(ft_A)**2 + np.abs(ft_B)**2).reshape(n_E, ny, nx)
    return ldos


def reconstruct_modes_lstsq(snapshots, snap_times, omega_arr, chi_R, n_modes=3):
    """
    Reconstruct spatial modes by multi-frequency least-squares decomposition.

    Instead of filtering one frequency at a time (which leaks nearby modes),
    simultaneously deconvolve all modes via:

        ψ(r, t_k) ≈ Σ_n φ_n(r) · exp(-i ω_n t_k)

    Solved as a Vandermonde least-squares problem per spatial point.
    Correctly handles damping (Im ω_n) — broad modes contribute mostly to
    early snapshots, narrow modes contribute throughout.

    Parameters
    ----------
    snapshots : ndarray (Ns, 2, ny, nx)
    snap_times : 1-D array (Ns,)
    omega_arr : 1-D complex array (from extract_resonances, sorted by |a|)
    chi_R : ndarray (ny, nx) — analysis mask
    n_modes : int

    Returns
    -------
    list of ndarray (2, ny, nx), normalised
    """
    n = min(n_modes, len(omega_arr))
    Ns = len(snap_times)
    ny, nx = snapshots.shape[2], snapshots.shape[3]

    # Truncate to time range where signal is significant.
    # Signal decays as exp(-Γ t/2); use snapshots up to t where
    # the strongest mode has decayed to ~1% of its initial value.
    Gamma_min = -2.0 * np.min(np.imag(omega_arr[:n]))
    if Gamma_min > 0:
        t_cut = -2.0 * np.log(0.01) / Gamma_min  # exp(-Γ/2 · t_cut) = 0.01
        mask_t = snap_times <= t_cut
        if np.sum(mask_t) < 2 * n:
            mask_t = np.ones(Ns, dtype=bool)  # fallback: use all
    else:
        mask_t = np.ones(Ns, dtype=bool)

    t_use = snap_times[mask_t]
    snap_use = snapshots[mask_t]
    Ns_use = len(t_use)

    # Vandermonde matrix: V[k, j] = exp(-i ω_j t_k)
    V = np.column_stack([np.exp(-1j * omega_arr[j] * t_use)
                         for j in range(n)])  # (Ns_use, n)

    modes = []
    for j in range(n):
        modes.append(np.zeros((2, ny, nx), dtype=complex))

    # Solve per spinor component: V @ coeffs = data
    for alpha in range(2):
        data = np.empty((Ns_use, ny * nx), dtype=complex)
        for k in range(Ns_use):
            data[k, :] = (snap_use[k, alpha] * chi_R).ravel()

        coeffs, *_ = np.linalg.lstsq(V, data, rcond=None)

        for j in range(n):
            modes[j][alpha] = coeffs[j].reshape(ny, nx)

    for j in range(n):
        modes[j] = _normalize(modes[j])

    return modes


# ═══════════════════════════════════════════════
# Symmetry-adapted superpositions from ±k runs
# ═══════════════════════════════════════════════

def _align_phase(phi_ref, phi):
    """Rotate global phase of *phi* so that <phi_ref|phi> is real & positive."""
    overlap = np.vdot(phi_ref, phi)
    if overlap == 0:
        return phi
    return np.exp(-1j * np.angle(overlap)) * phi


def symmetrise_modes(modes_plus, modes_minus):
    """
    Build even/odd superpositions from paired ±k filtered modes.

    Returns
    -------
    list of (phi_even, phi_odd) tuples
    """
    n_match = min(len(modes_plus), len(modes_minus))
    pairs = []
    for j in range(n_match):
        p = _normalize(modes_plus[j])
        m = _normalize(modes_minus[j])
        m = _align_phase(p, m)
        pairs.append((_normalize(p + m), _normalize(p - m)))
    return pairs


# ═══════════════════════════════════════════════
# Initial-state energy envelope (Dirac band projection)
# ═══════════════════════════════════════════════

def initial_energy_envelope(psi0, KX, KY, vF, m, nbins=400, Emax=None):
    """
    Energy distribution of the initial state from k-space band projection.

    Returns
    -------
    centers : 1-D array  (negative + positive energies)
    envelope : 1-D array (normalised to peak 1)
    """
    up_k = np.fft.fftn(psi0[0])
    dn_k = np.fft.fftn(psi0[1])

    dkx = vF * KX
    dky = vF * KY
    dz  = m * np.ones_like(KX)
    Ek  = np.sqrt(dkx**2 + dky**2 + dz**2)

    Ek_safe = np.where(Ek > 0, Ek, 1.0)
    nx_ = dkx / Ek_safe
    ny_ = dky / Ek_safe
    nz_ = dz  / Ek_safe

    a, b = up_k, dn_k

    # band projector for +E branch
    P11 = 0.5 * (1.0 + nz_)
    P22 = 0.5 * (1.0 - nz_)
    P12 = 0.5 * (nx_ - 1j * ny_)
    P21 = 0.5 * (nx_ + 1j * ny_)

    w_plus = (np.conj(a) * P11 * a +
              np.conj(a) * P12 * b +
              np.conj(b) * P21 * a +
              np.conj(b) * P22 * b).real

    w_tot = (np.abs(a)**2 + np.abs(b)**2).real
    w_minus = np.clip(w_tot - w_plus, 0.0, None)

    Ek_flat = Ek.flatten()
    w_plus_flat  = w_plus.flatten()
    w_minus_flat = w_minus.flatten()

    if Emax is None:
        Emax = float(np.percentile(Ek_flat, 99.5))

    bins = np.linspace(0.0, Emax, nbins + 1)
    centers_pos = 0.5 * (bins[:-1] + bins[1:])

    env_plus,  _ = np.histogram(Ek_flat, bins=bins, weights=w_plus_flat)
    env_minus, _ = np.histogram(Ek_flat, bins=bins, weights=w_minus_flat)

    centers  = np.concatenate([-centers_pos[::-1], centers_pos])
    envelope = np.concatenate([env_minus[::-1], env_plus])

    peak = envelope.max()
    if peak > 0:
        envelope = envelope / peak

    return centers, envelope