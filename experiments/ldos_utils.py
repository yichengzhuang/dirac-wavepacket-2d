"""
Shared utilities for LDOS study scripts.

Provides FFT-consistent reflection, C₂v symmetrization, error measurement,
and common propagation/LDOS loop helpers used across all study_*.py scripts.
"""

import os, sys, time
import numpy as np

# ── path setup (allow import from project root) ──
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import compute_ldos


# ══════════════════════════════════════════════════════════════════════════════
# FFT-consistent reflection and C₂v symmetry
# ══════════════════════════════════════════════════════════════════════════════

def fft_flip(arr, axis):
    """FFT-consistent reflection: j → (n-j) % n along *axis*."""
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)


def symmetrize_c2v(ldos):
    """
    Enforce C₂v symmetry by averaging over four reflections.

    Works for both 2D (ny, nx) and 3D (n_E, ny, nx) arrays.
    For 3D: x → axis=2, y → axis=1.
    For 2D: x → axis=1, y → axis=0.
    """
    if ldos.ndim == 3:
        ax_x, ax_y = 2, 1
    elif ldos.ndim == 2:
        ax_x, ax_y = 1, 0
    else:
        raise ValueError(f"Expected 2D or 3D array, got {ldos.ndim}D")
    r_x  = fft_flip(ldos, axis=ax_x)
    r_y  = fft_flip(ldos, axis=ax_y)
    r_xy = fft_flip(r_x,  axis=ax_y)
    return 0.25 * (ldos + r_x + r_y + r_xy)


def measure_c2v_error(ldos_2d):
    """
    Relative L1 asymmetry of a 2D density map under x- and y-reflection.

    Returns (x_err, y_err) where each is ||ρ(reflected) - ρ|| / ||ρ||.
    """
    r_x  = fft_flip(ldos_2d, axis=1)   # x → -x
    r_y  = fft_flip(ldos_2d, axis=0)   # y → -y
    total = np.sum(ldos_2d)
    if total == 0:
        return 0.0, 0.0
    x_err = np.sum(np.abs(r_x - ldos_2d)) / total
    y_err = np.sum(np.abs(r_y - ldos_2d)) / total
    return x_err, y_err


# ══════════════════════════════════════════════════════════════════════════════
# LDOS propagation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_ldos_sweep(
    # Grid
    n=256, L=10.0,
    # Physics
    vF=1.0, m=0.0,
    # Potential
    potential_option="harmonic_ellipse",
    pot_params=None,
    # CAP
    cap_radius=3.5, cap_strength=10.0, cap_power=2,
    # Mask
    mask_safety_factor=0.95, mask_margin=1.30,
    # Wavepacket
    sigma0=0.6, k_mag=3.0,
    # Launch conditions
    angles=None, positions=None,
    # Time
    dt=0.01, n_steps=10000, save_every=50,
    # Energy grid
    E_values=None,
    # Backend
    backend="auto",
    # Callbacks
    on_propagation_done=None,
    verbose=True,
):
    """
    Run multiple propagations and accumulate LDOS.

    Parameters
    ----------
    on_propagation_done : callable(j, M, ldos_one, ldos_sum) or None
        Called after each propagation with index j, total M,
        single-propagation LDOS, and running sum.
        Can be used to record per-propagation metrics.

    Returns
    -------
    dict with keys:
        'ldos_avg'  : (n_E, ny, nx) averaged LDOS
        'ldos_sum'  : (n_E, ny, nx) unnormalized sum
        'E_values'  : (n_E,)
        'chi_cpu'   : (ny, nx) analysis mask (numpy)
        'M'         : total number of propagations
        'grid'      : dict(X, Y, KX, KY, dx, dk)
        'elapsed'   : total wall time in seconds
    """
    if pot_params is None:
        pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    if positions is None:
        r_ring = 0.4
        ring_angs = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        positions = [(0.0, 0.0)] + [
            (r_ring * np.cos(a), r_ring * np.sin(a)) for a in ring_angs
        ]
    if E_values is None:
        E_values = np.linspace(0.5, 5.5, 300)

    M = len(angles) * len(positions)
    xp = get_backend(backend)

    # Build grid — need both CPU numpy and backend arrays
    x_np = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    # Build potential/CAP/mask on CPU numpy grids, transfer to backend
    V = build_potential(potential_option, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius, cap_strength, cap_power, xp)
    chi_R, _mask_info = build_analysis_mask(
        potential_option, X_np, Y_np, pot_params,
        cap_radius, mask_safety_factor, mask_margin, xp,
    )
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)
    sqrt_chi = xp.sqrt(chi_R)

    chi_cpu = xp.to_numpy(chi_R).real.astype(np.float64)
    n_E = len(E_values)
    ldos_sum = np.zeros((n_E, n, n))

    t0 = time.time()
    j = 0
    for x0, y0 in positions:
        for angle in angles:
            kx0 = k_mag * np.cos(angle)
            ky0 = k_mag * np.sin(angle)

            psi0_up, psi0_dn = gaussian_wavepacket(
                X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
                sigma=sigma0, theta=None, band="conduction",
                dx=dx, xp=xp,
            )

            result = propagate(
                psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
                n_steps=n_steps, dt=dt,
                save_every=save_every,
                corr_every=save_every,
                max_record=n_steps // save_every + 2,
                measure_transmission=False,
            )

            snapshots  = result["snapshots"]    # (Ns, 2, n, n) numpy
            snap_times = result["snap_times"]   # (Ns,)

            ldos_one = compute_ldos(snapshots, snap_times, E_values,
                                    chi_R=chi_cpu)
            ldos_sum += ldos_one
            j += 1

            if on_propagation_done is not None:
                on_propagation_done(j, M, ldos_one, ldos_sum)

            del snapshots, snap_times, ldos_one, result, psi0_up, psi0_dn

            if verbose:
                elapsed = time.time() - t0
                print(f"  [{j}/{M}] angle={np.degrees(angle):.0f}° "
                      f"pos=({x0:.2f},{y0:.2f})  "
                      f"{elapsed:.0f}s elapsed", flush=True)

    elapsed = time.time() - t0
    ldos_avg = ldos_sum / M

    return dict(
        ldos_avg=ldos_avg,
        ldos_sum=ldos_sum,
        E_values=E_values,
        chi_cpu=chi_cpu,
        M=M,
        dx=dx,
        elapsed=elapsed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Spectrum analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

def integrated_spectrum(ldos, chi):
    """Integrate LDOS over space (inside mask) → 1D spectrum."""
    return (ldos * chi[np.newaxis]).sum(axis=(1, 2))


def find_peaks_simple(spectrum, E_values, min_prominence=0.1):
    """
    Find peaks in a 1D spectrum. Returns (peak_indices, peak_energies).

    Uses scipy.signal.find_peaks if available, otherwise simple local-max.
    """
    try:
        from scipy.signal import find_peaks
        idx, props = find_peaks(spectrum / spectrum.max(),
                                prominence=min_prominence)
    except ImportError:
        # Fallback: simple local maximum detection
        idx = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                if spectrum[i] > min_prominence * spectrum.max():
                    idx.append(i)
        idx = np.array(idx)
    return idx, E_values[idx]


def estimate_fwhm(spectrum, E_values, peak_idx):
    """Estimate FWHM of a peak by half-maximum interpolation."""
    half_max = spectrum[peak_idx] / 2.0
    # Search left
    left = peak_idx
    while left > 0 and spectrum[left] > half_max:
        left -= 1
    # Search right
    right = peak_idx
    while right < len(spectrum) - 1 and spectrum[right] > half_max:
        right += 1
    return E_values[right] - E_values[left]


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path
