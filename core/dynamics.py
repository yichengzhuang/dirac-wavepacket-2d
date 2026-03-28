#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamics module: grid construction, spinor initialisation, CAP,
split-operator propagator, and the main propagation loop.
"""

import numpy as np

from potentials_2d import (
    hard_wall_circle, hard_wall_ellipse,
    soft_wall_circle, soft_wall_ellipse, soft_wall_stadium,
    harmonic_circle, harmonic_ellipse,
    circular_mask, elliptical_mask, stadium_mask,
)


# ═══════════════════════════════════════════════
# Grid
# ═══════════════════════════════════════════════

def make_grid(n, L):
    """
    Square grid [-L/2, L/2)^2 with FFT-conjugate k-grid.

    Returns
    -------
    X, Y, KX, KY, K, dx
    """
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="xy")
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    K = np.sqrt(KX**2 + KY**2)
    return X, Y, KX, KY, K, dx


# ═══════════════════════════════════════════════
# Spinor / wavepacket construction
# ═══════════════════════════════════════════════

def chiral_spinor(kx0, ky0, band="conduction", valley=1):
    """Chiral spinor for K (valley=+1) or K' (valley=-1) valley."""
    phi = np.arctan2(ky0, valley * kx0)
    if band == "conduction":
        chi = np.array([1.0, np.exp(1j * phi)], dtype=complex)
    elif band == "valence":
        chi = np.array([1.0, -np.exp(1j * phi)], dtype=complex)
    else:
        raise ValueError("band must be 'conduction' or 'valence'")
    return chi / np.sqrt(2.0)


def gaussian_wavepacket(
    X, Y, x0=0.0, y0=0.0, kx0=0.0, ky0=0.0,
    sigma=0.5, theta=None, band="conduction", dx=1.0, valley=1,
):
    """
    Two-component Gaussian wavepacket.

    theta=None → chiral spinor aligned with (kx0,ky0).
    theta=float → fixed relative phase (1, e^{iθ})/√2.
    Normalization uses dx² for continuous-consistent ∫|ψ|²d²r = 1.
    """
    env = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2.0 * sigma**2))
    phase = np.exp(1j * (kx0 * X + ky0 * Y))

    if theta is None:
        chi = chiral_spinor(kx0, ky0, band=band, valley=valley)
    else:
        chi = np.array([1.0, np.exp(1j * theta)], dtype=complex) / np.sqrt(2.0)

    psi = np.stack([env * phase * chi[0], env * phase * chi[1]], axis=0)
    norm2 = np.sum(np.abs(psi[0])**2 + np.abs(psi[1])**2) * dx**2
    return psi / np.sqrt(norm2 + 1e-300)


# ═══════════════════════════════════════════════
# Potential & mask builders
# ═══════════════════════════════════════════════

_POTENTIAL_MAP = {
    "hard_circle":      hard_wall_circle,
    "hard_ellipse":     hard_wall_ellipse,
    "soft_circle":      soft_wall_circle,
    "soft_ellipse":     soft_wall_ellipse,
    "soft_stadium":     soft_wall_stadium,
    "harmonic_circle":  harmonic_circle,
    "harmonic_ellipse": harmonic_ellipse,
}


def build_potential(option, X, Y, params):
    key = option.lower().strip()
    if key not in _POTENTIAL_MAP:
        raise ValueError(
            f"Unknown potential '{option}'. "
            f"Choose from: {list(_POTENTIAL_MAP.keys())}"
        )
    return _POTENTIAL_MAP[key](X, Y, **params)


def radial_CAP(X, Y, cap_radius, cap_strength=10.0, cap_power=2):
    R = np.sqrt(X**2 + Y**2)
    Rmax = np.max(R)
    width = max(Rmax - cap_radius, 1e-8)
    s = np.clip((R - cap_radius) / width, 0.0, None)
    return cap_strength * s**cap_power


def _clamp_to_cap(*semi, safety_factor, cap_radius, use_sum=False):
    """Scale semi-axes so that their max (or sum) stays within the CAP."""
    extent = sum(semi) if use_sum else max(semi)
    if extent > safety_factor * cap_radius:
        s = (safety_factor * cap_radius) / extent
        return tuple(v * s for v in semi)
    return semi


def build_analysis_mask(
    option, X, Y, geometry_params, cap_radius,
    safety_factor=0.95, mask_margin=2.0,
):
    """Auto-build an analysis mask matched to the potential geometry."""
    opt = option.lower().strip()

    if opt in ("hard_circle", "soft_circle", "harmonic_circle"):
        R0 = float(geometry_params.get("radius", geometry_params.get("r0")))
        (Rmask,) = _clamp_to_cap(mask_margin * R0,
                                  safety_factor=safety_factor, cap_radius=cap_radius)
        return (circular_mask(X, Y, radius=Rmask),
                {"mask_type": "circle", "radius": round(Rmask, 4)})

    if opt in ("hard_ellipse", "soft_ellipse", "harmonic_ellipse"):
        a, b = float(geometry_params["a"]), float(geometry_params["b"])
        aR, bR = _clamp_to_cap(mask_margin * a, mask_margin * b,
                                safety_factor=safety_factor, cap_radius=cap_radius)
        return (elliptical_mask(X, Y, aR=aR, bR=bR),
                {"mask_type": "ellipse", "aR": round(aR, 4), "bR": round(bR, 4)})

    if opt == "soft_stadium":
        a_half, b = float(geometry_params["a_half"]), float(geometry_params["b"])
        aR, bR = _clamp_to_cap(mask_margin * a_half, mask_margin * b,
                                safety_factor=safety_factor, cap_radius=cap_radius,
                                use_sum=True)
        return (stadium_mask(X, Y, a_half=aR, b=bR),
                {"mask_type": "stadium", "a_half": round(aR, 4), "b": round(bR, 4)})

    raise ValueError(f"Unknown option '{option}' for mask building.")


# ═══════════════════════════════════════════════
# Split-operator propagator (2D Dirac, ħ = 1)
# ═══════════════════════════════════════════════

def build_split_operator(vF, dt, m, V, W, KX, KY, K, valley=1):
    phase_VW = np.exp(-1j * V * dt / 2.0) * np.exp(-W * dt / 2.0)
    mass_up = np.exp(-1j * m * dt / 2.0)
    mass_dn = np.exp(+1j * m * dt / 2.0)

    def half_potential(psi):
        out = np.empty_like(psi)
        out[0] = phase_VW * mass_up * psi[0]
        out[1] = phase_VW * mass_dn * psi[1]
        return out

    theta_k = vF * K * dt
    cos_th = np.cos(theta_k)
    sin_th = np.sin(theta_k)
    with np.errstate(divide="ignore", invalid="ignore"):
        K_safe = np.where(K > 0, K, 1.0)
        off_upper = (valley * KX - 1j * KY) / K_safe
        off_lower = (valley * KX + 1j * KY) / K_safe

    def kinetic(psi):
        up_k = np.fft.fftn(psi[0])
        dn_k = np.fft.fftn(psi[1])
        up_new = cos_th * up_k - 1j * sin_th * (off_upper * dn_k)
        dn_new = cos_th * dn_k - 1j * sin_th * (off_lower * up_k)
        return np.stack([np.fft.ifftn(up_new),
                         np.fft.ifftn(dn_new)], axis=0)

    return half_potential, kinetic


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════

def density(psi):
    return np.abs(psi[0])**2 + np.abs(psi[1])**2


def overlap_R(a, b, sqrt_chi, dx=1.0):
    """<a|χ_R|b> with dx² for continuous-consistent inner product."""
    val = (np.vdot(a[0] * sqrt_chi, b[0] * sqrt_chi)
           + np.vdot(a[1] * sqrt_chi, b[1] * sqrt_chi))
    return val * dx**2


# ═══════════════════════════════════════════════
# Main propagation driver
# ═══════════════════════════════════════════════

def propagate(
    psi0, halfV, kinetic, sqrt_chi, dx,
    n_steps, dt,
    save_every=10,
    corr_every=10,
    max_record=2500,
    # ── real-time display (during propagation) ──
    live_callback=None,
    live_every=20,
    # ── file animation (streaming, memory-efficient) ──
    anim_writer=None,
    anim_callback=None,
    anim_every=100,
    # ── transmission measurement ──
    measure_transmission=True,
    transmission_every=10,
    # ── auto-stop ──
    auto_stop_T=None,
):
    """
    Split-operator propagation loop.

    Parameters
    ----------
    save_every : int
        Save full wavefunction snapshot every N steps (for mode filtering
        and post-propagation animation).
    corr_every : int
        Record C_R(t) every N steps.  Decimating (e.g. 10) makes the
        matrix-pencil tractable: L_pencil ∝ n_steps/corr_every instead
        of n_steps.  The returned dt_corr = dt * corr_every.
    max_record : int or None
        Stop recording after this many correlation/snapshot samples
        (caps memory).  Propagation continues to completion.
    live_callback, live_every : real-time display.
    anim_writer, anim_callback, anim_every : file animation.
    measure_transmission, transmission_every : survival probability.
    auto_stop_T : float or None
        Stop early when T = 1 - P(t)/P(0) >= this value. None to disable.

    Returns
    -------
    dict with keys:
        C_R, corr_times, dt_corr,
        snapshots, snap_times,
        P_inside, P_times  (if measure_transmission)
    """
    chi_R = sqrt_chi**2
    psi = psi0.copy()

    C_R_list = []
    corr_t_list = []
    snap_list = []
    snap_t_list = []
    P_list = []
    P_t_list = []

    record_count = 0
    recording = True
    P0 = None
    stopped_early = False
    actual_steps = n_steps

    for k in range(n_steps + 1):
        t = k * dt

        # ── decimated correlation + snapshots ──
        if recording and k % corr_every == 0:
            C_R_list.append(overlap_R(psi0, psi, sqrt_chi, dx))
            corr_t_list.append(t)

            if k % save_every == 0:
                snap_list.append(psi.copy())
                snap_t_list.append(t)

            record_count += 1
            if max_record is not None and record_count >= max_record:
                recording = False

        # ── transmission ──
        if measure_transmission and k % transmission_every == 0:
            P_val = float(np.sum(density(psi) * chi_R).real) * dx**2
            P_list.append(P_val)
            P_t_list.append(t)

            # ── auto-stop check ──
            if auto_stop_T is not None:
                if P0 is None:
                    P0 = P_val
                if P0 > 0:
                    T_now = 1.0 - P_val / P0
                    if T_now >= auto_stop_T:
                        print(f"    Auto-stop: T={T_now:.4f} >= {auto_stop_T} "
                              f"at t={t:.4f} (step {k}/{n_steps})")
                        actual_steps = k
                        stopped_early = True
                        break

        # ── real-time display ──
        if live_callback is not None and k % live_every == 0:
            live_callback(psi, t, k)

        # ── file animation frame ──
        if (anim_writer is not None and anim_callback is not None
                and k % anim_every == 0):
            anim_callback(psi, t, k)
            anim_writer.grab_frame()

        # ── propagate ──
        if k < n_steps:
            psi = halfV(psi)
            psi = kinetic(psi)
            psi = halfV(psi)

    steps_done = actual_steps if stopped_early else n_steps
    result = {
        "C_R":        np.asarray(C_R_list),
        "corr_times": np.asarray(corr_t_list),
        "dt_corr":    dt * corr_every,
        "snapshots":  np.asarray(snap_list),
        "snap_times": np.asarray(snap_t_list),
        "stopped_early": stopped_early,
        "steps_done":    steps_done,
    }
    if measure_transmission:
        result["P_inside"] = np.asarray(P_list)
        result["P_times"]  = np.asarray(P_t_list)
    return result