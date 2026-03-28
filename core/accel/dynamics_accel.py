"""
Accelerated dynamics module with backend abstraction.

Mirrors dynamics.py but:
  - All np.xxx → xp.xxx (backend parameter)
  - Spinor carried as (psi_up, psi_dn) to avoid stack/empty_like per step
  - Potentials/masks computed on CPU then transferred via xp.array()
"""

import sys
import os
import time
import numpy as np

# Allow importing potentials_2d from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from potentials_2d import (
    hard_wall_circle, hard_wall_ellipse,
    soft_wall_circle, soft_wall_ellipse, soft_wall_stadium,
    harmonic_circle, harmonic_ellipse,
    circular_mask, elliptical_mask, stadium_mask,
)


# ═══════════════════════════════════════════════
# Grid  (computed on backend)
# ═══════════════════════════════════════════════

def make_grid(n, L, xp):
    """
    Square grid [-L/2, L/2)^2 with FFT-conjugate k-grid.

    Returns (X, Y, KX, KY, K, dx) as backend arrays.
    """
    x = xp.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = float(L / n)
    X, Y = xp.meshgrid(x, x, indexing="xy")
    k = xp.fftfreq(n, d=dx)
    KX, KY = xp.meshgrid(k, k, indexing="xy")
    K = xp.sqrt(KX**2 + KY**2)
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
    sigma=0.5, theta=None, band="conduction", dx=1.0, xp=None, valley=1,
):
    """
    Two-component Gaussian wavepacket.  Returns (psi_up, psi_dn).
    """
    env = xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2.0 * sigma**2))
    phase = xp.exp(1j * (kx0 * X + ky0 * Y))

    if theta is None:
        chi = chiral_spinor(kx0, ky0, band=band, valley=valley)
    else:
        chi = np.array([1.0, np.exp(1j * theta)], dtype=complex) / np.sqrt(2.0)

    psi_up = env * phase * chi[0]
    psi_dn = env * phase * chi[1]
    norm2 = float(xp.sum(xp.abs(psi_up)**2 + xp.abs(psi_dn)**2).real) * dx**2
    s = 1.0 / np.sqrt(norm2 + 1e-300)
    return psi_up * s, psi_dn * s


# ═══════════════════════════════════════════════
# Potential & mask builders  (CPU → device)
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


def build_potential(option, X_np, Y_np, params, xp):
    """Evaluate potential on CPU NumPy grids, transfer to backend."""
    key = option.lower().strip()
    if key not in _POTENTIAL_MAP:
        raise ValueError(
            f"Unknown potential '{option}'. "
            f"Choose from: {list(_POTENTIAL_MAP.keys())}"
        )
    V_np = _POTENTIAL_MAP[key](X_np, Y_np, **params)
    return xp.array(V_np)


def radial_CAP(X_np, Y_np, cap_radius, cap_strength=10.0, cap_power=2, xp=None):
    """Build CAP on CPU, transfer to backend."""
    R = np.sqrt(X_np**2 + Y_np**2)
    Rmax = np.max(R)
    width = max(Rmax - cap_radius, 1e-8)
    s = np.clip((R - cap_radius) / width, 0.0, None)
    W_np = cap_strength * s**cap_power
    return xp.array(W_np)


def _clamp_to_cap(*semi, safety_factor, cap_radius, use_sum=False):
    """Scale semi-axes so that their max (or sum) stays within the CAP."""
    extent = sum(semi) if use_sum else max(semi)
    if extent > safety_factor * cap_radius:
        s = (safety_factor * cap_radius) / extent
        return tuple(v * s for v in semi)
    return semi


def build_analysis_mask(
    option, X_np, Y_np, geometry_params, cap_radius,
    safety_factor=0.95, mask_margin=2.0, xp=None,
):
    """Auto-build analysis mask on CPU, transfer to backend."""
    opt = option.lower().strip()

    if opt in ("hard_circle", "soft_circle", "harmonic_circle"):
        R0 = float(geometry_params.get("radius", geometry_params.get("r0")))
        (Rmask,) = _clamp_to_cap(mask_margin * R0,
                                  safety_factor=safety_factor, cap_radius=cap_radius)
        chi_np = circular_mask(X_np, Y_np, radius=Rmask)
        return xp.array(chi_np), {"mask_type": "circle", "radius": round(Rmask, 4)}

    if opt in ("hard_ellipse", "soft_ellipse", "harmonic_ellipse"):
        a, b = float(geometry_params["a"]), float(geometry_params["b"])
        aR, bR = _clamp_to_cap(mask_margin * a, mask_margin * b,
                                safety_factor=safety_factor, cap_radius=cap_radius)
        chi_np = elliptical_mask(X_np, Y_np, aR=aR, bR=bR)
        return xp.array(chi_np), {"mask_type": "ellipse", "aR": round(aR, 4), "bR": round(bR, 4)}

    if opt == "soft_stadium":
        a_half, b = float(geometry_params["a_half"]), float(geometry_params["b"])
        aR, bR = _clamp_to_cap(mask_margin * a_half, mask_margin * b,
                                safety_factor=safety_factor, cap_radius=cap_radius,
                                use_sum=True)
        chi_np = stadium_mask(X_np, Y_np, a_half=aR, b=bR)
        return xp.array(chi_np), {"mask_type": "stadium", "a_half": round(aR, 4), "b": round(bR, 4)}

    raise ValueError(f"Unknown option '{option}' for mask building.")


# ═══════════════════════════════════════════════
# Split-operator propagator (2D Dirac, ħ = 1)
# ═══════════════════════════════════════════════

def build_split_operator(vF, dt, m, V, W, KX, KY, K, xp, valley=1):
    """
    Build split-operator closures.

    Returns (half_potential, kinetic) that operate on (psi_up, psi_dn).
    All coefficient arrays live on the backend device.
    """
    phase_VW = xp.exp(-1j * V * dt / 2.0) * xp.exp(-W * dt / 2.0)
    mass_up = np.exp(-1j * m * dt / 2.0)  # scalar
    mass_dn = np.exp(+1j * m * dt / 2.0)  # scalar

    def half_potential(up, dn):
        return phase_VW * mass_up * up, phase_VW * mass_dn * dn

    theta_k = vF * K * dt
    cos_th = xp.cos(theta_k)
    sin_th = xp.sin(theta_k)
    K_safe = xp.where(K > 0, K, xp.array(1.0))
    off_upper = (valley * KX - 1j * KY) / K_safe
    off_lower = (valley * KX + 1j * KY) / K_safe

    def kinetic(up, dn):
        up_k = xp.fftn(up)
        dn_k = xp.fftn(dn)
        up_new = cos_th * up_k - 1j * sin_th * (off_upper * dn_k)
        dn_new = cos_th * dn_k - 1j * sin_th * (off_lower * up_k)
        return xp.ifftn(up_new), xp.ifftn(dn_new)

    return half_potential, kinetic


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════

def density(up, dn, xp):
    """Total probability density |ψ_A|² + |ψ_B|²."""
    return xp.abs(up)**2 + xp.abs(dn)**2


def overlap_R(a_up, a_dn, b_up, b_dn, sqrt_chi, dx, xp):
    """<a|χ_R|b> with dx² for continuous-consistent inner product.

    Always computed in float64 (CPU) for spectral analysis stability.
    """
    au = xp.to_numpy(a_up * sqrt_chi).astype(np.complex128).ravel()
    bu = xp.to_numpy(b_up * sqrt_chi).astype(np.complex128).ravel()
    ad = xp.to_numpy(a_dn * sqrt_chi).astype(np.complex128).ravel()
    bd = xp.to_numpy(b_dn * sqrt_chi).astype(np.complex128).ravel()
    val = np.vdot(au, bu) + np.vdot(ad, bd)
    return complex(val) * dx**2


# ═══════════════════════════════════════════════
# Main propagation driver
# ═══════════════════════════════════════════════

def propagate(
    psi0_up, psi0_dn, halfV, kinetic, sqrt_chi, chi_R, dx, xp,
    n_steps, dt,
    save_every=10,
    corr_every=10,
    max_record=2500,
    live_callback=None,
    live_every=20,
    anim_writer=None,
    anim_callback=None,
    anim_every=100,
    measure_transmission=True,
    transmission_every=10,
    auto_stop_T=None,
):
    """
    Split-operator propagation loop (accelerated).

    Spinor is carried as separate (psi_up, psi_dn) arrays on the backend
    device to avoid stack/empty_like allocation overhead per step.

    Parameters
    ----------
    auto_stop_T : float or None
        If set, stop propagation early when transmission T >= this value.
        T = 1 - P(t)/P(0).  Use 1.0 for "full escape" (practically ~0.99).

    Callbacks receive CPU NumPy arrays in (2, ny, nx) format.
    """
    psi_up = xp.copy(psi0_up)
    psi_dn = xp.copy(psi0_dn)

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

    t_start = time.perf_counter()

    for k in range(n_steps + 1):
        t = k * dt

        # ── decimated correlation + snapshots ──
        if recording and k % corr_every == 0:
            C_R_list.append(
                overlap_R(psi0_up, psi0_dn, psi_up, psi_dn, sqrt_chi, dx, xp)
            )
            corr_t_list.append(t)

            if k % save_every == 0:
                snap_np = np.stack([
                    xp.to_numpy(psi_up), xp.to_numpy(psi_dn)
                ], axis=0)
                snap_list.append(snap_np)
                snap_t_list.append(t)

            record_count += 1
            if max_record is not None and record_count >= max_record:
                recording = False

        # ── transmission ──
        if measure_transmission and k % transmission_every == 0:
            rho = density(psi_up, psi_dn, xp)
            P_val = float(xp.sum(rho * chi_R).real) * dx**2
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

        # ── real-time display (CPU arrays) ──
        if live_callback is not None and k % live_every == 0:
            psi_cpu = np.stack([
                xp.to_numpy(psi_up), xp.to_numpy(psi_dn)
            ], axis=0)
            live_callback(psi_cpu, t, k)

        # ── file animation frame (CPU arrays) ──
        if (anim_writer is not None and anim_callback is not None
                and k % anim_every == 0):
            psi_cpu = np.stack([
                xp.to_numpy(psi_up), xp.to_numpy(psi_dn)
            ], axis=0)
            anim_callback(psi_cpu, t, k)
            anim_writer.grab_frame()

        # ── propagate ──
        if k < n_steps:
            psi_up, psi_dn = halfV(psi_up, psi_dn)
            psi_up, psi_dn = kinetic(psi_up, psi_dn)
            psi_up, psi_dn = halfV(psi_up, psi_dn)

    elapsed = time.perf_counter() - t_start
    steps_done = actual_steps if stopped_early else n_steps
    print(f"    Propagation wall time: {elapsed:.2f} s  "
          f"({elapsed/max(steps_done,1)*1e3:.2f} ms/step, "
          f"{steps_done}/{n_steps} steps)")

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
