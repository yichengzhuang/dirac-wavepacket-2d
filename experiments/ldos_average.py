#!/usr/bin/env python3
"""
Multi-propagation LDOS averaging for massless 2D Dirac harmonic-ellipse billiard.

Goal (P0): obtain a 2D spatial LDOS map with correct C₂v symmetry for the
gapless (m=0) system, for comparison with STM experiments.

Strategy
--------
Launch M wave packets from many (direction, position) pairs.  For each
propagation, compute LDOS(E, r) via real-frequency Fourier transform using
spectral.compute_ldos().  Accumulate and average.

Why this works
--------------
  LDOS(E, r) ∝ Σ_n |ψ_n(r)|² Γ_n / [(E-E_n)² + (Γ_n/2)²] · |<ψ_n|ψ₀>|²

Each eigenmode density |ψ_n(r)|² has C₂v symmetry (Kramers theorem, §5 of
the paper).  Averaging over many initial states makes |<ψ_n|ψ₀>|² more
uniform across modes, so the accumulated LDOS faithfully reflects the true
spatial structure of the spectrum.

Key choices
-----------
  n=256     — reduced from 512 to keep per-propagation RAM ~400 MB
  save_every=50  — Δt_snap = 0.5, Nyquist OK for E < 6
  M=48      — 8 launch angles × 6 positions
  k_mag=3.0 — peak energy matches k_harm; σ_k~1.7 covers E~1.3–4.7
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ──
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import compute_ldos


# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Grid  (n=256 keeps per-propagation snapshot RAM manageable)
n  = 256
L  = 10.0

# Physics
vF = 1.0
m  = 0.0          # massless — gapless system

# Time evolution
dt         = 0.01
n_steps    = 10000    # T_max = 100
save_every = 50       # snapshot Δt = 0.5  →  201 snapshots / propagation
                      # Nyquist: E_max < π/Δt ≈ 6.3  ✓

# Potential: standard harmonic ellipse
POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

# CAP
cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

# Analysis mask
mask_safety_factor = 0.95
mask_margin        = 1.30

# Wavepacket
sigma0 = 0.6
k_mag  = 3.0   # peak |k|; energy E = vF*k_mag = 3, σ_E ≈ 1.7

# Energy grid for LDOS
E_min, E_max, n_E = 0.5, 5.5, 300
E_values = np.linspace(E_min, E_max, n_E)

# ── Initial conditions: 8 angles × 6 positions = 48 propagations ──
N_ang  = 8
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)

# Positions: origin + ring of 5 at r = 0.4 (well inside turning ellipse)
r_ring    = 0.4
ring_angs = np.linspace(0, 2 * np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [
    (r_ring * np.cos(a), r_ring * np.sin(a)) for a in ring_angs
]                           # 1 + 5 = 6 positions

M = N_ang * len(positions)  # 48

# ── Output ──
OUT_DIR = os.path.join(_root, "figures", "ldos_map")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def fft_flip(arr, axis):
    """FFT-consistent reflection: j → (n-j) % n along axis."""
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)


def symmetrize_c2v(ldos):
    """
    Enforce C₂v symmetry by averaging over four reflections.

    Input  : (n_E, ny, nx)
    Output : (n_E, ny, nx)  — symmetric under x→-x, y→-y
    """
    r_x  = fft_flip(ldos, axis=2)   # x → -x
    r_y  = fft_flip(ldos, axis=1)   # y → -y
    r_xy = fft_flip(r_x,  axis=1)   # (x,y) → (-x,-y)
    return 0.25 * (ldos + r_x + r_y + r_xy)


def measure_c2v_error(ldos_2d):
    """Return relative L1 asymmetry of a 2D density map."""
    r_x  = fft_flip(ldos_2d, axis=1)
    r_y  = fft_flip(ldos_2d, axis=0)
    x_err = np.sum(np.abs(r_x - ldos_2d)) / np.sum(ldos_2d)
    y_err = np.sum(np.abs(r_y - ldos_2d)) / np.sum(ldos_2d)
    return x_err, y_err


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(ldos_avg, ldos_sym, E_values, chi_cpu, L, M):
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]

    # Integrated LDOS spectrum (inside mask)
    spec_raw = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1, 2))
    spec_sym = (ldos_sym * chi_cpu[np.newaxis]).sum(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(E_values, spec_raw / spec_raw.max(), "b-",  lw=2, label="raw avg")
    ax.plot(E_values, spec_sym / spec_sym.max(), "r--", lw=1.5, label="C₂v sym")
    ax.set_xlabel("E (code units)")
    ax.set_ylabel("LDOS (integrated, a.u.)")
    ax.set_title(
        f"LDOS spectrum — massless harmonic ellipse, M={M} propagations"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ldos_spectrum.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  spectrum  → {out}")

    # Spatial maps at 6 energy peaks (symmetrized LDOS)
    peak_idx = np.argsort(spec_sym)[::-1][:6]
    peak_idx = np.sort(peak_idx)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    for j, idx in enumerate(peak_idx):
        ax  = axes[j]
        rho = ldos_sym[idx]
        im  = ax.imshow(rho, extent=extent, origin="lower",
                        cmap="inferno", vmin=0, vmax=rho.max())
        x_e, y_e = measure_c2v_error(rho)
        ax.set_title(
            f"E = {E_values[idx]:.3f}\n"
            f"x_err={x_e:.3f}  y_err={y_e:.3f}",
            fontsize=10,
        )
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"LDOS maps (C₂v symmetrized) — massless harmonic ellipse, M={M}",
        fontsize=13,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ldos_peak_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  peak maps → {out}")

    # Dense energy sweep — all E values
    ncols = 10
    nrows = (n_E + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    axes = axes.ravel()
    for j in range(n_E):
        ax  = axes[j]
        rho = ldos_sym[j]
        ax.imshow(rho, extent=extent, origin="lower",
                  cmap="inferno", vmin=0, vmax=rho.max())
        ax.set_title(f"E={E_values[j]:.2f}", fontsize=7)
        ax.axis("off")
    for j in range(n_E, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"LDOS sweep (C₂v sym) — massless harmonic ellipse, M={M}",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ldos_sweep.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  sweep     → {out}")

    # Raw vs symmetrized comparison at the strongest peak
    best = peak_idx[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, rho, title in [
        (ax1, ldos_avg[best], "Raw average"),
        (ax2, ldos_sym[best], "C₂v symmetrized"),
    ]:
        im = ax.imshow(rho, extent=extent, origin="lower",
                       cmap="inferno", vmin=0, vmax=rho.max())
        ax.set_title(f"{title}\nE = {E_values[best]:.3f}", fontsize=11)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        f"LDOS at strongest peak — massless harmonic ellipse, M={M}",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ldos_raw_vs_sym.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  raw/sym   → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    xp = get_backend("auto")
    print(f"Backend : {xp.name}")
    print(f"Grid    : {n}×{n},  L={L}")
    print(f"Runs    : {M}  ({N_ang} angles × {len(positions)} positions)")
    print(f"T_max   : {n_steps * dt:.0f},  Δt_snap = {dt * save_every:.3f}")
    print(f"E range : [{E_values[0]:.2f}, {E_values[-1]:.2f}],  {n_E} points")
    print(f"Out dir : {OUT_DIR}\n")

    # ── one-time grid / operators ──
    x_np         = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X_np, Y_np   = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    V      = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W      = radial_CAP(X_np, Y_np,
                        cap_radius=cap_radius,
                        cap_strength=cap_strength,
                        cap_power=cap_power, xp=xp)
    chi_R, mask_info = build_analysis_mask(
        POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius,
        safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp,
    )
    chi_cpu  = xp.to_numpy(chi_R)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)

    # ── accumulation loop ──
    ldos_sum     = np.zeros((n_E, n, n))
    t_wall_start = time.perf_counter()
    n_done       = 0

    for x0, y0 in positions:
        for angle in angles:
            kx0    = k_mag * np.cos(angle)
            ky0    = k_mag * np.sin(angle)
            n_done += 1
            print(f"[{n_done:2d}/{M}]  pos=({x0:+.2f},{y0:+.2f})"
                  f"  k=({kx0:+.2f},{ky0:+.2f})", flush=True)
            t0 = time.perf_counter()

            psi0_up, psi0_dn = gaussian_wavepacket(
                X, Y,
                x0=x0, y0=y0, kx0=kx0, ky0=ky0,
                sigma=sigma0, theta=None, band="conduction",
                dx=dx, xp=xp,
            )

            result = propagate(
                psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
                n_steps=n_steps, dt=dt,
                save_every=save_every,
                corr_every=save_every,   # keep same cadence; C_R unused
                max_record=n_steps // save_every + 2,
                measure_transmission=False,
            )

            snapshots  = result["snapshots"]   # (Ns, 2, n, n) numpy
            snap_times = result["snap_times"]  # (Ns,)

            ldos_one  = compute_ldos(snapshots, snap_times, E_values,
                                     chi_R=chi_cpu)
            ldos_sum += ldos_one

            # explicit free — Python GC may lag
            del snapshots, snap_times, ldos_one, result, psi0_up, psi0_dn

            elapsed = time.perf_counter() - t0
            wall    = time.perf_counter() - t_wall_start
            eta     = wall / n_done * (M - n_done)
            print(f"       {elapsed:.1f} s  |  ETA {eta / 60:.1f} min",
                  flush=True)

    # ── average & symmetrize ──
    ldos_avg = ldos_sum / M
    ldos_sym = symmetrize_c2v(ldos_avg)

    # ── save ──
    np.save(os.path.join(OUT_DIR, "ldos_avg.npy"), ldos_avg)
    np.save(os.path.join(OUT_DIR, "ldos_sym.npy"), ldos_sym)
    np.save(os.path.join(OUT_DIR, "E_values.npy"), E_values)
    print(f"\nSaved arrays → {OUT_DIR}/")

    # ── C₂v error report ──
    spec = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1, 2))
    top3 = np.argsort(spec)[::-1][:3]
    print("\nC₂v asymmetry at 3 strongest peaks (raw average):")
    print(f"  {'E':>6}  {'x_err':>8}  {'y_err':>8}")
    for idx in top3:
        x_e, y_e = measure_c2v_error(ldos_avg[idx])
        print(f"  {E_values[idx]:6.3f}  {x_e:8.4f}  {y_e:8.4f}")

    # ── figures ──
    print("\nGenerating figures …")
    make_figures(ldos_avg, ldos_sym, E_values, chi_cpu, L, M)

    total = time.perf_counter() - t_wall_start
    print(f"\nTotal wall time: {total:.1f} s  ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
