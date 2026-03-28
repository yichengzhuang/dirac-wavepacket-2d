#!/usr/bin/env python3
"""
Study: how does launch position affect the LDOS?

Fix potential (harmonic ellipse), mass (m=0), σ=0.6, k_mag=3.0, 8 directions.
Vary only the set of launch positions.

Groups:
  center     — (0,0) only           → M=8
  x_axis     — 4 pts along x        → M=32
  y_axis     — 4 pts along y        → M=32
  ring_small — r=0.4 ring (5 pts)   → M=40
  ring_large — r=0.8 ring (5 pts)   → M=40
  uniform    — all above            → M=120 (but we cap at M=48 for time)
"""
import sys, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import compute_ldos

OUT = os.path.join(_root, "figures", "ldos_position")
os.makedirs(OUT, exist_ok=True)

n=256; L=10.0; vF=1.0; m_mass=0.0
dt=0.01; n_steps=10000; save_every=50
POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
sigma0=0.6; k_mag=3.0

E_min, E_max, n_E = 0.5, 5.5, 100  # coarser for speed
E_values = np.linspace(E_min, E_max, n_E)

N_ang = 8
angles = np.linspace(0, 2*np.pi, N_ang, endpoint=False)

ring5 = lambda r: [(r*np.cos(a), r*np.sin(a))
                   for a in np.linspace(0, 2*np.pi, 5, endpoint=False)]

GROUPS = [
    ("center",     [(0.0, 0.0)]),
    ("x_axis",     [(0,0), (0.5,0), (1.0,0), (1.5,0)]),
    ("y_axis",     [(0,0), (0,0.3), (0,0.6), (0,0.9)]),
    ("ring_0.4",   ring5(0.4)),
    ("ring_0.8",   ring5(0.8)),
    ("mixed",      [(0,0)] + ring5(0.4) + [(0.5,0), (1.0,0), (0,0.3), (0,0.6)]),
]


def fft_flip(a, ax): return np.roll(np.flip(a, axis=ax), 1, axis=ax)
def sym_c2v(r):
    return 0.25*(r + fft_flip(r,1) + fft_flip(r,0) + fft_flip(fft_flip(r,1),0))
def c2v_err(rho):
    s = np.sum(rho)+1e-30
    return np.sum(np.abs(fft_flip(rho,1)-rho))/s, np.sum(np.abs(fft_flip(rho,0)-rho))/s


def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m_mass, V, W, KX, KY, K, xp)

    all_results = []

    for group_name, pos_list in GROUPS:
        M = N_ang * len(pos_list)
        print(f"\n{'='*50}")
        print(f"  {group_name}: {len(pos_list)} positions × {N_ang} angles = M={M}")
        print(f"{'='*50}")

        ldos_sum = np.zeros((n_E, n, n))
        t_start = time.perf_counter()

        for x0, y0 in pos_list:
            for angle in angles:
                kx0 = k_mag * np.cos(angle)
                ky0 = k_mag * np.sin(angle)
                psi0_up, psi0_dn = gaussian_wavepacket(
                    X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
                    sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
                result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
                    n_steps=n_steps, dt=dt, save_every=save_every,
                    corr_every=save_every, max_record=n_steps//save_every+2,
                    measure_transmission=False)
                ldos_one = compute_ldos(result["snapshots"], result["snap_times"],
                                         E_values, chi_R=chi_cpu)
                ldos_sum += ldos_one
                del result, ldos_one, psi0_up, psi0_dn

        ldos_avg = ldos_sum / M
        elapsed = time.perf_counter() - t_start
        print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # C₂v check
        spec = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1,2))
        best_idx = np.argmax(spec)
        xe, ye = c2v_err(ldos_avg[best_idx])
        print(f"  Peak E={E_values[best_idx]:.2f}: x_err={xe:.4f} y_err={ye:.4f}")

        all_results.append((group_name, ldos_avg, M))

    # ── Figures ──
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]

    # Pick a common energy near the strongest peak
    spec_ref = (all_results[0][1] * chi_cpu[np.newaxis]).sum(axis=(1,2))
    E_show_idx = np.argmax(spec_ref)
    E_show = E_values[E_show_idx]

    ng = len(all_results)
    ncols = min(ng, 3)
    nrows = (ng + ncols - 1) // ncols

    # Raw LDOS at one energy
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for k, (name, ldos, M_k) in enumerate(all_results):
        ax = axes[k // ncols, k % ncols]
        rho = ldos[E_show_idx]
        xe, ye = c2v_err(rho)
        ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=rho.max())
        ax.set_title(f"{name} (M={M_k})\nx={xe:.3f} y={ye:.3f}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(ng, nrows*ncols):
        axes[k//ncols, k%ncols].axis("off")
    fig.suptitle(f"LDOS at E={E_show:.2f} — raw, different launch positions", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "position_comparison_raw.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # C₂v symmetrized version
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for k, (name, ldos, M_k) in enumerate(all_results):
        ax = axes[k // ncols, k % ncols]
        rho = sym_c2v(ldos[E_show_idx])
        ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=rho.max())
        ax.set_title(f"{name} (M={M_k})", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(ng, nrows*ncols):
        axes[k//ncols, k%ncols].axis("off")
    fig.suptitle(f"LDOS at E={E_show:.2f} — C₂v symmetrized", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "position_comparison_sym.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Spectrum comparison
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, ldos, M_k in all_results:
        spec = (ldos * chi_cpu[np.newaxis]).sum(axis=(1,2))
        ax.plot(E_values, spec/spec.max(), lw=1.5, label=f"{name} (M={M_k})")
    ax.set_xlabel("E"); ax.set_ylabel("LDOS (norm.)")
    ax.set_title("LDOS spectrum vs launch position")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "spectrum_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # C₂v error bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [r[0] for r in all_results]
    x_errs = []
    y_errs = []
    for name, ldos, M_k in all_results:
        spec = (ldos * chi_cpu[np.newaxis]).sum(axis=(1,2))
        idx = np.argmax(spec)
        xe, ye = c2v_err(ldos[idx])
        x_errs.append(xe)
        y_errs.append(ye)
    x_pos = np.arange(len(names))
    ax.bar(x_pos - 0.15, x_errs, 0.3, label="x_err", color="C0")
    ax.bar(x_pos + 0.15, y_errs, 0.3, label="y_err", color="C1")
    ax.set_xticks(x_pos); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("C₂v error"); ax.set_title("Symmetry error vs position group (raw)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "c2v_error_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFigures → {OUT}/")
    print(f"Total: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
