#!/usr/bin/env python3
"""
LDOS for three billiard shapes: harmonic ellipse, harmonic circle, soft-wall stadium.
All matched to similar area (~7.5 code units²).
M=48 propagations each, n=256, n_E=300.
Outputs 2D LDOS maps + 1D line cuts for each shape.
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

OUT = os.path.join(_root, "figures", "ldos_shapes")
os.makedirs(OUT, exist_ok=True)

# ── Common parameters ──
n=256; L=10.0; vF=1.0; m_mass=0.0
dt=0.01; n_steps=10000; save_every=50   # T=100, Δt=0.5
sigma0=0.6; k_mag=3.0
cap_strength=10.0; cap_power=2

E_min, E_max, n_E = 0.5, 5.5, 300
E_values = np.linspace(E_min, E_max, n_E)

# ── Initial conditions: 8 angles × 6 positions = 48 ──
N_ang = 8
angles = np.linspace(0, 2*np.pi, N_ang, endpoint=False)
r_ring = 0.4
ring_angs = np.linspace(0, 2*np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [(r_ring*np.cos(a), r_ring*np.sin(a)) for a in ring_angs]
M = N_ang * len(positions)

# ── Shape configurations ──
# Target area: π*a*b = π*2*1.2 ≈ 7.54 for ellipse
# Circle: π*r² = 7.54 → r = 1.55
# Stadium: 2*a_half*2*b + π*b² = 7.54; with b=1.0, a_half=1.0 → 4+π=7.14 ≈ close

SHAPES = [
    {
        "name": "harmonic_ellipse",
        "potential": "harmonic_ellipse",
        "pot_params": dict(k_harm=3.0, a=2.0, b=1.2),
        "cap_radius": 3.5,
        "area_label": "πab=7.54",
    },
    {
        "name": "harmonic_circle",
        "potential": "harmonic_circle",
        "pot_params": dict(k_harm=3.0, r0=1.55),
        "cap_radius": 3.0,  # r0=1.55, need cap > 1.55; use 3.0
        "area_label": "πr²=7.55",
    },
    {
        "name": "soft_stadium",
        "potential": "soft_stadium",
        # Stadium: a_half=1.0, b=1.0 gives area ≈ 7.14
        # V0=70, lam=25, mu=5 gives smooth walls similar to soft_ellipse
        "pot_params": dict(a_half=1.0, b=1.0, V0=70.0, lam=25.0, mu=5.0),
        "cap_radius": 3.0,
        "area_label": "A≈7.14",
    },
]


def fft_flip(arr, axis):
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)

def sym_c2v(rho):
    rx = fft_flip(rho, axis=1)
    ry = fft_flip(rho, axis=0)
    return 0.25*(rho + rx + ry + fft_flip(rx, 0))

def c2v_err(rho):
    s = np.sum(rho) + 1e-30
    return (np.sum(np.abs(fft_flip(rho,1)-rho))/s,
            np.sum(np.abs(fft_flip(rho,0)-rho))/s)


def run_ldos_for_shape(shape, xp, X, Y, KX, KY, K, dx, X_np, Y_np):
    """Run M=48 LDOS for one shape. Returns ldos_avg (n_E, n, n)."""
    name = shape["name"]
    pot = shape["potential"]
    params = shape["pot_params"]
    cr = shape["cap_radius"]

    print(f"\n{'='*60}")
    print(f"  Shape: {name}  ({shape['area_label']})")
    print(f"{'='*60}")

    V = build_potential(pot, X_np, Y_np, params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cr,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, mask_info = build_analysis_mask(pot, X_np, Y_np, params,
        cap_radius=cr, safety_factor=0.95, mask_margin=1.30, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m_mass, V, W, KX, KY, K, xp)

    ldos_sum = np.zeros((n_E, n, n))
    t0 = time.perf_counter()
    n_done = 0

    for x0, y0 in positions:
        for angle in angles:
            kx0 = k_mag * np.cos(angle)
            ky0 = k_mag * np.sin(angle)
            n_done += 1

            psi0_up, psi0_dn = gaussian_wavepacket(
                X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
                sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)

            result = propagate(
                psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
                n_steps=n_steps, dt=dt, save_every=save_every,
                corr_every=save_every,
                max_record=n_steps//save_every+2,
                measure_transmission=False)

            ldos_one = compute_ldos(result["snapshots"], result["snap_times"],
                                     E_values, chi_R=chi_cpu)
            ldos_sum += ldos_one
            del result, ldos_one, psi0_up, psi0_dn

            if n_done % 8 == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / n_done * (M - n_done)
                print(f"  [{n_done}/{M}]  {elapsed:.0f}s  ETA {eta/60:.1f} min",
                      flush=True)

    ldos_avg = ldos_sum / M
    elapsed = time.perf_counter() - t0
    print(f"  {name} done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # C₂v check on raw average
    spec = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1,2))
    top3 = np.argsort(spec)[::-1][:3]
    print(f"  C₂v errors at top-3 peaks (raw average):")
    for idx in top3:
        xe, ye = c2v_err(ldos_avg[idx])
        print(f"    E={E_values[idx]:.3f}  x_err={xe:.4f}  y_err={ye:.4f}")

    return ldos_avg, chi_cpu


def make_plots(all_results):
    """Make comparison figures for all shapes."""
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]
    x = np.linspace(-half_L, half_L, n, endpoint=False)
    n_shapes = len(all_results)

    # ── 1. Integrated LDOS spectrum comparison ──
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, ldos_avg, chi_cpu in all_results:
        spec = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1,2))
        ax.plot(E_values, spec / spec.max(), lw=2, label=name)
    ax.set_xlabel("E"); ax.set_ylabel("LDOS (integrated, normalized)")
    ax.set_title("LDOS spectrum comparison (M=48, massless)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "spectrum_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Peak energy maps for each shape ──
    fig, axes = plt.subplots(n_shapes, 4, figsize=(16, 4*n_shapes))
    for i, (name, ldos_avg, chi_cpu) in enumerate(all_results):
        ldos_sym = np.array([sym_c2v(ldos_avg[j]) for j in range(n_E)])
        spec = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1,2))
        peaks = np.argsort(spec)[::-1][:4]
        peaks = np.sort(peaks)

        for j, idx in enumerate(peaks):
            ax = axes[i, j]
            rho = ldos_sym[idx]
            ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                      vmin=0, vmax=rho.max())
            ax.set_title(f"E={E_values[idx]:.2f}", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(name, fontsize=10)

    fig.suptitle("LDOS at peak energies (C₂v sym, M=48, massless)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "peak_maps_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. 1D line cuts for each shape ──
    fig, axes = plt.subplots(n_shapes, 2, figsize=(14, 5*n_shapes))
    for i, (name, ldos_avg, chi_cpu) in enumerate(all_results):
        ldos_sym = np.array([sym_c2v(ldos_avg[j]) for j in range(n_E)])
        ldos_Ex = ldos_sym[:, n//2, :]   # y=0 cut
        ldos_Ey = ldos_sym[:, :, n//2]   # x=0 cut

        ax = axes[i, 0]
        ax.pcolormesh(x, E_values, ldos_Ex, cmap="inferno", shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("E")
        ax.set_title(f"{name}: LDOS(E,x) at y=0")
        ax.set_xlim(-3, 3)

        ax = axes[i, 1]
        ax.pcolormesh(x, E_values, ldos_Ey, cmap="inferno", shading="auto")
        ax.set_xlabel("y"); ax.set_ylabel("E")
        ax.set_title(f"{name}: LDOS(E,y) at x=0")
        ax.set_xlim(-2, 2)

    fig.suptitle("1D LDOS line cuts (n_E=300, M=48, C₂v sym)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "linecuts_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures saved to {OUT}/")


def main():
    t_total = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")
    print(f"Grid: {n}×{n}, L={L}, T={n_steps*dt:.0f}, M={M}")
    print(f"E range: [{E_min}, {E_max}], n_E={n_E}")
    print(f"Shapes: {[s['name'] for s in SHAPES]}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    all_results = []
    for shape in SHAPES:
        ldos_avg, chi_cpu = run_ldos_for_shape(
            shape, xp, X, Y, KX, KY, K, dx, X_np, Y_np)

        # Save per-shape
        tag = shape["name"]
        np.save(os.path.join(OUT, f"ldos_{tag}.npy"), ldos_avg)
        all_results.append((tag, ldos_avg, chi_cpu))

    np.save(os.path.join(OUT, "E_values.npy"), E_values)

    make_plots(all_results)

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
