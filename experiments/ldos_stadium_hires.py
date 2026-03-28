#!/usr/bin/env python3
"""
High-resolution stadium scar LDOS.

Improvements over ldos_stadium_scar.py:
  - k_mag: 8 values (2,4,6,8,10,13,16,20) for seamless energy coverage
  - n=512 for better spatial resolution at high E
  - E range [0.5, 30], n_E=600
  - 6 launch positions (center + ring)
  - save_every=10 (Δt=0.1, Nyquist ~31)
  - Total: 8 × 6 × 8 = 384 propagations

Expected time: GPU ~2h, CPU ~8h
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

OUT = os.path.join(_root, "figures", "ldos_stadium_scar")
os.makedirs(OUT, exist_ok=True)

# ── Grid ──
n = 512; L = 10.0; vF = 1.0; m_mass = 0.0
dt = 0.01; n_steps = 10000; save_every = 10

# ── Stadium potential ──
POTENTIAL = "soft_stadium"
pot_params = dict(a_half=1.0, b=1.0, V0=70.0, lam=25.0, mu=5.0)
cap_radius = 3.0; cap_strength = 10.0; cap_power = 2

# ── Energy grid ──
E_values = np.linspace(0.5, 30.0, 600)

# ── Dense k_mag coverage to eliminate spectral gaps ──
K_MAGS = [2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0]
sigma0 = 0.6

# ── Launch: 8 angles × 6 positions ──
N_ang = 8
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
r_ring = 0.3
ring_angs = np.linspace(0, 2 * np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [(r_ring*np.cos(a), r_ring*np.sin(a)) for a in ring_angs]

def fft_flip(a, ax):
    return np.roll(np.flip(a, axis=ax), 1, axis=ax)

def sym_c2v(r):
    rx = fft_flip(r, axis=1)
    ry = fft_flip(r, axis=0)
    return 0.25 * (r + rx + ry + fft_flip(rx, 0))


def main():
    t_total = time.perf_counter()
    xp = get_backend("auto")
    n_total = len(K_MAGS) * len(positions) * N_ang
    print(f"Backend: {xp.name}")
    print(f"Grid: {n}×{n}, L={L}")
    print(f"k_mag: {K_MAGS} ({len(K_MAGS)} values)")
    print(f"Positions: {len(positions)}, Angles: {N_ang}")
    print(f"Total propagations: {n_total}")
    print(f"E range: [{E_values[0]}, {E_values[-1]}], n_E={len(E_values)}")
    print(f"Δt_snap={dt*save_every}, Nyquist E_max={np.pi/(dt*save_every):.1f}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    V = build_potential(POTENTIAL, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(POTENTIAL, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m_mass, V, W, KX, KY, K, xp)

    ldos_sum = np.zeros((len(E_values), n, n))
    n_done = 0

    for k_mag in K_MAGS:
        print(f"\n  k_mag = {k_mag} (E_peak ≈ {k_mag:.0f})")
        for x0, y0 in positions:
            for angle in angles:
                kx0 = k_mag * np.cos(angle)
                ky0 = k_mag * np.sin(angle)
                n_done += 1

                psi0_up, psi0_dn = gaussian_wavepacket(
                    X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
                    sigma=sigma0, theta=None, band="conduction",
                    dx=dx, xp=xp)

                result = propagate(
                    psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
                    n_steps=n_steps, dt=dt, save_every=save_every,
                    corr_every=save_every,
                    max_record=n_steps // save_every + 2,
                    measure_transmission=False)

                ldos_one = compute_ldos(
                    result["snapshots"], result["snap_times"],
                    E_values, chi_R=chi_cpu)
                ldos_sum += ldos_one
                del result, ldos_one, psi0_up, psi0_dn

                if n_done % 24 == 0:
                    elapsed = time.perf_counter() - t_total
                    eta = elapsed / n_done * (n_total - n_done)
                    print(f"    [{n_done}/{n_total}] {elapsed:.0f}s "
                          f"ETA {eta/60:.1f} min", flush=True)

    ldos_avg = ldos_sum / n_total

    np.save(os.path.join(OUT, "ldos_hires.npy"), ldos_avg)
    np.save(os.path.join(OUT, "E_values_hires.npy"), E_values)
    print(f"\nSaved LDOS → {OUT}/")

    # ── Figures ──
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]
    x_arr = np.linspace(-half_L, half_L, n, endpoint=False)

    # 1. Gallery 10×10
    n_show = 100
    idx = np.linspace(0, len(E_values) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(10, 10, figsize=(35, 35))
    for k, i in enumerate(idx):
        ax = axes[k // 10, k % 10]
        rho = sym_c2v(ldos_avg[i])
        ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=rho.max())
        ax.set_title(f"E={E_values[i]:.1f}", fontsize=6)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Stadium LDOS gallery E∈[0.5, 30]\n"
                 "n=512, k_mag=[2,4,6,8,10,13,16,20], M=384",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "gallery_hires.png"), dpi=120,
                bbox_inches="tight")
    plt.close()

    # 2. Spectrum
    spec = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(E_values, spec / spec.max(), "b-", lw=1)
    ax.set_xlabel("E"); ax.set_ylabel("LDOS (norm.)")
    ax.set_title("Stadium LDOS spectrum (high-res, n=512)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "spectrum_hires.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # 3. 1D line cuts
    ldos_sym = np.array([sym_c2v(ldos_avg[j]) for j in range(len(E_values))])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.pcolormesh(x_arr, E_values, ldos_sym[:, n//2, :],
                   cmap="inferno", shading="auto")
    ax1.set_xlabel("x"); ax1.set_ylabel("E")
    ax1.set_title("LDOS(E, x) at y=0"); ax1.set_xlim(-2.5, 2.5)
    ax2.pcolormesh(x_arr, E_values, ldos_sym[:, :, n//2],
                   cmap="inferno", shading="auto")
    ax2.set_xlabel("y"); ax2.set_ylabel("E")
    ax2.set_title("LDOS(E, y) at x=0"); ax2.set_xlim(-2, 2)
    fig.suptitle("Stadium 1D LDOS (high-res, n=512)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "linecuts_hires.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # 4. Scar candidates: top 16 peaks at E > 5
    from scipy.signal import find_peaks
    spec_high = spec.copy(); spec_high[E_values < 5] = 0
    peaks_idx, _ = find_peaks(spec_high, distance=3)
    if len(peaks_idx) > 0:
        peaks_sorted = peaks_idx[np.argsort(spec[peaks_idx])[::-1]]
        n_scar = min(16, len(peaks_sorted))
        ncols = 4
        nrows = (n_scar + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows))
        axes = np.array(axes).ravel()
        for j in range(n_scar):
            ax = axes[j]
            rho = sym_c2v(ldos_avg[peaks_sorted[j]])
            im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                          vmin=0, vmax=rho.max())
            ax.set_title(f"E={E_values[peaks_sorted[j]]:.2f}", fontsize=10)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.8)
        for j in range(n_scar, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Scar candidates (E > 5, 16 strongest peaks, n=512)",
                     fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "scar_hires.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

    # 5. Selected energies large panels (bouncing ball region)
    E_select = [3, 5, 8, 10, 13, 15, 18, 20, 23, 25]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.ravel()
    for j, E_target in enumerate(E_select):
        idx_e = np.argmin(np.abs(E_values - E_target))
        ax = axes[j]
        rho = sym_c2v(ldos_avg[idx_e])
        im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                      vmin=0, vmax=rho.max())
        ax.set_title(f"E={E_values[idx_e]:.1f}", fontsize=11)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Stadium LDOS at selected energies (n=512, high-res)",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "selected_energies_hires.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
