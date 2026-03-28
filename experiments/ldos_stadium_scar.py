#!/usr/bin/env python3
"""
Stadium LDOS targeting scar states at high energy.

Key physics:
- Bunimovich stadium: classically chaotic, scarred by unstable periodic orbits
- Bouncing ball orbits (vertical in rectangular part): E_n ≈ nπvF/(2b)
- Diamond/diagonal orbits: higher E, more complex pattern
- Need E ~ 10-30 to see clear scars (many wavelengths inside stadium)

Strategy:
- Multiple k_mag values to cover broad energy range
- save_every=10 (Δt=0.1) for Nyquist E_max~31
- E range [0.5, 25], 500 points
- V₀=70, λ=25, μ=5 (reference paper parameters)
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
n = 256; L = 10.0; vF = 1.0; m_mass = 0.0
dt = 0.01; n_steps = 10000; save_every = 10   # Δt=0.1, Nyquist~31

# ── Stadium potential (reference paper) ──
POTENTIAL = "soft_stadium"
pot_params = dict(a_half=1.0, b=1.0, V0=70.0, lam=25.0, mu=5.0)
cap_radius = 3.0; cap_strength = 10.0; cap_power = 2

# ── Energy grid ──
E_values = np.linspace(0.5, 25.0, 500)

# ── Multiple k_mag to cover full spectrum ──
# k_mag=3 excites E~1-5, k_mag=7 excites E~5-10, k_mag=12 excites E~10-15,
# k_mag=18 excites E~15-22
K_MAGS = [3.0, 7.0, 12.0, 18.0]
sigma0 = 0.6

# ── Launch: 8 angles × 4 positions per k_mag ──
N_ang = 8
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
positions = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (-0.3, 0.0)]

def fft_flip(a, ax):
    return np.roll(np.flip(a, axis=ax), 1, axis=ax)

def sym_c2v(r):
    rx = fft_flip(r, axis=1)
    ry = fft_flip(r, axis=0)
    return 0.25 * (r + rx + ry + fft_flip(rx, 0))


def main():
    t_total = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")
    print(f"Stadium: a_half={pot_params['a_half']}, b={pot_params['b']}, V0={pot_params['V0']}")
    print(f"E range: [{E_values[0]}, {E_values[-1]}], n_E={len(E_values)}")
    print(f"k_mag values: {K_MAGS}")
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
    n_total = len(K_MAGS) * len(positions) * N_ang
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

                if n_done % 16 == 0:
                    elapsed = time.perf_counter() - t_total
                    eta = elapsed / n_done * (n_total - n_done)
                    print(f"    [{n_done}/{n_total}] {elapsed:.0f}s "
                          f"ETA {eta/60:.1f} min", flush=True)

    ldos_avg = ldos_sum / n_total

    np.save(os.path.join(OUT, "ldos_stadium_scar.npy"), ldos_avg)
    np.save(os.path.join(OUT, "E_values.npy"), E_values)
    print(f"\nSaved LDOS → {OUT}/")

    # ── Figures ──
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]

    # 1. Big gallery: 10×10 = 100 panels
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
    fig.suptitle("Stadium LDOS gallery E∈[0.5, 25]\n"
                 "soft wall V₀=70, λ=25, μ=5, massless Dirac",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "gallery_100.png"), dpi=120,
                bbox_inches="tight")
    plt.close()

    # 2. Spectrum
    spec = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(E_values, spec / spec.max(), "b-", lw=1)
    ax.set_xlabel("E"); ax.set_ylabel("LDOS (norm.)")
    ax.set_title("Stadium LDOS spectrum E∈[0.5, 25]")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "spectrum.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # 3. 1D line cuts
    x_arr = np.linspace(-half_L, half_L, n, endpoint=False)
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
    fig.suptitle("Stadium 1D LDOS (E∈[0.5, 25])", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "linecuts.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # 4. Scar candidate gallery: top 12 peaks at E > 8
    high_E_mask = E_values > 8.0
    spec_high = spec.copy()
    spec_high[~high_E_mask] = 0
    from scipy.signal import find_peaks
    peaks_idx, _ = find_peaks(spec_high, distance=3)
    if len(peaks_idx) > 0:
        peaks_sorted = peaks_idx[np.argsort(spec[peaks_idx])[::-1]]
        n_scar = min(12, len(peaks_sorted))
        fig, axes = plt.subplots(2, min(6, n_scar),
                                 figsize=(4 * min(6, n_scar), 8))
        if n_scar <= 6:
            axes = axes.reshape(2, -1) if n_scar > 1 else np.array([[axes[0]], [axes[1]]])
        for j in range(n_scar):
            ax = axes[j // 6, j % 6] if n_scar > 6 else axes[0, j] if j < 6 else axes[1, j-6]
            idx_p = peaks_sorted[j]
            rho = sym_c2v(ldos_avg[idx_p])
            ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                      vmin=0, vmax=rho.max())
            ax.set_title(f"E={E_values[idx_p]:.2f}", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Scar candidates (E > 8, strongest peaks)", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "scar_candidates.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
