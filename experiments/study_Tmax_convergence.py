#!/usr/bin/env python3
"""
Study 1.2: T_max convergence of LDOS spectral resolution.

Theory prediction:
    FWHM ≈ max(Γ_phys, 2π/T_max)
    Plateau at T_max > T_crit = 2π/Γ ≈ 37 (for Γ=0.17)

Method:
    Run M=48 propagations with T_max=200 (n_steps=20000, save_every=50).
    Compute LDOS using subsets of snapshots → effective T_max = 10,20,...,200.
    Measure FWHM of strongest peak at each T_max.

Output: figures/study_Tmax/
"""

import sys, os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "scripts"))
sys.path.insert(0, _root)

from ldos_utils import (
    fft_flip, symmetrize_c2v, measure_c2v_error,
    integrated_spectrum, estimate_fwhm, ensure_dir,
)
from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import compute_ldos

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

n  = 256
L  = 10.0
vF = 1.0
m  = 0.0

dt         = 0.01
n_steps    = 20000   # T_max = 200
save_every = 50      # Δt = 0.5 → 401 snapshots

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

sigma0 = 0.6
k_mag  = 3.0

# Standard M=48
N_ang  = 8
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
r_ring    = 0.4
ring_angs = np.linspace(0, 2 * np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [
    (r_ring * np.cos(a), r_ring * np.sin(a)) for a in ring_angs
]

E_values = np.linspace(0.5, 5.5, 500)  # finer grid for FWHM measurement

# T_max values to test
T_MAX_VALUES = [10, 20, 30, 40, 60, 80, 100, 150, 200]

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_Tmax"))


def main():
    M = N_ang * len(positions)
    print(f"Study 1.2: T_max convergence, M={M}, T_max_final=200")

    xp = get_backend("auto")
    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius, cap_strength, cap_power, xp)
    chi_R, _ = build_analysis_mask(POTENTIAL_OPTION, X_np, Y_np, pot_params,
                                    cap_radius, 0.95, 1.30, xp)
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)
    sqrt_chi = xp.sqrt(chi_R)
    chi_cpu = xp.to_numpy(chi_R).real.astype(np.float64)

    n_E = len(E_values)
    ldos_sums = {t: np.zeros((n_E, n, n)) for t in T_MAX_VALUES}

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
                save_every=save_every, corr_every=save_every,
                max_record=n_steps // save_every + 2,
                measure_transmission=False,
            )
            snaps_np = result["snapshots"]
            times_np = result["snap_times"]
            j += 1

            # For each T_max, use only snapshots up to that time
            for t_max in T_MAX_VALUES:
                mask = times_np <= t_max + 1e-10
                snaps_sub = snaps_np[mask]
                times_sub = times_np[mask]
                ldos_one = compute_ldos(snaps_sub, times_sub, E_values, chi_cpu)
                ldos_sums[t_max] += ldos_one

            elapsed = time.time() - t0
            print(f"  [{j}/{M}] {elapsed:.0f}s", flush=True)

    total_time = time.time() - t0
    print(f"\nTotal propagation time: {total_time:.0f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════════

    results = {}
    spectra = {}
    for t_max in T_MAX_VALUES:
        ldos_avg = ldos_sums[t_max] / M
        spec = integrated_spectrum(ldos_avg, chi_cpu)
        spectra[t_max] = spec

        # Find strongest peak and measure FWHM
        peak_idx = np.argmax(spec)
        fwhm = estimate_fwhm(spec, E_values, peak_idx)

        # Also measure C₂v error at peak
        xe, ye = measure_c2v_error(ldos_avg[peak_idx])

        results[t_max] = dict(
            fwhm=float(fwhm),
            peak_energy=float(E_values[peak_idx]),
            peak_height=float(spec[peak_idx]),
            x_err=float(xe),
            y_err=float(ye),
            delta_E_res=float(2 * np.pi / t_max),
        )
        print(f"  T_max={t_max:3d}: FWHM={fwhm:.4f}, "
              f"δE_res={2*np.pi/t_max:.4f}, "
              f"peak={E_values[peak_idx]:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    # 1. FWHM vs T_max
    Ts = np.array(T_MAX_VALUES, dtype=float)
    fwhms = np.array([results[t]["fwhm"] for t in T_MAX_VALUES])
    delta_E_theory = 2 * np.pi / Ts

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(Ts, fwhms, "ko-", lw=2, ms=8, label="Measured FWHM")
    ax.semilogy(Ts, delta_E_theory, "b--", lw=1.5,
                label=r"$2\pi/T_\mathrm{max}$")
    # Estimate Gamma from the plateau
    gamma_est = np.median(fwhms[fwhms < 2*delta_E_theory[-1]*3])
    if gamma_est > 0:
        ax.axhline(gamma_est, color="r", ls=":", lw=1.5,
                    label=f"Γ_phys ≈ {gamma_est:.3f}")
        ax.semilogy(Ts, np.maximum(gamma_est, delta_E_theory), "g-",
                    lw=1.5, alpha=0.7, label=r"max(Γ, $2\pi/T$)")

    ax.set_xlabel(r"$T_\mathrm{max}$")
    ax.set_ylabel("FWHM of strongest peak")
    ax.set_title("Study 1.2: Spectral resolution convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fwhm_vs_Tmax.png"), dpi=150)
    plt.close(fig)
    print(f"  → fwhm_vs_Tmax.png")

    # 2. Spectra at selected T_max
    fig, ax = plt.subplots(figsize=(8, 4))
    for t_max in [20, 60, 100, 200]:
        if t_max in spectra:
            s = spectra[t_max]
            ax.plot(E_values, s / s.max(), label=f"T={t_max}", alpha=0.8)
    ax.set_xlabel("E")
    ax.set_ylabel("LDOS (normalized)")
    ax.set_title("Spectral convergence with propagation time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectrum_vs_Tmax.png"), dpi=150)
    plt.close(fig)
    print(f"  → spectrum_vs_Tmax.png")

    # 3. Save metrics
    metrics = {str(k): v for k, v in results.items()}
    metrics["total_time_s"] = total_time
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → metrics.json")

    print(f"\n{'='*60}")
    print(f"Study 1.2 Summary:")
    for t_max in [20, 40, 100, 200]:
        if t_max in results:
            r = results[t_max]
            print(f"  T={t_max}: FWHM={r['fwhm']:.4f}, "
                  f"δE_res={r['delta_E_res']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
