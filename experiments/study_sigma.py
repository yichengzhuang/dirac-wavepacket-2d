#!/usr/bin/env python3
"""
Study 1.3: Wavepacket width σ dependence of LDOS.

Theory predictions:
    σ_E = vF/(2σ) — energy bandwidth
    N_modes ≈ 4σ_E / ΔE(E_peak) — modes excited
    C₂v error should be independent of σ (exact symmetry)
    Small σ (broadband) → smoother LDOS, more modes superimposed
    Large σ (narrowband) → sharper peaks, fewer modes

Method:
    Fix k_mag=3, M=48. Scan σ = 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5

Output: figures/study_sigma/
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
    run_ldos_sweep, fft_flip, symmetrize_c2v, measure_c2v_error,
    integrated_spectrum, find_peaks_simple, ensure_dir,
)

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

SIGMA_VALUES = [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5]

n  = 256
L  = 10.0
vF = 1.0
m  = 0.0

dt         = 0.01
n_steps    = 10000
save_every = 50

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

k_mag = 3.0

# M=48 (8 angles × 6 positions)
N_ang  = 8
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
r_ring    = 0.4
ring_angs = np.linspace(0, 2 * np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [
    (r_ring * np.cos(a), r_ring * np.sin(a)) for a in ring_angs
]

E_values = np.linspace(0.5, 5.5, 300)

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_sigma"))


def main():
    M = N_ang * len(positions)
    print(f"Study 1.3: σ dependence, M={M}")
    print(f"σ values: {SIGMA_VALUES}")

    results = {}

    for sigma in SIGMA_VALUES:
        print(f"\n{'─'*50}")
        print(f"σ = {sigma}")
        sigma_E = vF / (2 * sigma)
        E_window = [k_mag - 2*sigma_E, k_mag + 2*sigma_E]
        print(f"  Theory: σ_E = {sigma_E:.3f}, "
              f"window = [{E_window[0]:.2f}, {E_window[1]:.2f}]")

        result = run_ldos_sweep(
            n=n, L=L, vF=vF, m=m,
            potential_option=POTENTIAL_OPTION, pot_params=pot_params,
            cap_radius=cap_radius, cap_strength=cap_strength,
            cap_power=cap_power,
            sigma0=sigma, k_mag=k_mag,
            angles=angles, positions=positions,
            dt=dt, n_steps=n_steps, save_every=save_every,
            E_values=E_values,
        )

        ldos_avg = result["ldos_avg"]
        chi_cpu = result["chi_cpu"]
        spec = integrated_spectrum(ldos_avg, chi_cpu)

        # Find peaks
        peak_idx, peak_E = find_peaks_simple(spec, E_values, min_prominence=0.05)
        n_peaks = len(peak_idx)

        # C₂v error at strongest peak
        strongest = np.argmax(spec)
        xe, ye = measure_c2v_error(ldos_avg[strongest])

        # Dynamic range: max/median of spectrum inside [1,5]
        mask_E = (E_values > 1) & (E_values < 5)
        dyn_range = spec[mask_E].max() / max(np.median(spec[mask_E]), 1e-30)

        results[sigma] = dict(
            sigma_E=float(sigma_E),
            E_window=E_window,
            n_peaks=n_peaks,
            x_err=float(xe),
            y_err=float(ye),
            peak_energy=float(E_values[strongest]),
            dynamic_range=float(dyn_range),
            spectrum=spec,
            ldos_at_peak=ldos_avg[strongest],
            elapsed=result["elapsed"],
        )

        print(f"  Result: {n_peaks} peaks, "
              f"C₂v=({xe:.5f},{ye:.5f}), "
              f"dyn_range={dyn_range:.1f}, "
              f"{result['elapsed']:.0f}s")

        # Save LDOS for later comparison
        np.save(os.path.join(OUT_DIR, f"ldos_sigma_{sigma}.npy"), ldos_avg)

    np.save(os.path.join(OUT_DIR, "E_values.npy"), E_values)

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    sigmas = SIGMA_VALUES
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]

    # 1. Excitation profiles (theoretical)
    fig, ax = plt.subplots(figsize=(8, 4))
    E_plot = np.linspace(0, 7, 500)
    for sigma in sigmas:
        sigma_E = vF / (2 * sigma)
        profile = np.exp(-(E_plot - k_mag)**2 / (2 * sigma_E**2))
        ax.plot(E_plot, profile, label=f"σ={sigma} (σ_E={sigma_E:.2f})")

    # Add Weyl level markers at E=3
    a_ell, b_ell, kh = 2.0, 1.2, 3.0
    delta_E_3 = 2*kh / (3*a_ell*b_ell*9)
    for i in range(30):
        E_level = 3 + (i - 15) * delta_E_3
        if 0.5 < E_level < 5.5:
            ax.axvline(E_level, color="gray", alpha=0.15, lw=0.5)

    ax.set_xlabel("E")
    ax.set_ylabel("Excitation probability")
    ax.set_title("Theoretical excitation profiles for different σ\n"
                 "(gray lines: estimated Weyl levels near E=3)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "excitation_profiles.png"), dpi=150)
    plt.close(fig)

    # 2. LDOS spectra comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, sigma in enumerate(sigmas):
        spec = results[sigma]["spectrum"]
        spec_norm = spec / spec.max()
        ax.plot(E_values, spec_norm + i * 0.3,
                label=f"σ={sigma}", alpha=0.8)
    ax.set_xlabel("E")
    ax.set_ylabel("LDOS (normalized, offset)")
    ax.set_title("Integrated LDOS spectrum vs σ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ldos_spectra.png"), dpi=150)
    plt.close(fig)

    # 3. C₂v error vs σ
    fig, ax = plt.subplots(figsize=(6, 4))
    x_errs = [results[s]["x_err"] for s in sigmas]
    y_errs = [results[s]["y_err"] for s in sigmas]
    ax.semilogy(sigmas, x_errs, "bo-", label="x_err", ms=8)
    ax.semilogy(sigmas, y_errs, "rs-", label="y_err", ms=8)
    ax.set_xlabel("σ (wavepacket width)")
    ax.set_ylabel("C₂v error")
    ax.set_title("C₂v symmetry error vs σ\n"
                 "(prediction: independent of σ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "c2v_vs_sigma.png"), dpi=150)
    plt.close(fig)

    # 4. Spatial comparison: smallest σ vs largest σ
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for i, sigma in enumerate([sigmas[0], sigmas[-1]]):
        rho = results[sigma]["ldos_at_peak"]
        ax = axes[i]
        im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno")
        xe = results[sigma]["x_err"]
        ye = results[sigma]["y_err"]
        E_pk = results[sigma]["peak_energy"]
        ax.set_title(f"σ={sigma}, E={E_pk:.2f}\n"
                     f"C₂v=({xe:.4f},{ye:.4f})")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Spatial LDOS at strongest peak", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spatial_comparison.png"), dpi=150)
    plt.close(fig)

    # 5. Metrics
    metrics = {}
    for sigma in sigmas:
        r = results[sigma]
        metrics[str(sigma)] = dict(
            sigma_E=r["sigma_E"],
            n_peaks=r["n_peaks"],
            x_err=r["x_err"],
            y_err=r["y_err"],
            dynamic_range=r["dynamic_range"],
            elapsed=r["elapsed"],
        )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Study 1.3 Summary:")
    print(f"{'σ':>6} {'σ_E':>6} {'n_peaks':>8} {'x_err':>10} {'y_err':>10} {'dyn_range':>10}")
    for sigma in sigmas:
        r = results[sigma]
        print(f"{sigma:6.2f} {r['sigma_E']:6.3f} {r['n_peaks']:8d} "
              f"{r['x_err']:10.5f} {r['y_err']:10.5f} {r['dynamic_range']:10.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
