#!/usr/bin/env python3
"""
Study 2.1: Mass dependence of LDOS — Klein tunneling suppression.

Theory predictions:
    - m=0:   Γ/ΔE ~ 1, smooth overlapping peaks
    - m~0.3: Transition region, Γ starts to decrease
    - m≥0.5: Γ/ΔE < 1, peaks become resolvable
    - m≥1:   Sharp peaks, Schrödinger-like behavior
    - C₂v error should be small for ALL m (F-02 confirmed)
    - FWHM(m) should follow max(Γ(m), 2π/T_max)

Method:
    Fix σ=0.6, k_mag=3, M=48.
    m = 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0

This is the CENTRAL RESULT of the paper.

Output: figures/study_mass/
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
    integrated_spectrum, find_peaks_simple, estimate_fwhm, ensure_dir,
)

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

MASS_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

n  = 256
L  = 10.0
vF = 1.0

dt         = 0.01
n_steps    = 10000
save_every = 50

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

sigma0 = 0.6
k_mag  = 3.0

# M=48
N_ang  = 8
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
r_ring    = 0.4
ring_angs = np.linspace(0, 2 * np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [
    (r_ring * np.cos(a), r_ring * np.sin(a)) for a in ring_angs
]

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_mass"))


def main():
    M = N_ang * len(positions)
    print(f"Study 2.1: Mass scan, M={M}")
    print(f"Mass values: {MASS_VALUES}")

    results = {}

    for mass in MASS_VALUES:
        print(f"\n{'─'*50}")
        print(f"m = {mass}")

        # Adjust energy range: E > m (gap edge)
        E_lo = max(0.5, mass + 0.1)
        E_hi = 5.5
        E_values = np.linspace(E_lo, E_hi, 300)

        result = run_ldos_sweep(
            n=n, L=L, vF=vF, m=mass,
            potential_option=POTENTIAL_OPTION, pot_params=pot_params,
            cap_radius=cap_radius, cap_strength=cap_strength,
            cap_power=cap_power,
            sigma0=sigma0, k_mag=k_mag,
            angles=angles, positions=positions,
            dt=dt, n_steps=n_steps, save_every=save_every,
            E_values=E_values,
        )

        ldos_avg = result["ldos_avg"]
        chi_cpu = result["chi_cpu"]
        spec = integrated_spectrum(ldos_avg, chi_cpu)

        # Peak analysis
        peak_idx, peak_E = find_peaks_simple(spec, E_values, min_prominence=0.03)
        n_peaks = len(peak_idx)

        # FWHM of strongest peak
        strongest = np.argmax(spec)
        fwhm = estimate_fwhm(spec, E_values, strongest)

        # C₂v error at strongest peak
        xe, ye = measure_c2v_error(ldos_avg[strongest])

        # Estimate Γ/ΔE from peak analysis
        if n_peaks >= 2:
            # Mean spacing between detected peaks
            spacings = np.diff(np.sort(peak_E))
            mean_spacing = np.mean(spacings) if len(spacings) > 0 else float("nan")
            gamma_over_dE = fwhm / mean_spacing if mean_spacing > 0 else float("nan")
        else:
            mean_spacing = float("nan")
            gamma_over_dE = float("nan")

        results[mass] = dict(
            E_values=E_values,
            spectrum=spec,
            ldos_at_peak=ldos_avg[strongest],
            n_peaks=n_peaks,
            fwhm=float(fwhm),
            x_err=float(xe),
            y_err=float(ye),
            peak_energy=float(E_values[strongest]),
            mean_spacing=float(mean_spacing),
            gamma_over_dE=float(gamma_over_dE),
            elapsed=result["elapsed"],
        )

        print(f"  Result: {n_peaks} peaks, FWHM={fwhm:.4f}, "
              f"Γ/ΔE={gamma_over_dE:.3f}, "
              f"C₂v=({xe:.5f},{ye:.5f}), "
              f"{result['elapsed']:.0f}s")

        # Save LDOS
        np.save(os.path.join(OUT_DIR, f"ldos_mass_{mass}.npy"), ldos_avg)
        np.save(os.path.join(OUT_DIR, f"E_values_mass_{mass}.npy"), E_values)

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]

    # 1. Spectra vs mass (MAIN PAPER FIGURE)
    display_masses = [0.0, 0.3, 0.5, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, mass in enumerate(display_masses):
        if mass in results:
            r = results[mass]
            spec = r["spectrum"]
            E_vals = r["E_values"]
            spec_norm = spec / spec.max()
            ax.plot(E_vals, spec_norm + i * 0.5,
                    lw=1.5, label=f"m={mass}")
    ax.set_xlabel("E (code units)")
    ax.set_ylabel("LDOS (normalized, offset)")
    ax.set_title("LDOS spectrum vs mass: smooth → sharp transition")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectra_vs_mass.png"), dpi=150)
    plt.close(fig)
    print(f"  → spectra_vs_mass.png")

    # 2. FWHM vs mass
    masses = sorted(results.keys())
    fwhms = [results[m_]["fwhm"] for m_ in masses]
    delta_E_res = 2 * np.pi / (n_steps * dt)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(masses, fwhms, "ko-", lw=2, ms=8, label="Measured FWHM")
    ax.axhline(delta_E_res, color="b", ls="--", lw=1,
               label=f"δE_res = {delta_E_res:.4f}")
    ax.set_xlabel("Mass m")
    ax.set_ylabel("FWHM of strongest peak")
    ax.set_title("Peak width vs mass\n"
                 "(Klein tunneling suppression)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fwhm_vs_mass.png"), dpi=150)
    plt.close(fig)
    print(f"  → fwhm_vs_mass.png")

    # 3. Γ/ΔE vs mass
    g_over_dE = [results[m_]["gamma_over_dE"] for m_ in masses]
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = [(m_, g) for m_, g in zip(masses, g_over_dE) if np.isfinite(g)]
    if valid:
        ms_v, gs_v = zip(*valid)
        ax.plot(ms_v, gs_v, "ko-", lw=2, ms=8)
        ax.axhline(1.0, color="r", ls="--", lw=1.5, label="Γ/ΔE = 1")
        ax.set_xlabel("Mass m")
        ax.set_ylabel("Γ/ΔE (estimated from LDOS)")
        ax.set_title("Mode overlap ratio vs mass\n"
                     "(>1: overlapping, <1: resolvable)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gamma_over_deltaE.png"), dpi=150)
    plt.close(fig)
    print(f"  → gamma_over_deltaE.png")

    # 4. Spatial gallery
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, mass in enumerate(MASS_VALUES):
        row, col = divmod(i, 5)
        ax = axes[row, col]
        rho = results[mass]["ldos_at_peak"]
        im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno")
        E_pk = results[mass]["peak_energy"]
        ax.set_title(f"m={mass}\nE={E_pk:.2f}", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    # Hide unused subplot if any
    if len(MASS_VALUES) < 10:
        axes[1, 4].axis("off")
    fig.suptitle("Spatial LDOS at strongest peak vs mass", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spatial_gallery.png"), dpi=150)
    plt.close(fig)
    print(f"  → spatial_gallery.png")

    # 5. C₂v error vs mass
    x_errs = [results[m_]["x_err"] for m_ in masses]
    y_errs = [results[m_]["y_err"] for m_ in masses]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(masses, x_errs, "bo-", label="x_err", ms=8)
    ax.semilogy(masses, y_errs, "rs-", label="y_err", ms=8)
    ax.set_xlabel("Mass m")
    ax.set_ylabel("C₂v error")
    ax.set_title("C₂v symmetry error vs mass\n"
                 "(prediction: small for all m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "c2v_vs_mass.png"), dpi=150)
    plt.close(fig)
    print(f"  → c2v_vs_mass.png")

    # 6. Metrics
    metrics = {}
    for mass in masses:
        r = results[mass]
        metrics[str(mass)] = dict(
            n_peaks=r["n_peaks"],
            fwhm=r["fwhm"],
            x_err=r["x_err"],
            y_err=r["y_err"],
            peak_energy=r["peak_energy"],
            mean_spacing=r["mean_spacing"],
            gamma_over_dE=r["gamma_over_dE"],
            elapsed=r["elapsed"],
        )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Study 2.1 Summary (CORE RESULT):")
    print(f"{'m':>6} {'FWHM':>8} {'Γ/ΔE':>8} {'n_peaks':>8} "
          f"{'x_err':>10} {'y_err':>10}")
    for mass in masses:
        r = results[mass]
        print(f"{mass:6.1f} {r['fwhm']:8.4f} {r['gamma_over_dE']:8.3f} "
              f"{r['n_peaks']:8d} {r['x_err']:10.5f} {r['y_err']:10.5f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
