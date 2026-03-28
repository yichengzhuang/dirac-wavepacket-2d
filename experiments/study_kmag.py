#!/usr/bin/env python3
"""
Study 1.4: k_mag (initial momentum) dependence of LDOS.

Theory predictions:
    E_peak = vF · k_mag — peak excitation energy
    Effective window: [k_mag - 2σ_E, k_mag + 2σ_E]
    Higher k_mag → probes higher energy → more modes, smaller ΔE
    Peak count should follow Weyl's law: N(E₂)-N(E₁) = ab(E₂³-E₁³)/(2k_harm)

Method:
    Fix σ=0.6, M=48. Scan k_mag = 1.0, 2.0, 3.0, 4.0, 5.0

Output: figures/study_kmag/
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

KMAG_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]

n  = 256
L  = 10.0
vF = 1.0
m  = 0.0
dt = 0.01
n_steps = 10000

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5
cap_strength = 10.0
cap_power = 2
sigma0 = 0.6

N_ang  = 8
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
r_ring = 0.4
ring_angs = np.linspace(0, 2 * np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [
    (r_ring * np.cos(a), r_ring * np.sin(a)) for a in ring_angs
]

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_kmag"))


def main():
    M = N_ang * len(positions)
    print(f"Study 1.4: k_mag scan, M={M}")
    print(f"k_mag values: {KMAG_VALUES}")

    results = {}

    for k_mag in KMAG_VALUES:
        print(f"\n{'─'*50}")
        print(f"k_mag = {k_mag}")

        sigma_E = vF / (2 * sigma0)
        E_lo = max(0.5, k_mag - 2.5)
        E_hi = k_mag + 2.5
        E_values = np.linspace(E_lo, E_hi, 300)

        # Adjust save_every for Nyquist: need E_Nyq > E_hi + 1
        E_nyq_needed = E_hi + 1
        save_every = max(10, int(np.pi / (dt * E_nyq_needed)))
        save_every = min(save_every, 50)

        print(f"  E range: [{E_lo:.1f}, {E_hi:.1f}], save_every={save_every}")

        result = run_ldos_sweep(
            n=n, L=L, vF=vF, m=m,
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

        peak_idx, peak_E = find_peaks_simple(spec, E_values, min_prominence=0.03)
        n_peaks = len(peak_idx)

        strongest = np.argmax(spec)
        xe, ye = measure_c2v_error(ldos_avg[strongest])

        results[k_mag] = dict(
            E_values=E_values,
            spectrum=spec,
            ldos_at_peak=ldos_avg[strongest],
            n_peaks=n_peaks,
            peak_energies=peak_E.tolist(),
            x_err=float(xe),
            y_err=float(ye),
            save_every=save_every,
            elapsed=result["elapsed"],
        )

        print(f"  Result: {n_peaks} peaks, C₂v=({xe:.5f},{ye:.5f})")

        np.save(os.path.join(OUT_DIR, f"ldos_kmag_{k_mag}.npy"), ldos_avg)
        np.save(os.path.join(OUT_DIR, f"E_values_kmag_{k_mag}.npy"), E_values)

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]

    # 1. Spectra at different k_mag (vertically offset)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, k_mag in enumerate(KMAG_VALUES):
        r = results[k_mag]
        spec = r["spectrum"]
        E_vals = r["E_values"]
        spec_norm = spec / spec.max()
        ax.plot(E_vals, spec_norm + i * 0.4, lw=1.5,
                label=f"k_mag={k_mag} ({r['n_peaks']} peaks)")
    ax.set_xlabel("E")
    ax.set_ylabel("LDOS (normalized, offset)")
    ax.set_title("LDOS spectrum at different k_mag")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectra_kmag.png"), dpi=150)
    plt.close(fig)

    # 2. Weyl staircase
    fig, ax = plt.subplots(figsize=(7, 5))
    # Collect all peaks across all k_mag runs
    all_peaks = []
    for k_mag in KMAG_VALUES:
        all_peaks.extend(results[k_mag]["peak_energies"])
    all_peaks = np.sort(np.unique(np.round(all_peaks, 3)))
    cumulative = np.arange(1, len(all_peaks) + 1)

    ax.step(all_peaks, cumulative, "b-", lw=2, label="Detected peaks")

    # Weyl prediction
    a_ell, b_ell, kh = 2.0, 1.2, 3.0
    E_weyl = np.linspace(0.5, 7, 500)
    N_weyl = a_ell * b_ell * E_weyl**3 / (2 * kh)
    ax.plot(E_weyl, N_weyl, "r--", lw=1.5, label="Weyl: $ab E^3/(2k_h)$")

    ax.set_xlabel("E")
    ax.set_ylabel("Cumulative mode count")
    ax.set_title("Mode staircase vs Weyl's law")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "weyl_staircase.png"), dpi=150)
    plt.close(fig)

    # 3. Spatial gallery
    fig, axes = plt.subplots(1, len(KMAG_VALUES),
                             figsize=(4*len(KMAG_VALUES), 4))
    for i, k_mag in enumerate(KMAG_VALUES):
        r = results[k_mag]
        rho = r["ldos_at_peak"]
        ax = axes[i]
        im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno")
        ax.set_title(f"k_mag={k_mag}", fontsize=10)
        ax.set_aspect("equal")
    fig.suptitle("Spatial LDOS at strongest peak vs k_mag", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spatial_gallery.png"), dpi=150)
    plt.close(fig)

    # Metrics
    metrics = {}
    for k_mag in KMAG_VALUES:
        r = results[k_mag]
        metrics[str(k_mag)] = dict(
            n_peaks=r["n_peaks"],
            x_err=r["x_err"],
            y_err=r["y_err"],
            elapsed=r["elapsed"],
        )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Study 1.4 Summary:")
    for k_mag in KMAG_VALUES:
        r = results[k_mag]
        print(f"  k_mag={k_mag}: {r['n_peaks']} peaks, "
              f"C₂v=({r['x_err']:.5f},{r['y_err']:.5f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
