#!/usr/bin/env python3
"""
CAP Step C (refined): Compare C1 vs C5 with M=48 to reduce statistical noise.

Previous M=16 test showed FWHM fluctuations of 40-80% that are likely
statistical, not physical. With M=48, the C₂v error drops to <0.05%
(established by Study 1.1), so spectral differences should be meaningful.

Only two configs:
  C1: L=10, n=256, cap_radius=3.5   (current default)
  C5: L=20, n=512, cap_radius=8.0   (safest)

Compare: integrated spectrum, FWHM, peak positions inside common mask.

Output: figures/cap_validate/
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
    run_ldos_sweep, integrated_spectrum, measure_c2v_error,
    find_peaks_simple, estimate_fwhm, ensure_dir,
)

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "cap_validate"))

# Only two extreme configs
CONFIGS = [
    (10.0, 256, 3.5, "C1_L10_R3.5"),
    (20.0, 512, 8.0, "C5_L20_R8.0"),
]

# Fixed physics
vF = 1.0; m = 0.0; dt = 0.01
n_steps = 10000; save_every = 50

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_strength = 10.0; cap_power = 2
sigma0 = 0.6; k_mag = 3.0

# M=48 (8 angles × 6 positions) — same as the validated ldos_average.py
N_ang = 8
angles = np.linspace(0, 2*np.pi, N_ang, endpoint=False)
r_ring = 0.4
ring_angs = np.linspace(0, 2*np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [
    (r_ring * np.cos(a), r_ring * np.sin(a)) for a in ring_angs
]

E_values = np.linspace(0.5, 5.5, 300)


def build_common_mask(X_np, Y_np, aR=2.6, bR=1.56):
    rho2 = (X_np / aR)**2 + (Y_np / bR)**2
    return (rho2 <= 1.0).astype(np.float64)


def main():
    M = N_ang * len(positions)
    print("=" * 60)
    print(f"CAP convergence (M={M}): C1 vs C5")
    print("=" * 60)

    results = {}
    spectra = {}

    for L, n_grid, cap_radius, label in CONFIGS:
        print(f"\n{'═'*60}")
        print(f"{label}: L={L}, n={n_grid}, cap_R={cap_radius}")

        result = run_ldos_sweep(
            n=n_grid, L=L, vF=vF, m=m,
            potential_option=POTENTIAL_OPTION, pot_params=pot_params,
            cap_radius=cap_radius, cap_strength=cap_strength,
            cap_power=cap_power,
            sigma0=sigma0, k_mag=k_mag,
            angles=angles, positions=positions,
            dt=dt, n_steps=n_steps, save_every=save_every,
            E_values=E_values,
        )

        ldos_avg = result["ldos_avg"]

        # Common mask
        x_np = np.linspace(-L/2, L/2, n_grid, endpoint=False)
        X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
        common_mask = build_common_mask(X_np, Y_np)

        spec = (ldos_avg * common_mask[np.newaxis]).sum(axis=(1, 2))
        spectra[label] = spec

        peak_idx, peak_E = find_peaks_simple(spec, E_values, min_prominence=0.03)
        strongest = np.argmax(spec)
        fwhm = estimate_fwhm(spec, E_values, strongest)
        xe, ye = measure_c2v_error(ldos_avg[strongest] * common_mask)

        results[label] = dict(
            L=L, n=n_grid, cap_radius=cap_radius,
            n_peaks=len(peak_idx), peak_energies=[float(e) for e in peak_E],
            peak_energy=float(E_values[strongest]),
            fwhm=float(fwhm),
            x_err=float(xe), y_err=float(ye),
            elapsed=result["elapsed"],
        )

        print(f"  {len(peak_idx)} peaks, E_peak={E_values[strongest]:.3f}, "
              f"FWHM={fwhm:.4f}, C₂v=({xe:.5f},{ye:.5f})")

        np.save(os.path.join(OUT_DIR, f"ldos_M48_{label}.npy"), ldos_avg)

    # ── Comparison ──
    c1 = "C1_L10_R3.5"
    c5 = "C5_L20_R8.0"
    s1 = spectra[c1] / spectra[c1].max()
    s5 = spectra[c5] / spectra[c5].max()

    l2 = np.sqrt(np.mean((s1 - s5)**2))
    peak_shift = abs(results[c1]["peak_energy"] - results[c5]["peak_energy"])
    fwhm_diff = abs(results[c1]["fwhm"] - results[c5]["fwhm"])
    fwhm_ref = results[c5]["fwhm"]
    fwhm_rel = fwhm_diff / fwhm_ref if fwhm_ref > 0 else 0

    # Peak-by-peak matching
    peaks1 = np.array(results[c1]["peak_energies"])
    peaks5 = np.array(results[c5]["peak_energies"])
    matched_shifts = []
    for p1 in peaks1:
        if len(peaks5) > 0:
            idx = np.argmin(np.abs(peaks5 - p1))
            matched_shifts.append(abs(p1 - peaks5[idx]))
    mean_peak_shift = np.mean(matched_shifts) if matched_shifts else float("nan")

    print(f"\n{'='*60}")
    print(f"C1 vs C5 comparison (M={M}):")
    print(f"  L2 spectral error:     {l2:.4f}")
    print(f"  Strongest peak shift:  {peak_shift:.4f}")
    print(f"  Mean peak-by-peak shift: {mean_peak_shift:.4f}")
    print(f"  FWHM: C1={results[c1]['fwhm']:.4f}, "
          f"C5={results[c5]['fwhm']:.4f}, "
          f"rel_diff={fwhm_rel:.3f}")
    print(f"  C₂v: C1=({results[c1]['x_err']:.5f},{results[c1]['y_err']:.5f}), "
          f"C5=({results[c5]['x_err']:.5f},{results[c5]['y_err']:.5f})")

    verdict = "CONVERGED" if l2 < 0.03 and fwhm_rel < 0.15 else \
              "MARGINAL" if l2 < 0.08 and fwhm_rel < 0.30 else "NOT CONVERGED"
    print(f"\n  *** VERDICT: {verdict} ***")
    print(f"{'='*60}")

    # ── Figure ──
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(E_values, s1, "b-", lw=1.5, label=f"C1: L=10, R=3.5 (current)")
    ax.plot(E_values, s5, "r--", lw=1.5, label=f"C5: L=20, R=8.0 (reference)")
    ax.set_ylabel("LDOS (normalized)")
    ax.set_title(f"CAP convergence (M={M}): mask-internal LDOS\n"
                 f"L2={l2:.4f}, FWHM_rel={fwhm_rel:.2f} → {verdict}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff = np.abs(s1 - s5)
    ax.semilogy(E_values, diff + 1e-15, "k-", lw=1)
    ax.axhline(0.05, color="r", ls="--", label="5%")
    ax.axhline(0.01, color="g", ls="--", label="1%")
    ax.set_xlabel("E")
    ax.set_ylabel("|ΔLDOS|")
    ax.set_title("Absolute difference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cap_convergence_M48.png"), dpi=150)
    plt.close(fig)
    print(f"  → cap_convergence_M48.png")

    # Save
    metrics = dict(
        l2_error=float(l2),
        peak_shift=float(peak_shift),
        mean_peak_shift=float(mean_peak_shift),
        fwhm_C1=results[c1]["fwhm"],
        fwhm_C5=results[c5]["fwhm"],
        fwhm_rel_diff=float(fwhm_rel),
        verdict=verdict,
        configs=results,
    )
    with open(os.path.join(OUT_DIR, "cap_M48_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
