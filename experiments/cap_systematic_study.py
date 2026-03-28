#!/usr/bin/env python3
"""
CAP Step C: Systematic Grid/CAP convergence.

Question: Does the CAP boundary affect the LDOS *inside the analysis mask*?

Method:
    Keep dx ≈ 0.039 constant, vary (L, n, cap_radius).
    For each config, run M=16 LDOS, compare the integrated spectrum
    and spatial maps INSIDE the chi_R mask (which is identical across configs).

    The mask is an ellipse (aR≈2.60, bR≈1.56), well inside all CAP configs.
    If LDOS inside this mask is the same for all configs, the CAP is fine.

Configs:
    C1: L=10,  n=256,  cap_radius=3.5   (current default)
    C2: L=10,  n=256,  cap_radius=4.5
    C3: L=15,  n=384,  cap_radius=5.5
    C4: L=15,  n=384,  cap_radius=6.5
    C5: L=20,  n=512,  cap_radius=8.0   (reference, most safe)

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

# ══════════════════════════════════════════════════════════════════════════════
# Configs: (L, n, cap_radius, label)
# ══════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    (10.0, 256, 3.5, "C1_L10_R3.5"),
    (10.0, 256, 4.5, "C2_L10_R4.5"),
    (15.0, 384, 5.5, "C3_L15_R5.5"),
    (15.0, 384, 6.5, "C4_L15_R6.5"),
    (20.0, 512, 8.0, "C5_L20_R8.0"),
]

# Fixed physics
vF = 1.0
m  = 0.0
dt = 0.01
n_steps = 10000
save_every = 50

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_strength = 10.0
cap_power = 2

sigma0 = 0.6
k_mag  = 3.0

# M=16 for speed (4 angles × 4 positions)
N_ang = 4
angles = np.linspace(0, 2*np.pi, N_ang, endpoint=False)
positions = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (-0.3, 0.0)]

E_values = np.linspace(0.5, 5.5, 300)

# Common mask for comparison: use the SMALLEST mask across all configs.
# For harmonic ellipse (a=2, b=1.2), mask_margin=1.30 gives
#   aR = min(1.30*2, 0.95*cap_radius)
#   bR = min(1.30*1.2, 0.95*cap_radius)
# The smallest cap_radius is 3.5, so aR = min(2.6, 3.325) = 2.6
# We'll build a common comparison mask at (aR=2.6, bR=1.56) on each grid.


def build_common_mask(X_np, Y_np, aR=2.6, bR=1.56):
    """Elliptical mask identical across all grid sizes."""
    rho2 = (X_np / aR)**2 + (Y_np / bR)**2
    return (rho2 <= 1.0).astype(np.float64)


def main():
    print("=" * 60)
    print("CAP Step C: Systematic Grid/CAP convergence")
    print(f"Configs: {len(CONFIGS)}")
    print(f"M = {N_ang * len(positions)} propagations per config")
    print(f"Common comparison mask: ellipse (2.60 × 1.56)")
    print("=" * 60)

    results = {}
    spectra = {}

    for L, n_grid, cap_radius, label in CONFIGS:
        print(f"\n{'═'*60}")
        dx = L / n_grid
        print(f"{label}: L={L}, n={n_grid}, dx={dx:.4f}, cap_R={cap_radius}")

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

        # Build common mask on THIS grid
        x_np = np.linspace(-L/2, L/2, n_grid, endpoint=False)
        X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
        common_mask = build_common_mask(X_np, Y_np)

        # Integrated spectrum inside common mask
        spec = (ldos_avg * common_mask[np.newaxis]).sum(axis=(1, 2))
        spectra[label] = spec

        # Peak analysis
        peak_idx, peak_E = find_peaks_simple(spec, E_values, min_prominence=0.03)

        # FWHM at strongest peak
        strongest = np.argmax(spec)
        fwhm = estimate_fwhm(spec, E_values, strongest)
        xe, ye = measure_c2v_error(ldos_avg[strongest] * common_mask)

        results[label] = dict(
            L=L, n=n_grid, cap_radius=cap_radius, dx=dx,
            n_peaks=len(peak_idx),
            peak_energy=float(E_values[strongest]),
            fwhm=float(fwhm),
            x_err=float(xe), y_err=float(ye),
            elapsed=result["elapsed"],
            peak_energies=[float(e) for e in peak_E],
        )

        print(f"  {len(peak_idx)} peaks, strongest E={E_values[strongest]:.3f}, "
              f"FWHM={fwhm:.4f}, C₂v=({xe:.5f},{ye:.5f}), "
              f"{result['elapsed']:.0f}s")

        # Save LDOS for later comparison
        np.save(os.path.join(OUT_DIR, f"ldos_{label}.npy"), ldos_avg)

    np.save(os.path.join(OUT_DIR, "E_values_cap_study.npy"), E_values)

    # ══════════════════════════════════════════════════════════════════════
    # Convergence analysis: compare all configs to C5 (reference)
    # ══════════════════════════════════════════════════════════════════════

    ref_label = "C5_L20_R8.0"
    ref_spec = spectra[ref_label]
    ref_spec_norm = ref_spec / ref_spec.max()

    print(f"\n{'═'*60}")
    print(f"Convergence relative to {ref_label}:")

    for label in [c[3] for c in CONFIGS]:
        spec = spectra[label]
        spec_norm = spec / spec.max()
        # L2 relative error
        l2_err = np.sqrt(np.mean((spec_norm - ref_spec_norm)**2))
        # Peak position difference
        peak_shift = abs(results[label]["peak_energy"] -
                         results[ref_label]["peak_energy"])
        # FWHM difference
        fwhm_diff = abs(results[label]["fwhm"] - results[ref_label]["fwhm"])
        fwhm_rel = fwhm_diff / results[ref_label]["fwhm"] if results[ref_label]["fwhm"] > 0 else 0

        results[label]["l2_vs_ref"] = float(l2_err)
        results[label]["peak_shift_vs_ref"] = float(peak_shift)
        results[label]["fwhm_rel_diff_vs_ref"] = float(fwhm_rel)

        status = "✓" if l2_err < 0.05 else ("⚠" if l2_err < 0.10 else "✗")
        print(f"  {label}: L2={l2_err:.4f} {status}, "
              f"peak_shift={peak_shift:.4f}, "
              f"FWHM_rel_diff={fwhm_rel:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    # 1. Spectrum comparison (all configs overlaid)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    for label in [c[3] for c in CONFIGS]:
        spec = spectra[label]
        ax.plot(E_values, spec / spec.max(), label=label, alpha=0.8)
    ax.set_ylabel("LDOS (normalized)")
    ax.set_title("CAP convergence: integrated LDOS inside common mask (2.60×1.56)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: difference from reference
    ax = axes[1]
    for label in [c[3] for c in CONFIGS[:-1]]:
        spec = spectra[label]
        diff = np.abs(spec/spec.max() - ref_spec_norm)
        ax.semilogy(E_values, diff + 1e-15, label=label, alpha=0.8)
    ax.axhline(0.05, color="r", ls="--", lw=1, label="5% threshold")
    ax.set_xlabel("E")
    ax.set_ylabel("|ΔLDOS| / max (vs C5)")
    ax.set_title("Difference from reference (C5: L=20, cap_R=8)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cap_convergence_spectra.png"), dpi=150)
    plt.close(fig)
    print(f"\n  → cap_convergence_spectra.png")

    # 2. Summary bar chart
    labels_short = [c[3].split("_", 1)[1] for c in CONFIGS[:-1]]
    l2_errs = [results[c[3]]["l2_vs_ref"] for c in CONFIGS[:-1]]
    fwhm_rels = [results[c[3]]["fwhm_rel_diff_vs_ref"] for c in CONFIGS[:-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.bar(labels_short, l2_errs, color=["red" if e > 0.1 else "orange" if e > 0.05 else "green" for e in l2_errs])
    ax.axhline(0.05, color="r", ls="--")
    ax.set_ylabel("L2 spectral error vs C5")
    ax.set_title("Spectral convergence")

    ax = axes[1]
    ax.bar(labels_short, fwhm_rels, color=["red" if e > 0.1 else "orange" if e > 0.05 else "green" for e in fwhm_rels])
    ax.axhline(0.1, color="r", ls="--")
    ax.set_ylabel("Relative FWHM difference vs C5")
    ax.set_title("Peak width convergence")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cap_convergence_summary.png"), dpi=150)
    plt.close(fig)
    print(f"  → cap_convergence_summary.png")

    # 3. Save metrics
    metrics_out = {k: {kk: vv for kk, vv in v.items()}
                   for k, v in results.items()}
    with open(os.path.join(OUT_DIR, "cap_systematic_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  → cap_systematic_metrics.json")

    # Summary table
    print(f"\n{'='*70}")
    print("CAP Step C Summary:")
    print(f"{'Config':>15} {'L':>4} {'n':>4} {'R_cap':>5} {'peaks':>5} "
          f"{'FWHM':>7} {'L2_err':>7} {'status':>7}")
    for L, n_grid, cap_radius, label in CONFIGS:
        r = results[label]
        l2 = r.get("l2_vs_ref", 0)
        status = "REF" if label == ref_label else \
                 ("✓" if l2 < 0.05 else ("⚠" if l2 < 0.1 else "✗"))
        print(f"{label:>15} {L:4.0f} {n_grid:4d} {cap_radius:5.1f} "
              f"{r['n_peaks']:5d} {r['fwhm']:7.4f} {l2:7.4f} {status:>7}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
