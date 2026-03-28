#!/usr/bin/env python3
"""
Study 1.5: Grid resolution convergence.

Theory: Split-operator error ~ O(dx²). At E=5, need ~10 pts/wavelength.
    n=64:  dx=0.156, 8 pts/λ  — marginal
    n=128: dx=0.078, 16 pts/λ — adequate
    n=256: dx=0.039, 32 pts/λ — well converged
    n=512: dx=0.020, 64 pts/λ — reference

Method: M=16 (fast), n = 64, 128, 256, 512.

Output: figures/study_grid/
"""

import sys, os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "scripts"))
sys.path.insert(0, _root)

from ldos_utils import (
    run_ldos_sweep, integrated_spectrum, measure_c2v_error, ensure_dir,
)

N_VALUES = [64, 128, 256, 512]

L  = 10.0
vF = 1.0
m  = 0.0
dt = 0.01
n_steps = 10000
save_every = 50

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5
cap_strength = 10.0
cap_power = 2
sigma0 = 0.6
k_mag = 3.0

# M=16 for speed
N_ang = 4
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
positions = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (-0.3, 0.0)]

E_values = np.linspace(0.5, 5.5, 200)

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_grid"))


def main():
    M = N_ang * len(positions)
    print(f"Study 1.5: Grid resolution, M={M}")

    results = {}
    spectra = {}

    for n_grid in N_VALUES:
        print(f"\n{'─'*50}")
        dx = L / n_grid
        print(f"n = {n_grid}, dx = {dx:.4f}")

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
        chi_cpu = result["chi_cpu"]
        spec = integrated_spectrum(ldos_avg, chi_cpu)
        spectra[n_grid] = spec

        strongest = np.argmax(spec)
        xe, ye = measure_c2v_error(ldos_avg[strongest])

        results[n_grid] = dict(
            dx=float(dx),
            x_err=float(xe),
            y_err=float(ye),
            peak_energy=float(E_values[strongest]),
            peak_height=float(spec[strongest]),
            elapsed=result["elapsed"],
        )

        print(f"  peak E={E_values[strongest]:.3f}, "
              f"C₂v=({xe:.5f},{ye:.5f}), {result['elapsed']:.0f}s")

        np.save(os.path.join(OUT_DIR, f"spectrum_n{n_grid}.npy"), spec)

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    # 1. Spectra comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    for n_grid in N_VALUES:
        spec = spectra[n_grid]
        ax.plot(E_values, spec / spec.max(), label=f"n={n_grid}", alpha=0.8)
    ax.set_xlabel("E")
    ax.set_ylabel("Integrated LDOS (normalized)")
    ax.set_title("Grid resolution convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectrum_vs_n.png"), dpi=150)
    plt.close(fig)

    # 2. Error vs dx (relative to n=512 reference)
    ref_spec = spectra[512] / spectra[512].max()
    dxs = []
    errors = []
    for n_grid in N_VALUES[:-1]:
        dx = L / n_grid
        spec_norm = spectra[n_grid] / spectra[n_grid].max()
        err = np.sqrt(np.mean((spec_norm - ref_spec)**2))
        dxs.append(dx)
        errors.append(err)
        print(f"  n={n_grid}: L2 error vs n=512 = {err:.6f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(dxs, errors, "ko-", lw=2, ms=8, label="Measured")
    # O(dx²) reference
    dx_ref = np.array(dxs)
    ax.loglog(dx_ref, errors[0] * (dx_ref/dxs[0])**2, "r--",
              label=r"$O(dx^2)$")
    ax.set_xlabel("dx = L/n")
    ax.set_ylabel("L2 spectral error vs n=512")
    ax.set_title("Grid convergence order")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "error_vs_dx.png"), dpi=150)
    plt.close(fig)

    # Metrics
    metrics = {str(k): v for k, v in results.items()}
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Study 1.5 Summary:")
    for n_grid in N_VALUES:
        r = results[n_grid]
        print(f"  n={n_grid}: dx={r['dx']:.4f}, "
              f"C₂v=({r['x_err']:.5f},{r['y_err']:.5f}), "
              f"{r['elapsed']:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
