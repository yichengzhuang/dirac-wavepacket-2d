#!/usr/bin/env python3
"""
Study 2.3: Confinement stiffness k_harm dependence.

Theory:
    ΔE = 2k_harm / (3ab·E²)  — level spacing ∝ k_harm
    N_peaks(E1,E2) = ab(E2³-E1³)/(2k_harm) — modes ∝ 1/k_harm
    Larger k_harm → tighter confinement, fewer modes, larger spacing

Method: k_harm = 1, 2, 3, 5, 10. M=24. Harmonic ellipse (a=2, b=1.2).

Output: figures/study_kharm/
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
    run_ldos_sweep, integrated_spectrum, find_peaks_simple,
    measure_c2v_error, ensure_dir,
)

KHARM_VALUES = [1.0, 2.0, 3.0, 5.0, 10.0]

n  = 256
L  = 10.0
vF = 1.0
m  = 0.0
dt = 0.01
n_steps = 10000
save_every = 50

cap_radius = 3.5
cap_strength = 10.0
cap_power = 2
sigma0 = 0.6
k_mag = 3.0

# M=24
N_ang = 6
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
positions = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (-0.3, 0.0)]

E_values = np.linspace(0.5, 5.5, 300)

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_kharm"))


def main():
    M = N_ang * len(positions)
    print(f"Study 2.3: k_harm scan, M={M}")

    results = {}

    for kh in KHARM_VALUES:
        print(f"\n{'─'*50}")
        print(f"k_harm = {kh}")

        pot_params = dict(k_harm=kh, a=2.0, b=1.2)
        # Theory prediction for level spacing at E=3
        delta_E_theory = 2*kh / (3*2.0*1.2*9)
        N_theory = 2.0*1.2*(5.5**3 - 0.5**3) / (2*kh)
        print(f"  Theory: ΔE(E=3)={delta_E_theory:.4f}, "
              f"N_modes(0.5-5.5)≈{N_theory:.0f}")

        result = run_ldos_sweep(
            n=n, L=L, vF=vF, m=m,
            potential_option="harmonic_ellipse", pot_params=pot_params,
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

        # Mean spacing from detected peaks
        if n_peaks >= 2:
            spacings = np.diff(np.sort(peak_E))
            mean_spacing = float(spacings.mean())
        else:
            mean_spacing = float("nan")

        strongest = np.argmax(spec)
        xe, ye = measure_c2v_error(ldos_avg[strongest])

        results[kh] = dict(
            spectrum=spec,
            n_peaks=n_peaks,
            mean_spacing=mean_spacing,
            delta_E_theory=float(delta_E_theory),
            N_theory=float(N_theory),
            x_err=float(xe),
            y_err=float(ye),
            elapsed=result["elapsed"],
        )

        print(f"  Result: {n_peaks} peaks (theory: ~{N_theory:.0f}), "
              f"<s>={mean_spacing:.4f} (theory: {delta_E_theory:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    # 1. Spectra vs k_harm
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, kh in enumerate(KHARM_VALUES):
        spec = results[kh]["spectrum"]
        spec_norm = spec / spec.max()
        ax.plot(E_values, spec_norm + i * 0.4, lw=1.5,
                label=f"k_harm={kh} ({results[kh]['n_peaks']} peaks)")
    ax.set_xlabel("E")
    ax.set_ylabel("LDOS (normalized, offset)")
    ax.set_title("LDOS spectrum vs confinement stiffness")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectra_vs_kharm.png"), dpi=150)
    plt.close(fig)

    # 2. Peak count vs k_harm
    khs = np.array(KHARM_VALUES)
    n_peaks_arr = np.array([results[kh]["n_peaks"] for kh in KHARM_VALUES])
    n_theory_arr = np.array([results[kh]["N_theory"] for kh in KHARM_VALUES])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(khs, n_peaks_arr, "ko-", lw=2, ms=8, label="Detected peaks")
    ax.plot(khs, n_theory_arr, "r--", lw=1.5, label="Weyl prediction")
    ax.set_xlabel("k_harm")
    ax.set_ylabel("Number of peaks in [0.5, 5.5]")
    ax.set_title("Mode count vs confinement stiffness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "peaks_vs_kharm.png"), dpi=150)
    plt.close(fig)

    # Metrics
    metrics = {}
    for kh in KHARM_VALUES:
        r = results[kh]
        metrics[str(kh)] = dict(
            n_peaks=r["n_peaks"],
            mean_spacing=r["mean_spacing"],
            delta_E_theory=r["delta_E_theory"],
            x_err=r["x_err"],
            y_err=r["y_err"],
        )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Study 2.3 Summary:")
    for kh in KHARM_VALUES:
        r = results[kh]
        print(f"  k_harm={kh}: {r['n_peaks']} peaks "
              f"(Weyl: {r['N_theory']:.0f}), "
              f"<s>={r['mean_spacing']:.4f} "
              f"(theory: {r['delta_E_theory']:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
