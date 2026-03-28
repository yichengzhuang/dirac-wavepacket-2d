#!/usr/bin/env python3
"""
Study 2.4: Soft-wall potential parameter dependence.

Theory predictions:
    V₀ (barrier height): V₀ < 2E → poor confinement, broad resonances
    μ  (wall steepness):  large μ → hard wall; effective barrier width ~ 1/μ
    λ  (interior flatness): large λ → flatter bottom, more billiard-like

Method: Soft-wall ellipse (a=2, b=1.2). M=24. Scan one parameter at a time.

Output: figures/study_softwall/
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
    measure_c2v_error, estimate_fwhm, ensure_dir,
)

# ── Scan configurations ──

# Each scan: vary one param, fix others at default
SCANS = {
    "V0": {
        "values": [10, 30, 50, 70, 100, 200],
        "fixed": dict(lam=25, mu=5),
        "label": r"$V_0$",
    },
    "lambda": {
        "values": [5, 15, 25, 50],
        "fixed": dict(V0=70, mu=5),
        "label": r"$\lambda$",
    },
    "mu": {
        "values": [1, 3, 5, 8, 12],
        "fixed": dict(V0=70, lam=25),
        "label": r"$\mu$",
    },
}

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

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_softwall"))


def make_pot_params(V0, lam, mu):
    return dict(V0=V0, lam=lam, mu=mu, a=2.0, b=1.2)


def main():
    M = N_ang * len(positions)
    print(f"Study 2.4: Soft-wall params, M={M}")

    all_results = {}

    for scan_name, scan_cfg in SCANS.items():
        print(f"\n{'═'*60}")
        print(f"Scanning: {scan_name}")

        scan_results = {}

        for val in scan_cfg["values"]:
            fixed = scan_cfg["fixed"].copy()
            if scan_name == "V0":
                fixed["V0"] = val
            elif scan_name == "lambda":
                fixed["lam"] = val
            elif scan_name == "mu":
                fixed["mu"] = val

            V0 = fixed.get("V0", 70)
            lam = fixed.get("lam", 25)
            mu = fixed.get("mu", 5)

            print(f"\n  V0={V0}, λ={lam}, μ={mu}")

            pot_params = make_pot_params(V0, lam, mu)

            result = run_ldos_sweep(
                n=n, L=L, vF=vF, m=m,
                potential_option="soft_ellipse",
                pot_params=pot_params,
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

            peak_idx, peak_E = find_peaks_simple(spec, E_values, 0.03)
            n_peaks = len(peak_idx)

            strongest = np.argmax(spec)
            fwhm = estimate_fwhm(spec, E_values, strongest)
            xe, ye = measure_c2v_error(ldos_avg[strongest])

            scan_results[val] = dict(
                spectrum=spec,
                n_peaks=n_peaks,
                fwhm=float(fwhm),
                x_err=float(xe),
                y_err=float(ye),
                elapsed=result["elapsed"],
            )

            print(f"    {n_peaks} peaks, FWHM={fwhm:.4f}, "
                  f"C₂v=({xe:.5f},{ye:.5f})")

        all_results[scan_name] = scan_results

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (scan_name, scan_cfg) in enumerate(SCANS.items()):
        ax = axes[i]
        scan_results = all_results[scan_name]
        for j, val in enumerate(scan_cfg["values"]):
            if val in scan_results:
                spec = scan_results[val]["spectrum"]
                spec_norm = spec / spec.max()
                ax.plot(E_values, spec_norm + j * 0.3, lw=1.2,
                        label=f"{scan_cfg['label']}={val}")
        ax.set_xlabel("E")
        ax.set_ylabel("LDOS (offset)")
        ax.set_title(f"Scan: {scan_cfg['label']}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "softwall_param_scan.png"), dpi=150)
    plt.close(fig)
    print(f"  → softwall_param_scan.png")

    # FWHM and peak count vs each parameter
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (scan_name, scan_cfg) in enumerate(SCANS.items()):
        scan_results = all_results[scan_name]
        vals = [v for v in scan_cfg["values"] if v in scan_results]
        fwhms = [scan_results[v]["fwhm"] for v in vals]
        n_peaks_arr = [scan_results[v]["n_peaks"] for v in vals]

        axes[0, i].plot(vals, fwhms, "ko-", lw=2, ms=8)
        axes[0, i].set_xlabel(scan_cfg["label"])
        axes[0, i].set_ylabel("FWHM")
        axes[0, i].set_title(f"FWHM vs {scan_cfg['label']}")
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(vals, n_peaks_arr, "bo-", lw=2, ms=8)
        axes[1, i].set_xlabel(scan_cfg["label"])
        axes[1, i].set_ylabel("N peaks")
        axes[1, i].set_title(f"Peak count vs {scan_cfg['label']}")
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "softwall_metrics.png"), dpi=150)
    plt.close(fig)
    print(f"  → softwall_metrics.png")

    # Metrics
    metrics = {}
    for scan_name, scan_results in all_results.items():
        metrics[scan_name] = {}
        for val, r in scan_results.items():
            metrics[scan_name][str(val)] = dict(
                n_peaks=r["n_peaks"],
                fwhm=r["fwhm"],
                x_err=r["x_err"],
                y_err=r["y_err"],
            )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Study 2.4 Summary:")
    for scan_name, scan_results in all_results.items():
        print(f"  {scan_name}:")
        for val, r in scan_results.items():
            print(f"    {val}: {r['n_peaks']} peaks, FWHM={r['fwhm']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
