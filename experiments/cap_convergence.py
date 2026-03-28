#!/usr/bin/env python3
"""
Phase 1: CAP convergence validation.

Sweep cap_strength, cap_radius, cap_power independently.
For each, extract MP poles in E∈[2,6] and check if (E,Γ) are stable.
If Γ changes >5% with CAP parameters, results are artifacts.
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
from spectral import extract_resonances

OUT = os.path.join(_root, "figures", "convergence")
os.makedirs(OUT, exist_ok=True)

# Fixed physics
n=256; L=10.0; vF=1.0; m=0.0
dt=0.01; n_steps=15000; save_every=10
POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
sigma0=0.6; k_mag=3.0
Gamma_cut=8.0; pencil_fraction=4; L_pencil_max=400

# Default CAP
DEFAULT_CAP = dict(cap_radius=3.5, cap_strength=10.0, cap_power=2)

# Sweep ranges
STRENGTH_VALUES = [2, 5, 10, 20, 50]
RADIUS_VALUES   = [2.5, 3.0, 3.5, 4.0, 4.5]
POWER_VALUES    = [1, 2, 3, 4]


def run_and_extract(xp, X, Y, KX, KY, K, dx, X_np, Y_np,
                    cap_radius, cap_strength, cap_power):
    """Run one propagation + MP extraction. Returns top-5 (E, Γ) in [2,6]."""
    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)

    kx0 = k_mag / np.sqrt(2); ky0 = k_mag / np.sqrt(2)
    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=0.0, y0=0.0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)

    result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
        max_record=n_steps//save_every+2, measure_transmission=False)

    spec = extract_resonances(result["C_R"], result["corr_times"],
        dt*save_every, Gamma_cut=Gamma_cut,
        pencil_fraction=pencil_fraction, L_pencil_max=L_pencil_max)

    E, G = spec["E"], spec["Gamma"]
    good = (E > 2.0) & (E < 6.0) & (G > 0) & (G < 2.0)
    E, G = E[good], G[good]
    n_use = min(5, len(E))
    return E[:n_use], G[:n_use]


def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    scans = [
        ("cap_strength", STRENGTH_VALUES, "cap_strength",
         lambda v: dict(cap_radius=3.5, cap_strength=v, cap_power=2)),
        ("cap_radius", RADIUS_VALUES, "cap_radius",
         lambda v: dict(cap_radius=v, cap_strength=10.0, cap_power=2)),
        ("cap_power", POWER_VALUES, "cap_power",
         lambda v: dict(cap_radius=3.5, cap_strength=10.0, cap_power=v)),
    ]

    for scan_name, values, xlabel, param_fn in scans:
        print(f"\n{'='*60}")
        print(f"  Scanning {scan_name}: {values}")
        print(f"{'='*60}")

        all_E = []; all_G = []
        for val in values:
            params = param_fn(val)
            print(f"\n  {scan_name}={val} ...")
            E, G = run_and_extract(xp, X, Y, KX, KY, K, dx, X_np, Y_np, **params)
            all_E.append(E); all_G.append(G)
            for j in range(len(E)):
                print(f"    E={E[j]:.4f}  Γ={G[j]:.4f}")

        # Reference = default value
        def_idx = values.index(DEFAULT_CAP.get(scan_name, values[len(values)//2]))
        E_ref, G_ref = all_E[def_idx], all_G[def_idx]

        # Compute relative deviation from reference
        print(f"\n  Relative Γ deviation from default ({scan_name}={values[def_idx]}):")
        for iv, val in enumerate(values):
            if len(all_G[iv]) > 0 and len(G_ref) > 0:
                # match by nearest E
                devs = []
                for j in range(min(len(all_G[iv]), len(G_ref))):
                    if j < len(G_ref) and G_ref[j] > 0:
                        devs.append(abs(all_G[iv][j] - G_ref[j]) / G_ref[j])
                avg_dev = np.mean(devs) if devs else float('nan')
                print(f"    {scan_name}={val}: avg |ΔΓ/Γ| = {avg_dev:.4f}"
                      f"  {'✓' if avg_dev < 0.05 else '⚠️' if avg_dev < 0.10 else '❌'}")

        # Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(values)))

        for iv, val in enumerate(values):
            if len(all_E[iv]) > 0:
                ax1.scatter([val]*len(all_E[iv]), all_E[iv],
                           c=[colors[iv]], s=40, zorder=3)
                ax2.scatter([val]*len(all_G[iv]), all_G[iv],
                           c=[colors[iv]], s=40, zorder=3)
                for j in range(len(all_E[iv])):
                    ax1.annotate(f"{all_E[iv][j]:.2f}", (val, all_E[iv][j]),
                                fontsize=6, alpha=0.7)

        ax1.axvline(DEFAULT_CAP.get(scan_name, values[len(values)//2]),
                    ls='--', color='red', alpha=0.5, label='default')
        ax2.axvline(DEFAULT_CAP.get(scan_name, values[len(values)//2]),
                    ls='--', color='red', alpha=0.5, label='default')

        ax1.set_xlabel(xlabel); ax1.set_ylabel("E")
        ax1.set_title(f"Energies vs {scan_name}"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.set_xlabel(xlabel); ax2.set_ylabel("Γ")
        ax2.set_title(f"Widths vs {scan_name}"); ax2.legend(); ax2.grid(alpha=0.3)

        fig.suptitle(f"CAP convergence: {scan_name} scan (m=0, harmonic ellipse)",
                     fontsize=12)
        plt.tight_layout()
        out = os.path.join(OUT, f"{scan_name}_scan.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Figure → {out}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.1f} s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
