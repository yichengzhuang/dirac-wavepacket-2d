#!/usr/bin/env python3
"""
Sweep CAP parameters (cap_radius, cap_strength, cap_power)
for the L=10 harmonic_ellipse setup.

Fixes propagation time at T=100 (converged) and varies one CAP param at a time.
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances

# ── shared physics (L=10 harmonic_ellipse) ──
BASE = dict(
    n=512, L=10.0,
    POTENTIAL_OPTION="harmonic_ellipse",
    pot_params=dict(k_harm=3.0, a=2.0, b=1.2),
    cap_radius=3.5, cap_strength=10.0, cap_power=2,
    mask_safety_factor=0.95, mask_margin=1.30,
    x0=0.0, y0=0.0, kx0=-1.0, ky0=0.0,
    sigma0=0.6, theta0=None, band0="conduction",
)

vF = 1.0
m  = 0.0
dt = 0.01
T_total = 100        # converged for L=10
n_steps = int(T_total / dt)
decim = 10
Gamma_cut       = 8.0
pencil_fraction = 4
L_pencil_max    = 400
n_modes         = 3


def run_one(cfg, xp):
    """Run a single propagation + spectral extraction, return (E, Gamma, a, P_end)."""
    n = cfg["n"]; L = cfg["L"]

    x_np = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    V = build_potential(cfg["POTENTIAL_OPTION"], X_np, Y_np, cfg["pot_params"], xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cfg["cap_radius"],
                   cap_strength=cfg["cap_strength"], cap_power=cfg["cap_power"], xp=xp)
    chi_R, _ = build_analysis_mask(
        cfg["POTENTIAL_OPTION"], X_np, Y_np, cfg["pot_params"],
        cap_radius=cfg["cap_radius"], safety_factor=cfg["mask_safety_factor"],
        mask_margin=cfg["mask_margin"], xp=xp,
    )
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)

    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=cfg["x0"], y0=cfg["y0"],
        kx0=cfg["kx0"], ky0=cfg["ky0"],
        sigma=cfg["sigma0"], theta=cfg["theta0"],
        band=cfg["band0"], dx=dx, xp=xp,
    )

    result = propagate(
        psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt,
        save_every=max(n_steps, 1),
        corr_every=decim,
        max_record=None,
        measure_transmission=True,
        transmission_every=max(n_steps // 10, 1),
    )

    C_R = result["C_R"]
    corr_times = result["corr_times"]
    dt_corr = dt * decim

    spec = extract_resonances(
        C_R, corr_times, dt_corr,
        Gamma_cut=Gamma_cut,
        pencil_fraction=pencil_fraction,
        L_pencil_max=L_pencil_max,
    )

    P_end = result["P_inside"][-1] if "P_inside" in result else float('nan')
    return spec["E"], spec["Gamma"], spec["a"], P_end


def sweep_param(param_name, param_values, xp):
    """Sweep one CAP parameter, return list of result dicts."""
    print(f"\n{'='*60}")
    print(f"  Sweeping: {param_name}")
    print(f"{'='*60}\n")

    # header
    print(f"{'value':>10s}  ", end="")
    for j in range(n_modes):
        print(f"{'E_'+str(j+1):>10s}  {'Γ_'+str(j+1):>10s}  ", end="")
    print(f"{'P(T)':>8s}  {'wall_s':>7s}")
    print("-" * 100)

    results = []
    for val in param_values:
        cfg = dict(BASE)
        cfg[param_name] = val
        t0 = time.perf_counter()
        E, Gamma, a, P_end = run_one(cfg, xp)
        wall = time.perf_counter() - t0

        results.append({"val": val, "E": E.copy(), "Gamma": Gamma.copy(),
                        "a": a.copy(), "P_end": P_end, "wall": wall})

        print(f"{val:10.3f}  ", end="")
        for j in range(n_modes):
            if j < len(E):
                print(f"{E[j]:10.6f}  {Gamma[j]:10.6f}  ", end="")
            else:
                print(f"{'---':>10s}  {'---':>10s}  ", end="")
        print(f"{P_end:8.4f}  {wall:7.1f}")

    return results


def plot_sweep(param_name, param_label, results, out_name):
    """Plot E_n and Gamma_n vs parameter value."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    vals = np.array([r["val"] for r in results])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["mode 1", "mode 2", "mode 3"]

    for j in range(n_modes):
        Ej = np.array([r["E"][j] if j < len(r["E"]) else np.nan for r in results])
        Gj = np.array([r["Gamma"][j] if j < len(r["Gamma"]) else np.nan for r in results])
        axes[0].plot(vals, Ej, "o-", color=colors[j], label=labels[j], markersize=5)
        axes[1].plot(vals, Gj, "o-", color=colors[j], label=labels[j], markersize=5)

    # reference lines from default
    ref_idx = None
    default_val = BASE[param_name]
    for i, r in enumerate(results):
        if abs(r["val"] - default_val) < 1e-10:
            ref_idx = i
            break

    if ref_idx is not None:
        ref = results[ref_idx]
        for j in range(min(n_modes, len(ref["E"]))):
            axes[0].axhline(ref["E"][j], color=colors[j], ls="--", alpha=0.3)
            axes[1].axhline(ref["Gamma"][j], color=colors[j], ls="--", alpha=0.3)
        axes[0].axvline(default_val, color="gray", ls=":", alpha=0.5, label="default")

    axes[0].set_ylabel("Energy  $E_n$", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].set_title(f"Sensitivity to {param_label}  (L=10 harmonic_ellipse, T={T_total})",
                      fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Width  $\\Gamma_n$", fontsize=13)
    axes[1].set_xlabel(param_label, fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Results", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


def main():
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    # ── 1. Sweep cap_radius ──
    # Potential well effective radius: ellipse a=2.0, b=1.2
    # Grid half-size: L/2 = 5.0
    # Need cap_radius > well size but < grid half-size
    cap_radius_vals = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    res_radius = sweep_param("cap_radius", cap_radius_vals, xp)
    plot_sweep("cap_radius", "CAP radius  $R_{\\mathrm{cap}}$",
               res_radius, "sweep_CAP_radius.png")

    # ── 2. Sweep cap_strength ──
    # Too weak → reflections; too strong → also reflections
    cap_strength_vals = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    res_strength = sweep_param("cap_strength", cap_strength_vals, xp)
    plot_sweep("cap_strength", "CAP strength  $W_0$",
               res_strength, "sweep_CAP_strength.png")

    # ── 3. Sweep cap_power ──
    # Controls smoothness of CAP onset
    cap_power_vals = [1, 2, 3, 4]
    res_power = sweep_param("cap_power", cap_power_vals, xp)
    plot_sweep("cap_power", "CAP power  $p$",
               res_power, "sweep_CAP_power.png")

    print("\nAll sweeps done.")


if __name__ == "__main__":
    main()
