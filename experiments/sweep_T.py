#!/usr/bin/env python3
"""
Sweep propagation time T to test sensitivity of spectral extraction.

Runs both L=10 and L=100 parameter sets, saves separate convergence plots.
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

# ── shared ──
vF = 1.0
m  = 0.0
dt = 0.01
decim = 10
Gamma_cut       = 8.0
pencil_fraction = 4
L_pencil_max    = 400
n_modes         = 3

# ── parameter sets ──
CONFIGS = {
    "L=10 harmonic_ellipse": dict(
        n=512, L=10.0,
        POTENTIAL_OPTION="harmonic_ellipse",
        pot_params=dict(k_harm=3.0, a=2.0, b=1.2),
        cap_radius=3.5, cap_strength=10.0, cap_power=2,
        mask_safety_factor=0.95, mask_margin=1.30,
        x0=0.0, y0=0.0, kx0=-1.0, ky0=0.0,
        sigma0=0.6, theta0=None, band0="conduction",
        T_values=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                  50, 60, 80, 100, 150, 200, 300],
        out_name="sweep_T_convergence_L10.png",
    ),
    "L=100 harmonic_circle": dict(
        n=512, L=100.0,
        POTENTIAL_OPTION="harmonic_circle",
        pot_params=dict(k_harm=3.0, r0=20.0),
        cap_radius=35.0, cap_strength=10.0, cap_power=2,
        mask_safety_factor=0.95, mask_margin=1.60,
        x0=0.0, y0=0.0, kx0=1.9, ky0=0.0,
        sigma0=3.0, theta0=None, band0="conduction",
        T_values=[20, 30, 40, 50, 60, 70, 80, 90, 100,
                  120, 150, 200, 300, 400],
        out_name="sweep_T_convergence_L100.png",
    ),
}


def run_sweep(cfg_name, cfg, xp):
    print(f"\n{'='*60}")
    print(f"  {cfg_name}")
    print(f"{'='*60}\n")

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

    results = []

    # header
    print(f"{'T':>6s}  {'n_steps':>7s}  {'samples':>7s}  ", end="")
    for j in range(n_modes):
        print(f"{'E_'+str(j+1):>10s}  {'Γ_'+str(j+1):>10s}  ", end="")
    print(f"{'P(T)':>8s}  {'wall_s':>7s}")
    print("-" * 120)

    for T_total in cfg["T_values"]:
        n_steps = int(T_total / dt)
        t0 = time.perf_counter()

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

        E = spec["E"]; Gamma = spec["Gamma"]; a = spec["a"]
        P_end = result["P_inside"][-1] if "P_inside" in result else float('nan')
        wall = time.perf_counter() - t0

        results.append({"T": T_total, "E": E.copy(), "Gamma": Gamma.copy(),
                        "a": a.copy(), "P_end": P_end, "wall": wall})

        print(f"{T_total:6.0f}  {n_steps:7d}  {len(C_R):7d}  ", end="")
        for j in range(n_modes):
            if j < len(E):
                print(f"{E[j]:10.6f}  {Gamma[j]:10.6f}  ", end="")
            else:
                print(f"{'---':>10s}  {'---':>10s}  ", end="")
        print(f"{P_end:8.4f}  {wall:7.1f}")

    # ── plot ──
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    Ts = np.array([r["T"] for r in results])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["mode 1", "mode 2", "mode 3"]

    for j in range(n_modes):
        Ej = np.array([r["E"][j] if j < len(r["E"]) else np.nan for r in results])
        Gj = np.array([r["Gamma"][j] if j < len(r["Gamma"]) else np.nan for r in results])
        axes[0].plot(Ts, Ej, "o-", color=colors[j], label=labels[j], markersize=4)
        axes[1].plot(Ts, Gj, "o-", color=colors[j], label=labels[j], markersize=4)

    ref = results[-1]
    for j in range(min(n_modes, len(ref["E"]))):
        axes[0].axhline(ref["E"][j], color=colors[j], ls="--", alpha=0.4)
        axes[1].axhline(ref["Gamma"][j], color=colors[j], ls="--", alpha=0.4)

    axes[0].set_ylabel("Energy  $E_n$", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].set_title(f"Convergence vs $T$  ({cfg_name})", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Width  $\\Gamma_n$", fontsize=13)
    axes[1].set_xlabel("Propagation time  $T$", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Results", cfg["out_name"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    return results


def main():
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    # Only run L=100
    run_sweep("L=100 harmonic_circle", CONFIGS["L=100 harmonic_circle"], xp)

    print("\nAll done.")


if __name__ == "__main__":
    main()
