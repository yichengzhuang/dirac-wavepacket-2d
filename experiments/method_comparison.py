#!/usr/bin/env python3
"""
Phase C1: Systematic comparison of ALL mode extraction methods.

For m=0 and m=1, compare:
  - Time filter (baseline)
  - FDM (validated)
  - DMD (new, best performer)
  - POD+MP (hybrid)

Metrics: frequency accuracy, spatial overlap, C2v error, orthogonality.
Output: summary table + comparison figures.
"""
import sys, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (make_grid, gaussian_wavepacket, build_potential,
    radial_CAP, build_analysis_mask, build_split_operator, propagate)
from spectral import extract_resonances, time_filter_mode
from scripts.dmd_extract import exact_dmd, spinor_overlap, c2v_err
from scripts.fdm_debug import build_probes_fixed, build_BA, solve_fdm, reconstruct_modes

OUT = os.path.join(_root, "figures", "dmd")
os.makedirs(OUT, exist_ok=True)

n=256; L=10.0; vF=1.0; dt=0.01; n_steps=15000; save_every=10
pot_params=dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
sigma0=0.6; k_mag=3.0; Gamma_cut=8.0


def run_prop(xp, mass, kx0, ky0, X, Y, KX, KY, K, dx, X_np, Y_np):
    V = build_potential("harmonic_ellipse", X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask("harmonic_ellipse", X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, mass, V, W, KX, KY, K, xp)
    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=0, y0=0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
    result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
        max_record=n_steps//save_every+2, measure_transmission=False)
    return result, xp.to_numpy(chi_R)


def run_fdm(snaps, times, chi_cpu, chi_flat, E_mp, dx2, dt_snap):
    """Run FDM with MP-guided probes, eps=1e-2."""
    E_probes = E_mp.copy()
    if len(E_probes) < 5:
        E_probes = np.sort(np.concatenate([
            E_probes, np.linspace(E_mp[0]-0.5, E_mp[-1]+0.5, 5-len(E_probes))]))
    probes, shifted = build_probes_fixed(snaps, times, E_probes, chi_flat)
    B, A = build_BA(probes, shifted, dx2)
    res = solve_fdm(B, A, dt_snap, eps_svd=1e-2, Gamma_max=Gamma_cut)
    modes = []
    if res["vecs"].shape[1] > 0:
        n_use = min(10, res["vecs"].shape[1])
        order = np.argsort(res["Gamma"])[:n_use]
        res["E"] = res["E"][order]
        res["Gamma"] = res["Gamma"][order]
        res["vecs"] = res["vecs"][:, order]
        modes = reconstruct_modes(res["vecs"], probes, n, chi_cpu, dx2)
    return res["E"], res["Gamma"], modes


def compare_methods(mass, label, xp, X, Y, KX, KY, K, dx, X_np, Y_np):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    dx2 = (L/n)**2
    dt_snap = dt * save_every
    kx0 = k_mag if mass > 0 else k_mag/np.sqrt(2)
    ky0 = 0.0 if mass > 0 else k_mag/np.sqrt(2)

    result, chi_cpu = run_prop(xp, mass, kx0, ky0, X, Y, KX, KY, K, dx, X_np, Y_np)
    snaps = result["snapshots"]; times = result["snap_times"]
    chi_flat = chi_cpu.ravel()

    # Scalar MP reference
    spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
        Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)
    E_lo = mass + 1.5 if mass > 0 else 2.0
    good = (spec["E"] > E_lo) & (spec["E"] < 6) & (spec["Gamma"] < 2)
    E_mp = spec["E"][good]; G_mp = spec["Gamma"][good]; omega_mp = spec["omega"][good]
    n_mp = min(5, len(E_mp))
    E_mp = E_mp[:n_mp]; G_mp = G_mp[:n_mp]; omega_mp = omega_mp[:n_mp]

    print(f"\nMP reference ({n_mp} poles):")
    for j in range(n_mp):
        print(f"  E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

    # Time filter modes (reference for overlap)
    tf_modes = [time_filter_mode(snaps, times, omega_mp[j], chi_cpu) for j in range(n_mp)]

    # ── Run each method ──
    methods = {}

    # 1. Time filter
    print("\nTime filter...")
    tf_ovs = [1.0] * n_mp  # overlap with self = 1
    tf_c2v = []
    for j in range(n_mp):
        rho = np.abs(tf_modes[j][0])**2 + np.abs(tf_modes[j][1])**2
        xe, ye = c2v_err(rho)
        tf_c2v.append((xe + ye) / 2)
    methods["Time filter"] = {"overlaps": tf_ovs, "c2v": tf_c2v, "E": E_mp, "Gamma": G_mp}

    # 2. FDM
    print("FDM...")
    try:
        E_fdm, G_fdm, fdm_modes = run_fdm(snaps, times, chi_cpu, chi_flat, E_mp, dx2, dt_snap)
        fdm_ovs = []
        fdm_c2v = []
        for i in range(n_mp):
            best_ov = 0.0
            best_c2v = 1.0
            for j in range(min(len(fdm_modes), 10)):
                if abs(E_fdm[j] - E_mp[i]) < 0.5:
                    ov = spinor_overlap(fdm_modes[j], tf_modes[i], chi_cpu, dx2)
                    if ov > best_ov:
                        best_ov = ov
                        rho = np.abs(fdm_modes[j][0])**2 + np.abs(fdm_modes[j][1])**2
                        xe, ye = c2v_err(rho)
                        best_c2v = (xe + ye) / 2
            fdm_ovs.append(best_ov)
            fdm_c2v.append(best_c2v)
        methods["FDM"] = {"overlaps": fdm_ovs, "c2v": fdm_c2v, "E": E_fdm, "Gamma": G_fdm}
    except Exception as e:
        print(f"  FDM failed: {e}")

    # 3. DMD (rank=30 for m=0, rank=10 for m=1)
    print("DMD...")
    rank = 10 if mass > 0 else 30
    n_eff = 150
    dmd = exact_dmd(snaps, times, chi_cpu, rank=rank, n_eff=n_eff)
    dmd_ovs = []
    dmd_c2v = []
    for i in range(n_mp):
        best_ov = 0.0
        best_c2v = 1.0
        for j in range(min(len(dmd["modes"]), 15)):
            if abs(dmd["E"][j] - E_mp[i]) < 0.5:
                ov = spinor_overlap(dmd["modes"][j], tf_modes[i], chi_cpu, dx2)
                if ov > best_ov:
                    best_ov = ov
                    rho = np.abs(dmd["modes"][j][0])**2 + np.abs(dmd["modes"][j][1])**2
                    xe, ye = c2v_err(rho)
                    best_c2v = (xe + ye) / 2
        dmd_ovs.append(best_ov)
        dmd_c2v.append(best_c2v)
    methods[f"DMD (r={rank})"] = {"overlaps": dmd_ovs, "c2v": dmd_c2v,
                                   "E": dmd["E"], "Gamma": dmd["Gamma"]}

    # 4. DMD rank=40 (m=0 only)
    if mass == 0:
        print("DMD rank=40...")
        dmd40 = exact_dmd(snaps, times, chi_cpu, rank=40, n_eff=150)
        dmd40_ovs = []
        dmd40_c2v = []
        for i in range(n_mp):
            best_ov = 0.0
            best_c2v = 1.0
            for j in range(min(len(dmd40["modes"]), 15)):
                if abs(dmd40["E"][j] - E_mp[i]) < 0.5:
                    ov = spinor_overlap(dmd40["modes"][j], tf_modes[i], chi_cpu, dx2)
                    if ov > best_ov:
                        best_ov = ov
                        rho = np.abs(dmd40["modes"][j][0])**2 + np.abs(dmd40["modes"][j][1])**2
                        xe, ye = c2v_err(rho)
                        best_c2v = (xe + ye) / 2
            dmd40_ovs.append(best_ov)
            dmd40_c2v.append(best_c2v)
        methods["DMD (r=40)"] = {"overlaps": dmd40_ovs, "c2v": dmd40_c2v,
                                  "E": dmd40["E"], "Gamma": dmd40["Gamma"]}

    # ── Print summary table ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*70}")
    print(f"\n{'Method':<16} {'avg overlap':>12} {'best overlap':>13} {'avg C₂v err':>12}")
    print("-" * 55)
    for name, data in methods.items():
        avg_ov = np.mean(data["overlaps"])
        best_ov = max(data["overlaps"])
        avg_c2v = np.mean(data["c2v"])
        print(f"{name:<16} {avg_ov:12.4f} {best_ov:13.4f} {avg_c2v:12.4f}")

    print(f"\nPer-mode overlaps:")
    print(f"{'Mode':<6} {'E':>8} {'Γ':>8}", end="")
    for name in methods:
        print(f" {name:>12}", end="")
    print()
    for i in range(n_mp):
        print(f"{i+1:<6} {E_mp[i]:8.4f} {G_mp[i]:8.4f}", end="")
        for name, data in methods.items():
            print(f" {data['overlaps'][i]:12.4f}", end="")
        print()

    # ── Figure ──
    method_names = list(methods.keys())
    n_methods = len(method_names)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: overlap per mode
    x = np.arange(n_mp)
    width = 0.8 / n_methods
    for im, name in enumerate(method_names):
        ax1.bar(x + im*width - 0.4 + width/2, methods[name]["overlaps"],
                width, label=name, color=colors[im % len(colors)])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"E={E_mp[i]:.2f}\nΓ={G_mp[i]:.2f}" for i in range(n_mp)], fontsize=7)
    ax1.set_ylabel("Spatial overlap with TF ref")
    ax1.set_title(f"Mode extraction quality ({label})")
    ax1.legend(fontsize=8); ax1.set_ylim(0, 1.1); ax1.grid(alpha=0.2)

    # Bar chart: C2v error
    for im, name in enumerate(method_names):
        ax2.bar(x + im*width - 0.4 + width/2, methods[name]["c2v"],
                width, label=name, color=colors[im % len(colors)])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"E={E_mp[i]:.2f}" for i in range(n_mp)], fontsize=7)
    ax2.set_ylabel("C₂v error (avg x+y)")
    ax2.set_title(f"Symmetry quality ({label})")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    plt.tight_layout()
    tag = "m1" if mass > 0 else "m0"
    out = os.path.join(OUT, f"C1_comparison_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nFigure → {out}")

    return methods


def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    m1_results = compare_methods(1.0, "m=1.0 (control)", xp, X, Y, KX, KY, K, dx, X_np, Y_np)
    m0_results = compare_methods(0.0, "m=0.0 (massless)", xp, X, Y, KX, KY, K, dx, X_np, Y_np)

    print(f"\nTotal: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
