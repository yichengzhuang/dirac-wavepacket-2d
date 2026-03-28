#!/usr/bin/env python3
"""
A5 improved: Independent DMD per propagation, then merge at mode level.

Previous A5 (concatenating snapshots) failed because SVD had to
represent all propagations simultaneously, diluting per-mode information.

Correct approach:
  1. Run DMD independently on each propagation
  2. Cluster poles from all propagations by energy proximity
  3. Average the spatial modes within each cluster
  4. Compare with single-propagation DMD and time filter

Tests m=0 with 4 propagation directions.
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

OUT = os.path.join(_root, "figures", "dmd")
os.makedirs(OUT, exist_ok=True)

n=256; L=10.0; vF=1.0; dt=0.01; n_steps=15000; save_every=10
pot_params=dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
sigma0=0.6; k_mag=3.0; Gamma_cut=8.0


def run_prop(xp, kx0, ky0, X, Y, KX, KY, K, dx, X_np, Y_np, chi_R):
    V = build_potential("harmonic_ellipse", X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, 0.0, V, W, KX, KY, K, xp)
    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=0, y0=0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
    result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
        max_record=n_steps//save_every+2, measure_transmission=False)
    return result


def merge_modes(all_dmd_results, chi_cpu, dx2, E_tol=0.15):
    """
    Merge DMD modes from independent propagations.

    For each cluster of poles with |ΔE| < E_tol:
      - Average E and Γ
      - Average spatial modes (after phase alignment)
    """
    # Collect all modes
    all_modes = []  # (E, Gamma, phi, prop_idx)
    for ip, dmd in enumerate(all_dmd_results):
        for j in range(len(dmd["E"])):
            if j < len(dmd["modes"]):
                all_modes.append((dmd["E"][j], dmd["Gamma"][j], dmd["modes"][j], ip))

    if not all_modes:
        return [], [], []

    all_modes.sort(key=lambda x: x[0])

    # Cluster by energy
    clusters = []
    used = set()
    for i in range(len(all_modes)):
        if i in used:
            continue
        cluster = [all_modes[i]]
        used.add(i)
        for j in range(i+1, len(all_modes)):
            if j in used:
                continue
            if abs(all_modes[j][0] - all_modes[i][0]) < E_tol:
                cluster.append(all_modes[j])
                used.add(j)
            elif all_modes[j][0] - all_modes[i][0] > E_tol:
                break
        clusters.append(cluster)

    # Merge each cluster
    merged_E = []
    merged_G = []
    merged_modes = []

    for cluster in clusters:
        E_avg = np.mean([c[0] for c in cluster])
        G_avg = np.mean([c[1] for c in cluster])

        if len(cluster) == 1:
            merged_modes.append(cluster[0][2])
        else:
            # Phase-align modes before averaging
            ref = cluster[0][2]
            aligned = [ref]
            for c in cluster[1:]:
                phi = c[2]
                # Align phase: maximize Re(<ref|phi>)
                ovlp = (np.vdot(ref[0]*chi_cpu, phi[0]) +
                        np.vdot(ref[1]*chi_cpu, phi[1])) * dx2
                phase = np.exp(-1j * np.angle(ovlp))
                aligned.append(phi * phase)

            # Average
            avg_phi = np.mean(aligned, axis=0)
            # Renormalize
            rho = np.abs(avg_phi[0])**2 + np.abs(avg_phi[1])**2
            norm = np.sqrt(np.sum(rho * chi_cpu) * dx2)
            if norm > 1e-30:
                avg_phi /= norm
            merged_modes.append(avg_phi)

        merged_E.append(E_avg)
        merged_G.append(G_avg)

    return np.array(merged_E), np.array(merged_G), merged_modes


def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx2 = (L/n)**2; dt_snap = dt*save_every

    chi_R, _ = build_analysis_mask("harmonic_ellipse", X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)

    # 5 propagation directions (45° baseline + 4 cardinal)
    angles = [np.pi/4, 0, np.pi/2, np.pi, 3*np.pi/2]
    all_dmd = []
    all_snaps = []
    all_times = []

    for ia, angle in enumerate(angles):
        kx0 = k_mag * np.cos(angle)
        ky0 = k_mag * np.sin(angle)
        print(f"\nPropagation {ia+1}/{len(angles)}: k=({kx0:+.2f},{ky0:+.2f})")
        result = run_prop(xp, kx0, ky0, X, Y, KX, KY, K, dx, X_np, Y_np, chi_R)
        all_snaps.append(result["snapshots"])
        all_times.append(result["snap_times"])

        # Independent DMD
        dmd = exact_dmd(result["snapshots"], result["snap_times"], chi_cpu,
                        rank=30, n_eff=150)
        print(f"  DMD: {len(dmd['E'])} poles")
        all_dmd.append(dmd)
        del result

    # MP reference from first propagation
    # (re-extract since we didn't save C_R above — use DMD frequencies instead)
    E_ref = all_dmd[0]["E"]
    G_ref = all_dmd[0]["Gamma"]
    good = (E_ref > 2) & (E_ref < 6) & (G_ref < 2)
    E_mp = E_ref[good][:8]; G_mp = G_ref[good][:8]

    # TF modes from first propagation
    omega_mp = E_mp - 1j*G_mp/2
    tf_modes = [time_filter_mode(all_snaps[0], all_times[0], omega_mp[j], chi_cpu)
                for j in range(len(E_mp))]

    # ── Single DMD (baseline) ──
    print(f"\n--- Single DMD (prop 1 only) ---")
    single = all_dmd[0]
    single_ovs = []
    for i in range(len(E_mp)):
        best = 0.0
        for j in range(min(len(single["modes"]), 15)):
            if abs(single["E"][j] - E_mp[i]) < 0.5:
                best = max(best, spinor_overlap(single["modes"][j], tf_modes[i], chi_cpu, dx2))
        single_ovs.append(best)
    print(f"  avg overlap: {np.mean(single_ovs):.4f}  [{', '.join(f'{o:.3f}' for o in single_ovs)}]")

    # ── Merged DMD (all propagations) ──
    for M_use in [2, 3, 5]:
        print(f"\n--- Merged DMD (M={M_use} independent propagations) ---")
        E_m, G_m, modes_m = merge_modes(all_dmd[:M_use], chi_cpu, dx2)
        print(f"  {len(E_m)} merged clusters")

        merged_ovs = []
        for i in range(len(E_mp)):
            best = 0.0
            for j in range(len(modes_m)):
                if abs(E_m[j] - E_mp[i]) < 0.5:
                    best = max(best, spinor_overlap(modes_m[j], tf_modes[i], chi_cpu, dx2))
            merged_ovs.append(best)
        print(f"  avg overlap: {np.mean(merged_ovs):.4f}  [{', '.join(f'{o:.3f}' for o in merged_ovs)}]")

    # ── Figure: single vs merged modes ──
    E_m5, G_m5, modes_m5 = merge_modes(all_dmd[:5], chi_cpu, dx2)
    extent = [-L/2, L/2, -L/2, L/2]

    def fft_flip(a, ax): return np.roll(np.flip(a, axis=ax), 1, axis=ax)
    def sym_c2v(r): return 0.25*(r+fft_flip(r,1)+fft_flip(r,0)+fft_flip(fft_flip(r,1),0))

    n_show = min(6, len(single["modes"]), len(modes_m5))
    if n_show > 0:
        fig, axes = plt.subplots(3, n_show, figsize=(4.5*n_show, 13))
        if n_show == 1: axes = axes[:, np.newaxis]

        for j in range(n_show):
            # Row 0: single DMD
            rho = np.abs(single["modes"][j][0])**2 + np.abs(single["modes"][j][1])**2
            axes[0,j].imshow(sym_c2v(rho), extent=extent, origin="lower", cmap="inferno",
                             vmin=0, vmax=rho.max())
            axes[0,j].set_title(f"Single DMD\nE={single['E'][j]:.3f}", fontsize=8)
            axes[0,j].set_aspect("equal"); axes[0,j].set_xticks([]); axes[0,j].set_yticks([])

            # Row 1: merged (find nearest E)
            if len(E_m5) > 0:
                k = np.argmin(np.abs(E_m5 - single["E"][j]))
                rho_m = np.abs(modes_m5[k][0])**2 + np.abs(modes_m5[k][1])**2
                ov = spinor_overlap(modes_m5[k], single["modes"][j], chi_cpu, dx2)
                axes[1,j].imshow(sym_c2v(rho_m), extent=extent, origin="lower", cmap="inferno",
                                 vmin=0, vmax=rho_m.max())
                axes[1,j].set_title(f"Merged (M=5)\nE={E_m5[k]:.3f}\nov={ov:.3f}", fontsize=8)
                axes[1,j].set_aspect("equal"); axes[1,j].set_xticks([]); axes[1,j].set_yticks([])

            # Row 2: TF reference
            dists = np.abs(E_mp - single["E"][j])
            if len(dists) > 0:
                i_tf = np.argmin(dists)
                rho_tf = np.abs(tf_modes[i_tf][0])**2 + np.abs(tf_modes[i_tf][1])**2
                axes[2,j].imshow(sym_c2v(rho_tf), extent=extent, origin="lower", cmap="inferno",
                                 vmin=0, vmax=rho_tf.max())
                axes[2,j].set_title(f"TF ref E={E_mp[i_tf]:.3f}", fontsize=8)
                axes[2,j].set_aspect("equal"); axes[2,j].set_xticks([]); axes[2,j].set_yticks([])

        axes[0,0].set_ylabel("Single DMD", fontsize=10)
        axes[1,0].set_ylabel("Merged M=5", fontsize=10)
        axes[2,0].set_ylabel("Time filter", fontsize=10)
        fig.suptitle("A5 improved: Independent DMD + mode-level merge (m=0)", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "A5_independent_merge.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nFigure → {OUT}/A5_independent_merge.png")

    print(f"\nTotal: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
