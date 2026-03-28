#!/usr/bin/env python3
"""
POD + scalar MP hybrid method for spatial mode extraction.

Strategy:
  1. SVD/POD of snapshot matrix → r spatial basis functions u_j(r) + time coefficients c_j(t)
  2. Scalar matrix pencil on each c_j(t) → extract frequencies ω_n for each POD component
  3. Cluster poles across POD components (same physical mode appears in multiple c_j)
  4. Reconstruct spatial modes: φ_n(r) = Σ_j w_{nj} u_j(r)

Why this might be better than DMD:
  - POD separates spatial compression from spectral extraction
  - Scalar MP is well-validated (decades of development)
  - Each c_j(t) is a 1D signal — MP handles it robustly
  - Mode reconstruction uses POD basis which is optimal in L2 sense

Tests:
  - m=1 control (overlap > 0.95)
  - m=0 comparison with DMD
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
from spectral import extract_resonances, time_filter_mode, matrix_pencil
from scripts.dmd_extract import spinor_overlap, c2v_err

OUT = os.path.join(_root, "figures", "dmd")
os.makedirs(OUT, exist_ok=True)

n=256; L=10.0; vF=1.0; dt=0.01; n_steps=15000; save_every=10
POTENTIAL="harmonic_ellipse"; pot_params=dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
sigma0=0.6; k_mag=3.0; Gamma_cut=8.0


def pod_mp_extract(snapshots, snap_times, chi_cpu, rank=20, n_eff=None,
                   pencil_fraction=4, L_pencil_max=200):
    """
    POD + scalar MP hybrid extraction.

    Returns dict with E, Gamma, modes, pod_basis, pod_coeffs
    """
    Ns = len(snap_times)
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    D = 2 * ny * nx
    dt_snap = snap_times[1] - snap_times[0]

    if n_eff is not None:
        Ns = min(Ns, n_eff)

    # Build snapshot matrix (D, Ns) with mask
    X = np.zeros((D, Ns), dtype=complex)
    for k in range(Ns):
        X[:ny*nx, k] = (snapshots[k, 0] * chi_cpu).ravel()
        X[ny*nx:, k] = (snapshots[k, 1] * chi_cpu).ravel()

    # Step 1: SVD/POD
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    r = min(rank, len(S))
    U_r = U[:, :r]     # (D, r) — POD spatial basis
    S_r = S[:r]         # (r,)
    Vh_r = Vh[:r, :]    # (r, Ns) — temporal coefficients (scaled)

    # POD time coefficients: c_j(t_k) = S_j * Vh_r[j, k]
    C_pod = np.diag(S_r) @ Vh_r  # (r, Ns) — each row is a time series

    print(f"  POD: rank={r}, SVD spectrum: {S_r[:5]/S_r[0]}")

    # Step 2: Scalar MP on each POD coefficient
    all_poles = []  # (omega, amplitude, pod_index)
    for j in range(r):
        c_j = C_pod[j, :]  # (Ns,) time series
        try:
            L_pen = min(Ns // pencil_fraction, L_pencil_max)
            z_raw, _ = matrix_pencil(c_j, L_pen)
            omega_raw = (1j / dt_snap) * np.log(z_raw + 0j)
            E_raw = np.real(omega_raw)
            G_raw = -2.0 * np.imag(omega_raw)

            # Filter physical poles
            ok = ((np.abs(z_raw) < 1) & (np.abs(z_raw) > 0.01) &
                  (G_raw > 0) & (G_raw < Gamma_cut) &
                  np.isfinite(E_raw) & np.isfinite(G_raw))

            for idx in np.where(ok)[0]:
                # Amplitude of this pole in this POD component
                all_poles.append((omega_raw[idx], E_raw[idx], G_raw[idx], j))
        except Exception:
            continue

    if not all_poles:
        return dict(E=np.array([]), Gamma=np.array([]), modes=[], pod_basis=U_r)

    # Step 3: Cluster poles across POD components
    # Group poles with |ΔE| < 0.1
    poles_sorted = sorted(all_poles, key=lambda p: p[1])  # sort by E
    clusters = []
    used = set()
    for i, (omega_i, E_i, G_i, pod_i) in enumerate(poles_sorted):
        if i in used:
            continue
        cluster = [(omega_i, E_i, G_i, pod_i)]
        used.add(i)
        for j in range(i+1, len(poles_sorted)):
            if j in used:
                continue
            if abs(poles_sorted[j][1] - E_i) < 0.15:
                cluster.append(poles_sorted[j])
                used.add(j)
            elif poles_sorted[j][1] - E_i > 0.2:
                break
        # Representative: median E, min Gamma
        E_rep = np.median([c[1] for c in cluster])
        G_rep = np.min([c[2] for c in cluster])
        omega_rep = complex(E_rep, -G_rep/2)
        clusters.append((omega_rep, E_rep, G_rep, cluster))

    print(f"  {len(all_poles)} raw poles → {len(clusters)} clusters")

    # Step 4: Reconstruct spatial modes
    # For each cluster, solve for the POD weights that best reconstruct the signal
    modes = []
    E_out = []
    G_out = []

    for omega_rep, E_rep, G_rep, cluster in clusters:
        if E_rep < 0 or E_rep > 10:
            continue

        # Build the ideal time signal for this mode: e^{-i ω t}
        ideal = np.exp(-1j * omega_rep * snap_times[:Ns])  # (Ns,)

        # Project onto POD coefficients: find w such that Σ_j w_j c_j(t) ≈ a * ideal(t)
        # This is a least-squares problem: C_pod.T @ w ≈ ideal (up to amplitude)
        # More precisely: w_j = <c_j, ideal> / ||ideal||²
        inner = C_pod @ ideal.conj()  # (r,) — projection of ideal onto each POD coeff
        w = inner / (np.linalg.norm(ideal)**2)  # (r,) — POD weights

        # Spatial mode: φ(r) = U_r @ w
        phi_flat = U_r @ w  # (D,)
        phi = np.zeros((2, ny, nx), dtype=complex)
        phi[0] = phi_flat[:ny*nx].reshape(ny, nx)
        phi[1] = phi_flat[ny*nx:].reshape(ny, nx)

        # Normalize
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        norm = np.sqrt(np.sum(rho * chi_cpu) * (L/n)**2)
        if norm > 1e-30:
            phi /= norm
            modes.append(phi)
            E_out.append(E_rep)
            G_out.append(G_rep)

    # Sort by Gamma
    order = np.argsort(G_out)
    E_out = np.array(E_out)[order]
    G_out = np.array(G_out)[order]
    modes = [modes[i] for i in order]

    return dict(E=E_out, Gamma=G_out, modes=modes,
                pod_basis=U_r, svd_spectrum=S, rank_used=r)


def run_prop(xp, mass, kx0, ky0, X, Y, KX, KY, K, dx, X_np, Y_np):
    V = build_potential(POTENTIAL, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(POTENTIAL, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, mass, V, W, KX, KY, K, xp)
    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=0.0, y0=0.0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
    result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
        max_record=n_steps//save_every+2, measure_transmission=False)
    return result, xp.to_numpy(chi_R)


def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx2 = (L/n)**2
    dt_snap = dt * save_every
    extent = [-L/2, L/2, -L/2, L/2]

    for mass, label in [(1.0, "m=1"), (0.0, "m=0")]:
        print(f"\n{'='*60}")
        print(f"  POD+MP: {label}")
        print(f"{'='*60}")

        kx0 = k_mag if mass > 0 else k_mag/np.sqrt(2)
        ky0 = 0.0 if mass > 0 else k_mag/np.sqrt(2)

        result, chi_cpu = run_prop(xp, mass, kx0, ky0,
                                    X, Y, KX, KY, K, dx, X_np, Y_np)
        snaps = result["snapshots"]; times = result["snap_times"]

        # Scalar MP reference
        spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
                                  Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)
        good = (spec["E"] > (mass + 1.5)) & (spec["E"] < 6) & (spec["Gamma"] < 2)
        E_mp = spec["E"][good]; G_mp = spec["Gamma"][good]; omega_mp = spec["omega"][good]
        n_mp = min(5 if mass > 0 else 8, len(E_mp))
        E_mp = E_mp[:n_mp]; G_mp = G_mp[:n_mp]; omega_mp = omega_mp[:n_mp]

        print(f"\nMP reference: {n_mp} poles")
        for j in range(n_mp):
            print(f"  E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

        tf_modes = [time_filter_mode(snaps, times, omega_mp[j], chi_cpu)
                    for j in range(n_mp)]

        # POD+MP with rank scan
        print(f"\nPOD+MP rank scan:")
        print(f"{'rank':>5} {'clusters':>9} {'avg_ov':>8} {'best_ov':>8}  per-mode")

        best_avg = -1; best_result = None
        for r in [6, 10, 15, 20, 30, 40]:
            n_eff_use = min(len(times), max(60, 3*r))
            pod = pod_mp_extract(snaps, times, chi_cpu, rank=r, n_eff=n_eff_use)
            if len(pod['E']) == 0:
                print(f"{r:5d}         0")
                continue
            ovs = []
            for i in range(n_mp):
                best_ov = 0.0
                for j in range(min(len(pod['modes']), 15)):
                    if abs(pod['E'][j] - E_mp[i]) < 0.5:
                        ov = spinor_overlap(pod['modes'][j], tf_modes[i], chi_cpu, dx2)
                        best_ov = max(best_ov, ov)
                ovs.append(best_ov)
            avg = np.mean(ovs)
            print(f"{r:5d} {len(pod['E']):9d} {avg:8.4f} {max(ovs):8.4f}  "
                  f"[{', '.join(f'{o:.3f}' for o in ovs)}]")
            if avg > best_avg:
                best_avg = avg; best_result = pod

        # Figure: best POD+MP modes vs TF
        if best_result and len(best_result['modes']) > 0:
            n_show = min(5, len(best_result['modes']), n_mp)
            fig, axes = plt.subplots(2, max(n_show,1), figsize=(4*max(n_show,1), 8))
            if n_show == 1:
                axes = axes[:, np.newaxis]
            for j in range(n_show):
                phi = best_result['modes'][j]
                rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
                ax = axes[0, j]
                ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                          vmin=0, vmax=rho.max())
                ax.set_title(f"POD+MP\nE={best_result['E'][j]:.3f}\nΓ={best_result['Gamma'][j]:.3f}",
                             fontsize=8)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

                dists = np.abs(E_mp - best_result['E'][j])
                if len(dists) > 0:
                    i_tf = np.argmin(dists)
                    rho_tf = np.abs(tf_modes[i_tf][0])**2 + np.abs(tf_modes[i_tf][1])**2
                    ov = spinor_overlap(phi, tf_modes[i_tf], chi_cpu, dx2)
                    ax = axes[1, j]
                    ax.imshow(rho_tf, extent=extent, origin="lower", cmap="inferno",
                              vmin=0, vmax=rho_tf.max())
                    ax.set_title(f"TF E={E_mp[i_tf]:.3f}\nov={ov:.4f}", fontsize=8)
                    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

            axes[0, 0].set_ylabel("POD+MP", fontsize=10)
            axes[1, 0].set_ylabel("Time filter", fontsize=10)
            tag = label.replace("=", "")
            fig.suptitle(f"POD+MP vs Time filter ({label})", fontsize=12)
            plt.tight_layout()
            fig.savefig(os.path.join(OUT, f"B1_pod_mp_{tag}.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nFigure → {OUT}/B1_pod_mp_{tag}.png")

    print(f"\nTotal: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
