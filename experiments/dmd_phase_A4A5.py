#!/usr/bin/env python3
"""
DMD Phase A4 + A5:

A4: Fine rank scan (25-45) + n_eff study for m=0
    - Where exactly does rank=50 collapse?
    - Does limiting n_eff (effective time window) help?

A5: Multi-propagation DMD
    - Concatenate snapshots from M propagations (different k directions)
    - Does more spatial diversity improve mode extraction?
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
from spectral import extract_resonances, time_filter_mode
from scripts.dmd_extract import exact_dmd, spinor_overlap, c2v_err

OUT = os.path.join(_root, "figures", "dmd")
os.makedirs(OUT, exist_ok=True)

n=256; L=10.0; vF=1.0; dt=0.01; n_steps=15000; save_every=10
POTENTIAL="harmonic_ellipse"; pot_params=dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
sigma0=0.6; k_mag=3.0; Gamma_cut=8.0


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


def compute_overlaps(dmd_res, E_mp, tf_modes, chi_cpu, dx2):
    """Compute per-mode overlap between DMD and TF reference."""
    overlaps = []
    for i in range(len(E_mp)):
        best_ov = 0.0
        for j in range(min(len(dmd_res['modes']), 15)):
            if abs(dmd_res['E'][j] - E_mp[i]) < 0.5:
                ov = spinor_overlap(dmd_res['modes'][j], tf_modes[i], chi_cpu, dx2)
                best_ov = max(best_ov, ov)
        overlaps.append(best_ov)
    return overlaps


def multi_prop_dmd(all_snapshots, all_snap_times, chi_cpu, rank, n_eff_per=None):
    """
    Multi-propagation DMD: concatenate snapshots from multiple propagations.

    Each propagation contributes N_eff columns to the snapshot matrix.
    Different propagations excite different modes → more spatial diversity.
    """
    ny, nx = all_snapshots[0].shape[2], all_snapshots[0].shape[3]
    D = 2 * ny * nx
    dt_snap = all_snap_times[0][1] - all_snap_times[0][0]

    # Concatenate snapshots
    cols = []
    for snaps, times in zip(all_snapshots, all_snap_times):
        Ns = len(times)
        if n_eff_per is not None:
            Ns = min(Ns, n_eff_per)
        X_part = np.zeros((D, Ns), dtype=complex)
        for k in range(Ns):
            X_part[:ny*nx, k] = (snaps[k, 0] * chi_cpu).ravel()
            X_part[ny*nx:, k] = (snaps[k, 1] * chi_cpu).ravel()
        cols.append(X_part)

    # Stack all propagations side by side
    X_all = np.hstack(cols)  # (D, total_cols)
    total_cols = X_all.shape[1]

    # Build shift matrices: within each propagation block, shift by 1
    X0_parts = []
    X1_parts = []
    for X_part in cols:
        X0_parts.append(X_part[:, :-1])
        X1_parts.append(X_part[:, 1:])
    X0 = np.hstack(X0_parts)
    X1 = np.hstack(X1_parts)

    # SVD
    U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    r = min(rank, len(S))
    U_r = U[:, :r]; S_r = S[:r]; Vh_r = Vh[:r, :]

    # Reduced dynamics
    A_tilde = (U_r.conj().T @ X1) @ (Vh_r.conj().T * (1.0/S_r)[None, :])
    lam, w = np.linalg.eig(A_tilde)

    with np.errstate(invalid='ignore', divide='ignore'):
        omega = 1j * np.log(lam + 0j) / dt_snap
    E_all = np.real(omega)
    Gamma_all = -2.0 * np.imag(omega)

    ok = (np.isfinite(E_all) & np.isfinite(Gamma_all) &
          (Gamma_all > 0) & (Gamma_all < Gamma_cut) &
          (np.abs(lam) < 1.0) & (np.abs(lam) > 0.01))

    # Exact DMD modes
    Phi = X1 @ (Vh_r.conj().T * (1.0/S_r)[None, :]) @ w

    E_phys = E_all[ok]; Gamma_phys = Gamma_all[ok]; omega_phys = omega[ok]
    order = np.argsort(Gamma_phys)
    E_phys = E_phys[order]; Gamma_phys = Gamma_phys[order]; omega_phys = omega_phys[order]

    modes = []
    for idx in np.where(ok)[0][order]:
        phi_flat = Phi[:, idx]
        phi = np.zeros((2, ny, nx), dtype=complex)
        phi[0] = phi_flat[:ny*nx].reshape(ny, nx)
        phi[1] = phi_flat[ny*nx:].reshape(ny, nx)
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        norm = np.sqrt(np.sum(rho * chi_cpu) * (L/n)**2)
        if norm > 1e-30:
            phi /= norm
        modes.append(phi)

    return dict(E=E_phys, Gamma=Gamma_phys, omega=omega_phys,
                modes=modes, svd_spectrum=S, rank_used=r,
                total_cols=total_cols)


def main():
    t_total = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx2 = (L/n)**2
    dt_snap = dt * save_every

    # ── Single propagation for A4 ──
    kx0 = k_mag/np.sqrt(2); ky0 = k_mag/np.sqrt(2)
    print("Propagation 1 (45°) for A4...")
    result1, chi_cpu = run_prop(xp, 0.0, kx0, ky0, X, Y, KX, KY, K, dx, X_np, Y_np)
    snaps1 = result1["snapshots"]
    times1 = result1["snap_times"]

    # MP reference
    spec = extract_resonances(result1["C_R"], result1["corr_times"], dt_snap,
                              Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)
    good = (spec["E"] > 2) & (spec["E"] < 6) & (spec["Gamma"] < 2)
    E_mp = spec["E"][good]; G_mp = spec["Gamma"][good]; omega_mp = spec["omega"][good]
    n_mp = min(8, len(E_mp))
    E_mp = E_mp[:n_mp]; G_mp = G_mp[:n_mp]; omega_mp = omega_mp[:n_mp]

    tf_modes = [time_filter_mode(snaps1, times1, omega_mp[j], chi_cpu)
                for j in range(n_mp)]

    print(f"MP reference: {n_mp} poles")

    # ═══════════════════════════════════════════
    # A4: Fine rank scan + n_eff study
    # ═══════════════════════════════════════════
    print("\n" + "="*60)
    print("  A4: Fine rank scan + n_eff study")
    print("="*60)

    # A4a: Fine rank scan (rank 5 to 60)
    ranks = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60]
    rank_results = []

    print(f"\n{'rank':>5} {'poles':>6} {'avg_ov':>8} "
          f"{'ov_E3.97':>9} {'ov_E4.45':>9} {'ov_E3.52':>9}")

    for r in ranks:
        n_eff_use = min(len(times1), max(60, 3*r))
        dmd = exact_dmd(snaps1, times1, chi_cpu, rank=r, n_eff=n_eff_use)
        ovs = compute_overlaps(dmd, E_mp, tf_modes, chi_cpu, dx2)
        avg = np.mean(ovs)

        # Find overlaps for specific modes
        ov_397 = ovs[2] if n_mp > 2 else 0  # E≈3.97
        ov_445 = ovs[3] if n_mp > 3 else 0  # E≈4.45
        ov_352 = ovs[0] if n_mp > 0 else 0  # E≈3.52

        print(f"{r:5d} {len(dmd['E']):6d} {avg:8.4f} "
              f"{ov_397:9.4f} {ov_445:9.4f} {ov_352:9.4f}")
        rank_results.append((r, len(dmd['E']), avg, ovs.copy()))

    # A4b: n_eff study at best rank (30)
    print(f"\nn_eff study at rank=30:")
    print(f"{'n_eff':>6} {'avg_ov':>8} {'best_ov':>8}")

    neff_values = [20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    neff_results = []
    for ne in neff_values:
        if ne > len(times1):
            continue
        dmd = exact_dmd(snaps1, times1, chi_cpu, rank=30, n_eff=ne)
        ovs = compute_overlaps(dmd, E_mp, tf_modes, chi_cpu, dx2)
        avg = np.mean(ovs)
        print(f"{ne:6d} {avg:8.4f} {max(ovs):8.4f}")
        neff_results.append((ne, avg, max(ovs)))

    # A4 figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Overlap vs rank
    ax = axes[0]
    r_vals = [rr[0] for rr in rank_results]
    avg_vals = [rr[2] for rr in rank_results]
    ax.plot(r_vals, avg_vals, 'bo-', ms=5, label='avg overlap')
    # Per-mode
    for i_mode, label in [(0, 'E≈3.52'), (2, 'E≈3.97'), (3, 'E≈4.45')]:
        if i_mode < n_mp:
            ax.plot(r_vals, [rr[3][i_mode] for rr in rank_results],
                    's--', ms=3, label=label)
    ax.set_xlabel("rank r"); ax.set_ylabel("overlap")
    ax.set_title("A4a: Overlap vs rank (m=0)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)

    # Overlap vs n_eff
    ax = axes[1]
    ne_vals = [rr[0] for rr in neff_results]
    ax.plot(ne_vals, [rr[1] for rr in neff_results], 'ro-', ms=5, label='avg')
    ax.plot(ne_vals, [rr[2] for rr in neff_results], 'gs-', ms=5, label='best')
    ax.set_xlabel("n_eff"); ax.set_ylabel("overlap")
    ax.set_title("A4b: Overlap vs n_eff (rank=30)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)

    # SVD spectrum
    ax = axes[2]
    dmd_ref = exact_dmd(snaps1, times1, chi_cpu, rank=30, n_eff=100)
    S = dmd_ref['svd_spectrum']
    ax.semilogy(S/S[0], 'ko-', ms=3)
    ax.axhline(1e-3, ls='--', color='red', label='1e-3')
    ax.set_xlabel("index"); ax.set_ylabel("σ/σ_max")
    ax.set_title("SVD spectrum (n_eff=100)"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "A4_rank_neff_scan.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nA4 figure → {OUT}/A4_rank_neff_scan.png")

    # ═══════════════════════════════════════════
    # A5: Multi-propagation DMD
    # ═══════════════════════════════════════════
    print("\n" + "="*60)
    print("  A5: Multi-propagation DMD")
    print("="*60)

    # Run 3 more propagations at different angles
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    all_snaps = [snaps1]
    all_times = [times1]

    for ia, angle in enumerate(angles):
        kx = k_mag * np.cos(angle)
        ky = k_mag * np.sin(angle)
        print(f"Propagation {ia+2}/{len(angles)+1}: k=({kx:+.2f},{ky:+.2f})")
        res, _ = run_prop(xp, 0.0, kx, ky, X, Y, KX, KY, K, dx, X_np, Y_np)
        all_snaps.append(res["snapshots"])
        all_times.append(res["snap_times"])
        del res

    # Test with increasing number of propagations
    print(f"\nMulti-prop DMD (rank=30, n_eff_per=50):")
    print(f"{'M':>3} {'total_cols':>10} {'poles':>6} {'avg_ov':>8} {'best_ov':>8}  per-mode")

    multi_results = []
    for M in [1, 2, 3, 5]:
        dmd_m = multi_prop_dmd(all_snaps[:M], all_times[:M], chi_cpu,
                                rank=30, n_eff_per=50)
        ovs = compute_overlaps(dmd_m, E_mp, tf_modes, chi_cpu, dx2)
        avg = np.mean(ovs)
        print(f"{M:3d} {dmd_m['total_cols']:10d} {len(dmd_m['E']):6d} "
              f"{avg:8.4f} {max(ovs):8.4f}  "
              f"[{', '.join(f'{o:.3f}' for o in ovs)}]")
        multi_results.append((M, dmd_m, ovs))

    # Also test with larger rank for multi-prop
    print(f"\nMulti-prop M=5, rank scan:")
    for r in [10, 20, 30, 40, 50]:
        dmd_m = multi_prop_dmd(all_snaps[:5], all_times[:5], chi_cpu,
                                rank=r, n_eff_per=50)
        ovs = compute_overlaps(dmd_m, E_mp, tf_modes, chi_cpu, dx2)
        print(f"  rank={r:2d}: avg={np.mean(ovs):.4f}  best={max(ovs):.4f}  "
              f"[{', '.join(f'{o:.3f}' for o in ovs)}]")

    # A5 figure: mode comparison for best multi-prop
    best_M = 5
    dmd_best = multi_prop_dmd(all_snaps[:best_M], all_times[:best_M], chi_cpu,
                               rank=30, n_eff_per=50)
    extent = [-L/2, L/2, -L/2, L/2]
    n_show = min(6, len(dmd_best['modes']))

    if n_show > 0:
        fig, axes = plt.subplots(2, max(n_show, 1), figsize=(4*max(n_show,1), 8))
        if n_show == 1:
            axes = axes[:, np.newaxis]
        for j in range(n_show):
            phi = dmd_best['modes'][j]
            rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
            xe, ye = c2v_err(rho)

            ax = axes[0, j]
            ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                      vmin=0, vmax=rho.max())
            ax.set_title(f"DMD E={dmd_best['E'][j]:.3f}\nΓ={dmd_best['Gamma'][j]:.3f}\n"
                         f"x={xe:.3f} y={ye:.3f}", fontsize=8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

            # TF reference
            dists = np.abs(E_mp - dmd_best['E'][j])
            if len(dists) > 0:
                i_tf = np.argmin(dists)
                rho_tf = np.abs(tf_modes[i_tf][0])**2 + np.abs(tf_modes[i_tf][1])**2
                ov = spinor_overlap(phi, tf_modes[i_tf], chi_cpu, dx2)
                ax = axes[1, j]
                ax.imshow(rho_tf, extent=extent, origin="lower", cmap="inferno",
                          vmin=0, vmax=rho_tf.max())
                ax.set_title(f"TF E={E_mp[i_tf]:.3f}\noverlap={ov:.4f}", fontsize=8)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

        axes[0, 0].set_ylabel("DMD (M=5)", fontsize=10)
        axes[1, 0].set_ylabel("Time filter", fontsize=10)
        fig.suptitle(f"Multi-prop DMD (M={best_M}, rank=30) vs Time filter — m=0",
                     fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "A5_multiprop_modes.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nA5 figure → {OUT}/A5_multiprop_modes.png")

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
