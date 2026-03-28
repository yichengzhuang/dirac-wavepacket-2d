#!/usr/bin/env python3
"""
Dynamic Mode Decomposition (DMD) for spatial eigenmode extraction.

Unlike FDM/Block MP which project onto coarse channels, DMD operates
directly on the full spatial grid — every pixel is a channel.

Algorithm (Exact DMD):
  X0 = [ψ(t_0), ..., ψ(t_{N-2})]     (D × N-1)
  X1 = [ψ(t_1), ..., ψ(t_{N-1})]     (D × N-1)  shifted
  SVD: X0 = U S V†,  truncate to rank r
  Ã = U_r† X1 V_r S_r⁻¹              (r × r)
  Ã w = λ w  →  ω = i ln(λ) / Δt
  Spatial modes: Φ = X1 V_r S_r⁻¹ W   (D × r)

Phases:
  A1: Basic implementation
  A2: m=1 control (overlap > 0.95 with time-filter?)
  A3: m=0 narrow modes
  A4: rank r scan
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances, time_filter_mode

OUT = os.path.join(_root, "figures", "dmd")
os.makedirs(OUT, exist_ok=True)

# Physics
n = 256; L = 10.0; vF = 1.0
dt = 0.01; n_steps = 15000; save_every = 10  # Δt=0.1, fine temporal resolution
POTENTIAL = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5; cap_strength = 10.0; cap_power = 2
sigma0 = 0.6; k_mag = 3.0
Gamma_cut = 8.0


# ╔═══════════════════════════════════════════════════════════════╗
# ║  DMD Core                                                    ║
# ╚═══════════════════════════════════════════════════════════════╝

def exact_dmd(snapshots, snap_times, chi_cpu, rank=None, n_eff=None):
    """
    Exact DMD on masked spinor snapshots.

    Parameters
    ----------
    snapshots : (Ns, 2, ny, nx) real/complex
    snap_times : (Ns,)
    chi_cpu : (ny, nx) mask
    rank : int or None — SVD truncation rank. None = auto (elbow method)
    n_eff : int or None — use only first n_eff snapshots. None = all

    Returns
    -------
    dict with E, Gamma, omega, modes (list of (2,ny,nx)), svd_spectrum, rank_used
    """
    Ns = len(snap_times)
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    dt_snap = snap_times[1] - snap_times[0]

    # Truncate to effective time window
    if n_eff is not None:
        Ns = min(Ns, n_eff)

    # Build snapshot matrix: (D, Ns) where D = 2*ny*nx
    # Apply mask to each snapshot
    D = 2 * ny * nx
    X = np.zeros((D, Ns), dtype=complex)
    for k in range(Ns):
        X[:ny*nx, k] = (snapshots[k, 0] * chi_cpu).ravel()
        X[ny*nx:, k] = (snapshots[k, 1] * chi_cpu).ravel()

    # Shift matrices
    X0 = X[:, :-1]   # (D, Ns-1)
    X1 = X[:, 1:]    # (D, Ns-1)

    # SVD of X0
    U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    # S shape: (min(D, Ns-1),) = (Ns-1,) since D >> Ns

    # Auto rank selection: find elbow in singular values
    if rank is None:
        # Use ratio criterion: keep components where S[i]/S[0] > 1e-3
        rank = max(2, int(np.sum(S / S[0] > 1e-3)))
        rank = min(rank, len(S) // 2)  # don't use more than half

    r = min(rank, len(S))

    U_r = U[:, :r]      # (D, r)
    S_r = S[:r]          # (r,)
    Vh_r = Vh[:r, :]     # (r, Ns-1)

    # Reduced dynamics: Ã = U_r† X1 V_r S_r⁻¹
    A_tilde = (U_r.conj().T @ X1) @ (Vh_r.conj().T * (1.0 / S_r)[None, :])
    # A_tilde shape: (r, r)

    # Eigendecomposition
    lam, w = np.linalg.eig(A_tilde)

    # λ → ω
    with np.errstate(invalid='ignore', divide='ignore'):
        omega = 1j * np.log(lam + 0j) / dt_snap

    E_all = np.real(omega)
    Gamma_all = -2.0 * np.imag(omega)

    # Physical filter
    ok = (np.isfinite(E_all) & np.isfinite(Gamma_all) &
          (Gamma_all > 0) & (Gamma_all < Gamma_cut) &
          (np.abs(lam) < 1.0) & (np.abs(lam) > 0.01))

    # Exact DMD modes: Φ = X1 V_r S_r⁻¹ W
    Phi = X1 @ (Vh_r.conj().T * (1.0 / S_r)[None, :]) @ w  # (D, r)

    # Extract physical modes
    E_phys = E_all[ok]
    Gamma_phys = Gamma_all[ok]
    omega_phys = omega[ok]

    # Sort by Gamma (narrowest first)
    order = np.argsort(Gamma_phys)
    E_phys = E_phys[order]
    Gamma_phys = Gamma_phys[order]
    omega_phys = omega_phys[order]

    # Reconstruct spatial modes
    modes = []
    for idx in np.where(ok)[0][order]:
        phi_flat = Phi[:, idx]
        phi = np.zeros((2, ny, nx), dtype=complex)
        phi[0] = phi_flat[:ny*nx].reshape(ny, nx)
        phi[1] = phi_flat[ny*nx:].reshape(ny, nx)
        # Normalize inside mask
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        norm = np.sqrt(np.sum(rho * chi_cpu) * (L/n)**2)
        if norm > 1e-30:
            phi = phi / norm
        modes.append(phi)

    return dict(
        E=E_phys, Gamma=Gamma_phys, omega=omega_phys,
        modes=modes,
        svd_spectrum=S, rank_used=r,
        lam_all=lam, E_all=E_all, Gamma_all=Gamma_all,
    )


def spinor_overlap(a, b, chi, dx2):
    """Properly normalized overlap."""
    ab = (np.vdot(a[0]*chi, b[0]) + np.vdot(a[1]*chi, b[1])) * dx2
    aa = (np.vdot(a[0]*chi, a[0]) + np.vdot(a[1]*chi, a[1])) * dx2
    bb = (np.vdot(b[0]*chi, b[0]) + np.vdot(b[1]*chi, b[1])) * dx2
    denom = np.real(aa) * np.real(bb)
    return float(np.abs(ab)**2 / denom) if denom > 1e-30 else 0.0


def fft_flip(a, ax):
    return np.roll(np.flip(a, axis=ax), 1, axis=ax)

def c2v_err(rho):
    s = np.sum(rho) + 1e-30
    return (np.sum(np.abs(fft_flip(rho,1)-rho))/s,
            np.sum(np.abs(fft_flip(rho,0)-rho))/s)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Propagation helper                                          ║
# ╚═══════════════════════════════════════════════════════════════╝

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
    chi_cpu = xp.to_numpy(chi_R)
    psi0 = np.stack([xp.to_numpy(psi0_up), xp.to_numpy(psi0_dn)])
    return result, chi_cpu, psi0


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Phase A2: m=1 control                                      ║
# ╚═══════════════════════════════════════════════════════════════╝

def phase_A2(xp, X, Y, KX, KY, K, dx, X_np, Y_np):
    print("=" * 60)
    print("  A2: m=1.0 control experiment")
    print("=" * 60)

    dx2 = (L/n)**2
    dt_snap = dt * save_every

    result, chi_cpu, psi0 = run_prop(xp, 1.0, k_mag, 0.0,
                                      X, Y, KX, KY, K, dx, X_np, Y_np)
    snapshots = result["snapshots"]
    snap_times = result["snap_times"]

    # Scalar MP reference
    spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
                              Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)
    E_mp = spec["E"]; G_mp = spec["Gamma"]; omega_mp = spec["omega"]
    good = (E_mp > 0) & (G_mp < 1.0)
    E_mp, G_mp, omega_mp = E_mp[good], G_mp[good], omega_mp[good]
    n_mp = min(5, len(E_mp))

    print(f"\nScalar MP: {n_mp} conduction poles")
    for j in range(n_mp):
        print(f"  E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

    # Time-filter reference modes
    tf_modes = [time_filter_mode(snapshots, snap_times, omega_mp[j], chi_cpu)
                for j in range(n_mp)]

    # DMD with various ranks
    print(f"\nDMD (auto rank)...")
    dmd = exact_dmd(snapshots, snap_times, chi_cpu, rank=None)
    print(f"  Auto rank: {dmd['rank_used']}, {len(dmd['E'])} physical poles")

    # Also test with fixed ranks
    for r in [4, 6, 8, 10, 15, 20]:
        dmd_r = exact_dmd(snapshots, snap_times, chi_cpu, rank=r)
        print(f"  rank={r:2d}: {len(dmd_r['E']):3d} poles", end="")
        # Match to MP and compute overlap
        if len(dmd_r['E']) > 0 and n_mp > 0:
            overlaps = []
            for i in range(n_mp):
                best_ov = 0.0
                for j in range(min(len(dmd_r['modes']), 10)):
                    if abs(dmd_r['E'][j] - E_mp[i]) < 0.3:
                        ov = spinor_overlap(dmd_r['modes'][j], tf_modes[i], chi_cpu, dx2)
                        best_ov = max(best_ov, ov)
                overlaps.append(best_ov)
            avg_ov = np.mean(overlaps)
            print(f"  avg_overlap={avg_ov:.4f}  "
                  f"[{', '.join(f'{o:.3f}' for o in overlaps)}]")
        else:
            print()

    # Detailed comparison for best rank
    best_r = dmd['rank_used']
    dmd_best = dmd
    print(f"\n--- Detailed comparison (rank={best_r}) ---")
    print(f"{'MP#':>4} {'E_MP':>8} {'Γ_MP':>8} {'E_DMD':>8} {'Γ_DMD':>8} "
          f"{'|ΔE|':>8} {'ΔΓ/Γ':>8} {'overlap':>8} {'x_err':>8} {'y_err':>8}")

    for i in range(n_mp):
        if len(dmd_best['E']) == 0:
            break
        dists = np.abs(dmd_best['E'] - E_mp[i])
        j = np.argmin(dists)
        dE = float(dists[j])
        relG = abs(dmd_best['Gamma'][j] - G_mp[i]) / G_mp[i] if G_mp[i] > 0 else 0
        ov = spinor_overlap(dmd_best['modes'][j], tf_modes[i], chi_cpu, dx2) if j < len(dmd_best['modes']) else 0
        rho = np.abs(dmd_best['modes'][j][0])**2 + np.abs(dmd_best['modes'][j][1])**2
        xe, ye = c2v_err(rho)
        print(f"{i+1:4d} {E_mp[i]:8.4f} {G_mp[i]:8.4f} {dmd_best['E'][j]:8.4f} "
              f"{dmd_best['Gamma'][j]:8.4f} {dE:8.4f} {relG:8.3f} {ov:8.4f} {xe:8.4f} {ye:8.4f}")

    # SVD spectrum plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(dmd['svd_spectrum'] / dmd['svd_spectrum'][0], 'o-', ms=3)
    ax1.axhline(1e-3, ls='--', color='red', label='1e-3 threshold')
    ax1.set_xlabel("index"); ax1.set_ylabel("σ / σ_max")
    ax1.set_title(f"SVD spectrum (m=1, rank={best_r})"); ax1.legend(); ax1.grid(alpha=0.3)

    # Pole scatter
    ax2.scatter(E_mp[:n_mp], G_mp[:n_mp], s=80, marker='o', facecolors='none',
               edgecolors='C0', lw=2, label='MP', zorder=3)
    if len(dmd_best['E']) > 0:
        ax2.scatter(dmd_best['E'][:10], dmd_best['Gamma'][:10],
                   s=40, marker='x', color='C1', lw=2, label='DMD', zorder=3)
    ax2.set_xlabel("E"); ax2.set_ylabel("Γ")
    ax2.set_title("Poles: MP vs DMD (m=1)"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "A2_m1_control.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure → {OUT}/A2_m1_control.png")

    return dmd_best, E_mp, G_mp, tf_modes


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Phase A3: m=0 test                                         ║
# ╚═══════════════════════════════════════════════════════════════╝

def phase_A3(xp, X, Y, KX, KY, K, dx, X_np, Y_np):
    print("\n" + "=" * 60)
    print("  A3: m=0 massless test")
    print("=" * 60)

    dx2 = (L/n)**2
    dt_snap = dt * save_every
    kx0 = k_mag / np.sqrt(2); ky0 = k_mag / np.sqrt(2)

    result, chi_cpu, psi0 = run_prop(xp, 0.0, kx0, ky0,
                                      X, Y, KX, KY, K, dx, X_np, Y_np)
    snapshots = result["snapshots"]
    snap_times = result["snap_times"]

    # Scalar MP
    spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
                              Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)
    E_mp = spec["E"]; G_mp = spec["Gamma"]; omega_mp = spec["omega"]
    good = (E_mp > 2.0) & (E_mp < 6.0) & (G_mp < 2.0)
    E_mp, G_mp, omega_mp = E_mp[good], G_mp[good], omega_mp[good]
    n_mp = min(8, len(E_mp))

    print(f"\nScalar MP: {n_mp} poles in [2,6]")
    for j in range(n_mp):
        print(f"  E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

    # Time-filter modes
    tf_modes = [time_filter_mode(snapshots, snap_times, omega_mp[j], chi_cpu)
                for j in range(n_mp)]

    # DMD rank scan
    print(f"\nDMD rank scan (m=0):")
    print(f"{'rank':>5} {'poles':>6} {'avg_ov':>8} {'best_ov':>8}  per-mode overlaps")

    best_avg = -1; best_r = 4
    for r in [4, 6, 8, 10, 12, 15, 20, 30, 50]:
        n_eff_use = min(len(snap_times), max(50, 3*r))  # ensure enough data
        dmd_r = exact_dmd(snapshots, snap_times, chi_cpu, rank=r, n_eff=n_eff_use)
        if len(dmd_r['E']) == 0:
            print(f"{r:5d} {0:6d}")
            continue
        overlaps = []
        for i in range(n_mp):
            best_ov = 0.0
            for j in range(min(len(dmd_r['modes']), 15)):
                if abs(dmd_r['E'][j] - E_mp[i]) < 0.5:
                    ov = spinor_overlap(dmd_r['modes'][j], tf_modes[i], chi_cpu, dx2)
                    best_ov = max(best_ov, ov)
            overlaps.append(best_ov)
        avg_ov = np.mean(overlaps)
        print(f"{r:5d} {len(dmd_r['E']):6d} {avg_ov:8.4f} {max(overlaps):8.4f}  "
              f"[{', '.join(f'{o:.3f}' for o in overlaps)}]")
        if avg_ov > best_avg:
            best_avg = avg_ov; best_r = r

    # Run best rank for detailed analysis
    dmd_best = exact_dmd(snapshots, snap_times, chi_cpu, rank=best_r,
                         n_eff=min(len(snap_times), max(50, 3*best_r)))
    print(f"\nBest rank: {best_r}, avg overlap: {best_avg:.4f}")

    # Mode density plots
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
            ax.set_title(f"E={dmd_best['E'][j]:.3f}\nΓ={dmd_best['Gamma'][j]:.3f}\n"
                         f"x={xe:.3f} y={ye:.3f}", fontsize=8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

            # Compare with TF mode at nearest energy
            dists = np.abs(E_mp[:n_mp] - dmd_best['E'][j])
            if len(dists) > 0:
                i_tf = np.argmin(dists)
                rho_tf = np.abs(tf_modes[i_tf][0])**2 + np.abs(tf_modes[i_tf][1])**2
                ax = axes[1, j]
                ax.imshow(rho_tf, extent=extent, origin="lower", cmap="inferno",
                          vmin=0, vmax=rho_tf.max())
                ov = spinor_overlap(phi, tf_modes[i_tf], chi_cpu, dx2)
                ax.set_title(f"TF ref E={E_mp[i_tf]:.3f}\noverlap={ov:.4f}", fontsize=8)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

        axes[0, 0].set_ylabel("DMD", fontsize=11)
        axes[1, 0].set_ylabel("Time filter", fontsize=11)
        fig.suptitle(f"DMD modes vs Time filter (m=0, rank={best_r})", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "A3_m0_modes.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure → {OUT}/A3_m0_modes.png")

    return dmd_best


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Main                                                        ║
# ╚═══════════════════════════════════════════════════════════════╝

def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")
    print(f"Grid: {n}×{n}, T={n_steps*dt:.0f}, Δt_snap={dt*save_every:.3f}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    dmd_m1, E_mp_m1, G_mp_m1, tf_m1 = phase_A2(xp, X, Y, KX, KY, K, dx, X_np, Y_np)
    dmd_m0 = phase_A3(xp, X, Y, KX, KY, K, dx, X_np, Y_np)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
