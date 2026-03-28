#!/usr/bin/env python3
"""
FDM systematic debug & validation.

Part 0: Fixed FDM core (consistent probe ranges, tunable SVD, diagnostics)
Part 1: Massive control (m=1.0) — validate FDM against time-filter ground truth
Part 2: Massless parameter scan (m=0) — sweep eps_svd × P_per × probe strategy
Part 3: Independent quality metrics (correlation recon, spectral match, orthogonality)
Part 4: Alternative probe strategies (MP-guided, complex-freq, Krylov, oversampled)
"""

import sys, os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import svd as la_svd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances, time_filter_mode

OUT_DIR = os.path.join(_root, "figures", "fdm_debug")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Fixed parameters
# ═══════════════════════════════════════════════════════════════════
n  = 256;  L = 10.0;  vF = 1.0
dt = 0.01; n_steps = 15000; save_every = 10   # T=150, Δt_snap=0.1
POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5; cap_strength = 10.0; cap_power = 2
mask_safety_factor = 0.95; mask_margin = 1.30
sigma0 = 0.6; k_mag = 3.0
Gamma_cut = 8.0; pencil_fraction = 4; L_pencil_max = 400


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Part 0 — Fixed FDM core                                        ║
# ╚═══════════════════════════════════════════════════════════════════╝

def build_probes_fixed(snapshots, snap_times, E_probes, chi_flat):
    """
    Build energy-windowed probes with consistent summation range.
    Both probes and shifted probes use m = 0 .. Ns-2.
    """
    Ns = len(snap_times)
    P  = len(E_probes)
    N_sp = snapshots.shape[2] * snapshots.shape[3]

    snap_flat = snapshots.reshape(Ns, 2, N_sp)
    snap_chi  = snap_flat * chi_flat[None, None, :]

    # phases at m=0..Ns-2
    phases = np.exp(1j * np.outer(snap_times[:Ns-1], E_probes))  # (Ns-1, P)

    S_orig  = snap_chi[:Ns-1].reshape(Ns-1, 2 * N_sp)
    S_shift = snap_chi[1:].reshape(Ns-1, 2 * N_sp)

    probes  = (phases.T @ S_orig).reshape(P, 2, N_sp)
    shifted = (phases.T @ S_shift).reshape(P, 2, N_sp)
    return probes, shifted


def build_probes_complex(snapshots, snap_times, omega_probes, chi_flat):
    """Probes with complex frequencies (damped Fourier filter)."""
    Ns = len(snap_times)
    P  = len(omega_probes)
    N_sp = snapshots.shape[2] * snapshots.shape[3]

    snap_flat = snapshots.reshape(Ns, 2, N_sp)
    snap_chi  = snap_flat * chi_flat[None, None, :]

    # e^{+i ω* t} compensates decay (ω = E - iΓ/2 → +iω* = +iE + Γ/2)
    phases = np.exp(1j * np.outer(snap_times[:Ns-1],
                                   np.conj(omega_probes)))  # (Ns-1, P)

    S_orig  = snap_chi[:Ns-1].reshape(Ns-1, 2 * N_sp)
    S_shift = snap_chi[1:].reshape(Ns-1, 2 * N_sp)

    probes  = (phases.T @ S_orig).reshape(P, 2, N_sp)
    shifted = (phases.T @ S_shift).reshape(P, 2, N_sp)
    return probes, shifted


def build_probes_krylov(snapshots, snap_times, t_indices, chi_flat):
    """Krylov probes: raw snapshots at selected times."""
    P = len(t_indices)
    N_sp = snapshots.shape[2] * snapshots.shape[3]

    snap_flat = snapshots.reshape(len(snap_times), 2, N_sp)
    snap_chi  = snap_flat * chi_flat[None, None, :]

    probes  = snap_chi[t_indices].copy()                # (P, 2, N_sp)
    shifted = snap_chi[np.minimum(t_indices + 1,
                                   len(snap_times) - 1)].copy()
    return probes.reshape(P, 2, N_sp), shifted.reshape(P, 2, N_sp)


def build_BA(probes, shifted, dx2):
    """B and A matrices from probe arrays (P, 2, N_sp)."""
    pf = probes.reshape(len(probes), -1)
    sf = shifted.reshape(len(shifted), -1)
    B = (pf.conj() @ pf.T) * dx2
    A = (pf.conj() @ sf.T) * dx2
    return B, A


def solve_fdm(B, A, dt_snap, eps_svd=1e-2, Gamma_max=8.0):
    """
    Solve A c = z B c with SVD regularisation.
    Returns dict with full diagnostics.
    """
    U_b, s_b, Vh_b = la_svd(B)
    s_max = s_b[0]
    keep  = s_b > eps_svd * s_max
    n_keep = int(keep.sum())

    s_inv_sqrt = 1.0 / np.sqrt(s_b[keep])
    B_inv_half = (U_b[:, keep] * s_inv_sqrt[None, :]) @ U_b[:, keep].conj().T

    A_red = B_inv_half @ A @ B_inv_half
    z_all, vecs_red = np.linalg.eig(A_red)
    vecs_all = B_inv_half @ vecs_red

    with np.errstate(invalid='ignore', divide='ignore'):
        omega_all = 1j * np.log(z_all + 0j) / dt_snap
    E_all     = np.real(omega_all)
    Gamma_all = -2.0 * np.imag(omega_all)

    ok = (np.isfinite(E_all) & np.isfinite(Gamma_all) &
          (Gamma_all > 0) & (Gamma_all < Gamma_max) & (np.abs(z_all) < 1.0))

    return dict(
        E=E_all[ok], Gamma=Gamma_all[ok], omega=omega_all[ok],
        vecs=vecs_all[:, ok],
        # diagnostics
        z_all=z_all, E_all=E_all, Gamma_all=Gamma_all,
        svd_spectrum=s_b, n_keep=n_keep,
        cond=s_max / s_b[keep][-1] if n_keep > 0 else np.inf,
    )


def reconstruct_modes(vecs, probes, nn, chi_cpu, dx2):
    """Build spatial modes from FDM eigenvectors."""
    probes_full = probes.reshape(len(probes), -1)  # (P, 2*N_sp)
    modes = []
    for col in range(vecs.shape[1]):
        c   = vecs[:, col]
        phi = (c @ probes_full).reshape(2, nn, nn)
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        norm = np.sqrt(np.sum(rho * chi_cpu) * dx2)
        if norm > 1e-30:
            phi = phi / norm
        modes.append(phi)
    return modes


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Quality metrics                                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝

def metric_spectral_match(E_fdm, G_fdm, E_mp, G_mp, sig_E=0.2, sig_G=0.2):
    """For each MP pole, find best FDM match. Returns score in [0,1]."""
    if len(E_fdm) == 0 or len(E_mp) == 0:
        return 0.0
    score = 0.0
    for i in range(len(E_mp)):
        best = 0.0
        for j in range(len(E_fdm)):
            s = (np.exp(-abs(E_fdm[j] - E_mp[i]) / sig_E) *
                 np.exp(-abs(G_fdm[j] - G_mp[i]) / sig_G))
            best = max(best, s)
        score += best
    return score / len(E_mp)


def metric_corr_recon(C_R, corr_times, omega_fdm, modes, psi0_cpu,
                      chi_cpu, dx2):
    """Reconstruct C_R from FDM modes and compare."""
    if len(modes) == 0:
        return 1.0
    # amplitudes a_n = <psi0|chi|phi_n>
    a = np.array([
        (np.vdot(psi0_cpu[0] * chi_cpu, m[0]) +
         np.vdot(psi0_cpu[1] * chi_cpu, m[1])) * dx2
        for m in modes
    ])
    # reconstruct: C_R^FDM(t) = Σ |a_n|² e^{-iω_n t}
    C_recon = np.zeros(len(corr_times), dtype=complex)
    for nn_idx in range(len(modes)):
        C_recon += np.abs(a[nn_idx])**2 * np.exp(
            -1j * omega_fdm[nn_idx] * corr_times)
    err = np.linalg.norm(C_R - C_recon) / (np.linalg.norm(C_R) + 1e-30)
    return float(err)


def metric_orthogonality(modes, chi_cpu, dx2):
    """Max off-diagonal overlap |<φ_i|χ|φ_j>|."""
    N = len(modes)
    if N < 2:
        return 0.0
    O = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            O[i, j] = (np.vdot(modes[i][0] * chi_cpu, modes[j][0]) +
                        np.vdot(modes[i][1] * chi_cpu, modes[j][1])) * dx2
    offdiag = np.abs(O) - np.diag(np.diag(np.abs(O)))
    return float(np.max(offdiag))


def metric_snapshot_recon(snapshots, snap_times, omega_fdm, modes,
                          chi_cpu, dx2, n_samples=20):
    """Average relative error of snapshot reconstruction inside mask."""
    if len(modes) == 0:
        return 1.0
    idx = np.linspace(0, len(snap_times) - 1, n_samples, dtype=int)
    # amplitudes from t=0 snapshot
    a = np.array([
        (np.vdot(modes[nn_idx][0] * chi_cpu, snapshots[0, 0] * chi_cpu) +
         np.vdot(modes[nn_idx][1] * chi_cpu, snapshots[0, 1] * chi_cpu)) * dx2
        for nn_idx in range(len(modes))
    ])
    errors = []
    for k in idx:
        recon = np.zeros((2, chi_cpu.shape[0], chi_cpu.shape[1]), dtype=complex)
        for nn_idx in range(len(modes)):
            phase = a[nn_idx] * np.exp(-1j * omega_fdm[nn_idx] * snap_times[k])
            recon += phase * modes[nn_idx]
        actual = snapshots[k] * chi_cpu[None]
        err  = np.sqrt(np.sum(np.abs(recon - actual)**2) * dx2)
        norm = np.sqrt(np.sum(np.abs(actual)**2) * dx2)
        if norm > 1e-20:
            errors.append(err / norm)
    return float(np.mean(errors)) if errors else 1.0


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Propagation helper                                              ║
# ╚═══════════════════════════════════════════════════════════════════╝

def run_propagation(xp, mass, kx0, ky0, X, Y, KX, KY, K, dx,
                    X_np, Y_np, V_np, chi_cpu, chi_flat):
    """Run one propagation. Returns snapshots, snap_times, C_R, etc."""
    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(
        POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, mass, V, W, KX, KY, K, xp)

    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=0.0, y0=0.0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)

    result = propagate(
        psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt,
        save_every=save_every, corr_every=save_every,
        max_record=n_steps // save_every + 2,
        measure_transmission=False)

    psi0_cpu = np.stack([xp.to_numpy(psi0_up), xp.to_numpy(psi0_dn)])
    return dict(
        snapshots=result["snapshots"], snap_times=result["snap_times"],
        C_R=result["C_R"], corr_times=result["corr_times"],
        psi0_cpu=psi0_cpu,
    )


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FDM run wrapper                                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝

def run_fdm(snapshots, snap_times, chi_flat, chi_cpu, dx2, dt_snap,
            E_probes, eps_svd=1e-2, strategy="uniform",
            omega_probes=None, krylov_indices=None):
    """Run FDM with specified probe strategy. Returns solve_fdm result + modes."""
    if strategy == "complex" and omega_probes is not None:
        probes, shifted = build_probes_complex(
            snapshots, snap_times, omega_probes, chi_flat)
    elif strategy == "krylov" and krylov_indices is not None:
        probes, shifted = build_probes_krylov(
            snapshots, snap_times, krylov_indices, chi_flat)
    else:
        probes, shifted = build_probes_fixed(
            snapshots, snap_times, E_probes, chi_flat)

    B, A = build_BA(probes, shifted, dx2)
    res  = solve_fdm(B, A, dt_snap, eps_svd=eps_svd, Gamma_max=Gamma_cut)

    modes = []
    if res["vecs"].shape[1] > 0:
        n_show = min(10, res["vecs"].shape[1])
        order  = np.argsort(res["Gamma"])[:n_show]
        res["E"]     = res["E"][order]
        res["Gamma"] = res["Gamma"][order]
        res["omega"] = res["omega"][order]
        res["vecs"]  = res["vecs"][:, order]
        modes = reconstruct_modes(res["vecs"], probes, n, chi_cpu, dx2)

    res["modes"] = modes
    return res


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Part 1 — Massive control (m=1.0)                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

def part1_massive(xp, X, Y, KX, KY, K, dx, X_np, Y_np, V_np,
                  chi_cpu, chi_flat, dx2):
    print("\n" + "="*70)
    print("  PART 1: Massive control (m=1.0)")
    print("="*70)

    dt_snap = dt * save_every

    # ── propagate ──
    print("Propagating m=1.0 ...")
    data = run_propagation(xp, mass=1.0, kx0=k_mag, ky0=0.0,
                           X=X, Y=Y, KX=KX, KY=KY, K=K, dx=dx,
                           X_np=X_np, Y_np=Y_np, V_np=None,
                           chi_cpu=chi_cpu, chi_flat=chi_flat)

    # ── matrix pencil ──
    print("Matrix pencil ...")
    spec = extract_resonances(data["C_R"], data["corr_times"], dt_snap,
                              Gamma_cut=Gamma_cut, pencil_fraction=pencil_fraction,
                              L_pencil_max=L_pencil_max)
    E_mp, G_mp, omega_mp = spec["E"], spec["Gamma"], spec["omega"]
    n_mp = min(5, len(E_mp))
    print(f"  MP found {len(E_mp)} poles, showing top {n_mp}:")
    for j in range(n_mp):
        print(f"    E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

    # ── time-filter modes ──
    print("Time filtering ...")
    tf_modes = [time_filter_mode(data["snapshots"], data["snap_times"],
                                  omega_mp[j], chi_cpu)
                for j in range(n_mp)]

    # ── FDM with MP-guided probes ──
    print("FDM (MP-guided probes, eps_svd=1e-2) ...")
    E_probes = E_mp[:n_mp].copy()
    if len(E_probes) < 5:
        E_probes = np.sort(np.concatenate([
            E_probes,
            np.linspace(E_mp[0] - 0.5, E_mp[-1] + 0.5, 5 - len(E_probes))
        ]))
    fdm = run_fdm(data["snapshots"], data["snap_times"], chi_flat, chi_cpu,
                  dx2, dt_snap, E_probes, eps_svd=1e-2, strategy="mp_guided")

    E_fdm, G_fdm = fdm["E"], fdm["Gamma"]
    print(f"  FDM found {len(E_fdm)} poles:")
    for j in range(min(8, len(E_fdm))):
        print(f"    E={E_fdm[j]:.4f}  Γ={G_fdm[j]:.4f}")

    # ── match & compare ──
    print("\nMatching FDM ↔ MP:")
    matches = []
    for i in range(n_mp):
        if len(E_fdm) == 0:
            break
        dists = np.abs(E_fdm - E_mp[i])
        j_best = np.argmin(dists)
        dE = abs(E_fdm[j_best] - E_mp[i])
        dG = abs(G_fdm[j_best] - G_mp[i])
        relG = dG / G_mp[i] if G_mp[i] > 0 else np.inf

        # spatial overlap with time-filtered mode
        overlap = 0.0
        if j_best < len(fdm["modes"]) and i < len(tf_modes):
            phi_f = fdm["modes"][j_best]
            phi_t = tf_modes[i]
            ovlp = (np.vdot(phi_f[0] * chi_cpu, phi_t[0]) +
                    np.vdot(phi_f[1] * chi_cpu, phi_t[1])) * dx2
            overlap = float(np.abs(ovlp)**2)

        print(f"  MP#{i+1} E={E_mp[i]:.4f} Γ={G_mp[i]:.4f}  ←→  "
              f"FDM#{j_best+1} E={E_fdm[j_best]:.4f} Γ={G_fdm[j_best]:.4f}  "
              f"|ΔE|={dE:.4f}  ΔΓ/Γ={relG:.3f}  overlap={overlap:.4f}")
        matches.append(dict(E_mp=E_mp[i], G_mp=G_mp[i],
                            E_fdm=E_fdm[j_best], G_fdm=G_fdm[j_best],
                            dE=dE, relG=relG, overlap=overlap))

    # ── metrics ──
    sm = metric_spectral_match(E_fdm, G_fdm, E_mp[:n_mp], G_mp[:n_mp])
    cr = metric_corr_recon(data["C_R"], data["corr_times"], fdm["omega"],
                           fdm["modes"], data["psi0_cpu"], chi_cpu, dx2)
    orth = metric_orthogonality(fdm["modes"], chi_cpu, dx2)
    sr = metric_snapshot_recon(data["snapshots"], data["snap_times"],
                               fdm["omega"], fdm["modes"], chi_cpu, dx2)
    print(f"\n  Metrics: spectral_match={sm:.4f}  corr_recon_err={cr:.4f}  "
          f"ortho_max={orth:.4f}  snap_recon_err={sr:.4f}")

    # ── figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.scatter(E_mp[:n_mp], G_mp[:n_mp], s=80, marker='o', facecolors='none',
               edgecolors='C0', linewidths=2, label='Matrix pencil', zorder=3)
    if len(E_fdm) > 0:
        ax.scatter(E_fdm, G_fdm, s=40, marker='x', color='C1', linewidths=2,
                   label='FDM', zorder=3)
    ax.set_xlabel("E"); ax.set_ylabel("Γ")
    ax.set_title("Part 1: m=1.0 pole comparison"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(matches)), [m["relG"] for m in matches], color='C2')
    ax.axhline(0.15, ls='--', color='r', label='15% threshold')
    ax.set_xlabel("mode index"); ax.set_ylabel("|ΔΓ/Γ|")
    ax.set_title("Relative Γ error"); ax.legend()

    ax = axes[2]
    ax.bar(range(len(matches)), [m["overlap"] for m in matches], color='C3')
    ax.axhline(0.85, ls='--', color='r', label='0.85 threshold')
    ax.set_xlabel("mode index"); ax.set_ylabel("|overlap|²")
    ax.set_title("Spatial overlap with time-filter"); ax.legend()

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "part1_massive.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure → {out}")

    return dict(matches=matches, sm=sm, cr=cr, orth=orth, sr=sr,
                E_mp=E_mp, G_mp=G_mp, E_fdm=E_fdm, G_fdm=G_fdm,
                svd=fdm["svd_spectrum"])


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Part 2 — Massless parameter scan (m=0)                         ║
# ╚═══════════════════════════════════════════════════════════════════╝

def part2_scan(xp, X, Y, KX, KY, K, dx, X_np, Y_np,
               chi_cpu, chi_flat, dx2):
    print("\n" + "="*70)
    print("  PART 2: Massless parameter scan (m=0)")
    print("="*70)

    dt_snap = dt * save_every
    kx0 = k_mag / np.sqrt(2); ky0 = k_mag / np.sqrt(2)

    # ── propagate ──
    print(f"Propagating m=0 (45° launch, k=({kx0:.2f},{ky0:.2f})) ...")
    data = run_propagation(xp, mass=0.0, kx0=kx0, ky0=ky0,
                           X=X, Y=Y, KX=KX, KY=KY, K=K, dx=dx,
                           X_np=X_np, Y_np=Y_np, V_np=None,
                           chi_cpu=chi_cpu, chi_flat=chi_flat)

    # ── matrix pencil ──
    print("Matrix pencil ...")
    spec = extract_resonances(data["C_R"], data["corr_times"], dt_snap,
                              Gamma_cut=Gamma_cut, pencil_fraction=pencil_fraction,
                              L_pencil_max=L_pencil_max)
    E_mp, G_mp, omega_mp = spec["E"], spec["Gamma"], spec["omega"]
    n_mp = min(8, len(E_mp))
    print(f"  MP: {len(E_mp)} poles. Top {n_mp}:")
    for j in range(n_mp):
        print(f"    E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

    # ── parameter scan ──
    eps_values  = [1e-1, 1e-2, 1e-3, 1e-4]
    P_values    = [5, 10, 15, 20, 30]
    strategies  = ["uniform", "mp_guided", "mixed"]

    results = np.full((len(eps_values), len(P_values), len(strategies), 2),
                      np.nan)  # last dim: [spectral_match, corr_err]

    best_score = -1.0
    best_params = None
    best_fdm = None

    print(f"\nScanning {len(eps_values)}×{len(P_values)}×{len(strategies)} "
          f"= {len(eps_values)*len(P_values)*len(strategies)} combinations ...")

    for ie, eps in enumerate(eps_values):
        for ip, P_per in enumerate(P_values):
            for ist, strat in enumerate(strategies):
                # build probe energies
                if strat == "uniform":
                    E_pr = np.linspace(1.0, 5.0, P_per)
                elif strat == "mp_guided":
                    E_pr = E_mp[:min(n_mp, P_per)].copy()
                    if len(E_pr) < P_per:
                        fill = np.linspace(
                            max(0.5, E_mp[0] - 0.5),
                            min(6.0, E_mp[min(n_mp-1, len(E_mp)-1)] + 0.5),
                            P_per - len(E_pr))
                        E_pr = np.sort(np.concatenate([E_pr, fill]))
                else:  # mixed
                    n_mp_use = min(n_mp, P_per // 2)
                    E_mp_part = E_mp[:n_mp_use]
                    n_fill = P_per - n_mp_use
                    E_fill = np.linspace(1.0, 5.0, n_fill)
                    E_pr = np.sort(np.unique(np.concatenate([E_mp_part, E_fill])))
                    if len(E_pr) < P_per:
                        E_pr = np.linspace(1.0, 5.0, P_per)

                try:
                    fdm = run_fdm(data["snapshots"], data["snap_times"],
                                  chi_flat, chi_cpu, dx2, dt_snap,
                                  E_pr, eps_svd=eps, strategy=strat)
                    sm = metric_spectral_match(fdm["E"], fdm["Gamma"],
                                               E_mp[:n_mp], G_mp[:n_mp])
                    cr = metric_corr_recon(data["C_R"], data["corr_times"],
                                           fdm["omega"], fdm["modes"],
                                           data["psi0_cpu"], chi_cpu, dx2)
                    results[ie, ip, ist, 0] = sm
                    results[ie, ip, ist, 1] = cr

                    if sm > best_score:
                        best_score = sm
                        best_params = (eps, P_per, strat)
                        best_fdm = fdm
                except Exception as ex:
                    warnings.warn(f"  Failed: eps={eps}, P={P_per}, "
                                  f"strat={strat}: {ex}")

    print(f"\nBest: eps_svd={best_params[0]:.0e}, P_per={best_params[1]}, "
          f"strat={best_params[2]}  →  spectral_match={best_score:.4f}")
    if best_fdm is not None:
        print(f"  FDM poles ({len(best_fdm['E'])}):")
        for j in range(min(8, len(best_fdm["E"]))):
            print(f"    E={best_fdm['E'][j]:.4f}  Γ={best_fdm['Gamma'][j]:.4f}")

    # ── figures ──
    # heatmap for each strategy
    fig, axes = plt.subplots(1, len(strategies), figsize=(5*len(strategies), 4))
    if len(strategies) == 1:
        axes = [axes]
    for ist, strat in enumerate(strategies):
        ax = axes[ist]
        mat = results[:, :, ist, 0]  # spectral_match
        im = ax.imshow(mat, aspect='auto', origin='lower',
                       vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(P_values)))
        ax.set_xticklabels(P_values)
        ax.set_yticks(range(len(eps_values)))
        ax.set_yticklabels([f"{e:.0e}" for e in eps_values])
        ax.set_xlabel("P_per"); ax.set_ylabel("eps_svd")
        ax.set_title(f"{strat}\nbest={np.nanmax(mat):.3f}")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Part 2: Spectral matching score (m=0)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "part2_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Heatmap → {out}")

    # best poles scatter
    if best_fdm is not None and len(best_fdm["E"]) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.scatter(E_mp[:n_mp], G_mp[:n_mp], s=80, marker='o',
                    facecolors='none', edgecolors='C0', linewidths=2,
                    label='MP', zorder=3)
        ax1.scatter(best_fdm["E"], best_fdm["Gamma"], s=40, marker='x',
                    color='C1', linewidths=2, label='FDM (best)', zorder=3)
        ax1.set_xlabel("E"); ax1.set_ylabel("Γ")
        ax1.set_title(f"Best FDM: eps={best_params[0]:.0e} P={best_params[1]} {best_params[2]}")
        ax1.legend(); ax1.grid(alpha=0.3)

        # z-plane
        z_all = best_fdm["z_all"]
        theta = np.linspace(0, 2*np.pi, 200)
        ax2.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.5)
        ax2.scatter(np.real(z_all), np.imag(z_all), s=10, alpha=0.5,
                    label='all z')
        ok_z = best_fdm["z_all"][np.abs(best_fdm["z_all"]) < 1]
        ax2.set_xlabel("Re(z)"); ax2.set_ylabel("Im(z)")
        ax2.set_title("z-plane (all eigenvalues)"); ax2.set_aspect('equal')
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "part2_best_poles.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Best poles → {out}")

    # SVD spectrum
    if best_fdm is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        s = best_fdm["svd_spectrum"]
        ax.semilogy(s / s[0], 'o-', ms=4)
        for eps in eps_values:
            ax.axhline(eps, ls='--', lw=0.8, alpha=0.5,
                       label=f"eps={eps:.0e}")
        ax.set_xlabel("index"); ax.set_ylabel("σ / σ_max")
        ax.set_title("B matrix SVD spectrum (best params)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, "part2_svd_spectrum.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  SVD → {out}")

    return dict(results=results, best_params=best_params,
                best_score=best_score, best_fdm=best_fdm,
                E_mp=E_mp, G_mp=G_mp, omega_mp=omega_mp,
                data=data, eps_values=eps_values,
                P_values=P_values, strategies=strategies)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Part 3 — All metrics for best params                           ║
# ╚═══════════════════════════════════════════════════════════════════╝

def part3_metrics(p2_result, chi_cpu, dx2):
    print("\n" + "="*70)
    print("  PART 3: Quality metrics for best parameter set")
    print("="*70)

    fdm  = p2_result["best_fdm"]
    data = p2_result["data"]
    E_mp = p2_result["E_mp"]
    G_mp = p2_result["G_mp"]

    if fdm is None or len(fdm["modes"]) == 0:
        print("  No valid FDM result from Part 2. Skipping.")
        return {}

    sm = metric_spectral_match(fdm["E"], fdm["Gamma"], E_mp, G_mp)
    cr = metric_corr_recon(data["C_R"], data["corr_times"], fdm["omega"],
                           fdm["modes"], data["psi0_cpu"], chi_cpu, dx2)
    orth = metric_orthogonality(fdm["modes"], chi_cpu, dx2)
    sr = metric_snapshot_recon(data["snapshots"], data["snap_times"],
                               fdm["omega"], fdm["modes"], chi_cpu, dx2)

    print(f"  spectral_match   = {sm:.4f}")
    print(f"  corr_recon_err   = {cr:.4f}")
    print(f"  ortho_max_offdiag= {orth:.4f}")
    print(f"  snap_recon_err   = {sr:.4f}")

    # figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    labels = ["Spectral\nmatch ↑", "Corr recon\nerror ↓",
              "Ortho max\noff-diag ↓", "Snap recon\nerror ↓"]
    vals = [sm, cr, orth, sr]
    colors = ['C0', 'C1', 'C2', 'C3']
    for ax, lab, val, col in zip(axes, labels, vals, colors):
        ax.bar([0], [val], color=col, width=0.5)
        ax.set_ylabel(lab)
        ax.set_xticks([])
        ax.text(0, val + 0.02, f"{val:.4f}", ha='center', fontsize=10)
    fig.suptitle(f"Part 3: Metrics (m=0, best params: eps={p2_result['best_params'][0]:.0e} "
                 f"P={p2_result['best_params'][1]} {p2_result['best_params'][2]})",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "part3_metrics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure → {out}")

    return dict(sm=sm, cr=cr, orth=orth, sr=sr)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Part 4 — Alternative probe strategies                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

def part4_strategies(p2_result, chi_cpu, chi_flat, dx2):
    print("\n" + "="*70)
    print("  PART 4: Alternative probe strategies (m=0)")
    print("="*70)

    data = p2_result["data"]
    E_mp = p2_result["E_mp"]
    G_mp = p2_result["G_mp"]
    omega_mp = p2_result["omega_mp"]
    dt_snap = dt * save_every
    n_mp = min(8, len(E_mp))

    best_eps = p2_result["best_params"][0] if p2_result["best_params"] else 1e-2
    strats = {}

    # Strategy 1: MP-guided (real energies from matrix pencil)
    print("Strategy 1: MP-guided (real E from MP) ...")
    E_pr = E_mp[:n_mp].copy()
    if len(E_pr) < 5:
        E_pr = np.sort(np.concatenate([
            E_pr, np.linspace(1.0, 5.0, 5 - len(E_pr))]))
    fdm1 = run_fdm(data["snapshots"], data["snap_times"], chi_flat, chi_cpu,
                   dx2, dt_snap, E_pr, eps_svd=best_eps, strategy="mp_guided")
    sm1 = metric_spectral_match(fdm1["E"], fdm1["Gamma"], E_mp[:n_mp], G_mp[:n_mp])
    cr1 = metric_corr_recon(data["C_R"], data["corr_times"], fdm1["omega"],
                            fdm1["modes"], data["psi0_cpu"], chi_cpu, dx2)
    strats["MP-guided"] = (sm1, cr1, fdm1)
    print(f"  spec_match={sm1:.4f}  corr_err={cr1:.4f}  poles={len(fdm1['E'])}")

    # Strategy 2: Complex-frequency probes
    print("Strategy 2: Complex-freq (ω = E - iΓ/2 from MP) ...")
    omega_pr = omega_mp[:n_mp].copy()
    if len(omega_pr) < 5:
        extra = np.linspace(1.0, 5.0, 5 - len(omega_pr))
        omega_pr = np.concatenate([omega_pr, extra - 0.3j])
    fdm2 = run_fdm(data["snapshots"], data["snap_times"], chi_flat, chi_cpu,
                   dx2, dt_snap, None, eps_svd=best_eps, strategy="complex",
                   omega_probes=omega_pr)
    sm2 = metric_spectral_match(fdm2["E"], fdm2["Gamma"], E_mp[:n_mp], G_mp[:n_mp])
    cr2 = metric_corr_recon(data["C_R"], data["corr_times"], fdm2["omega"],
                            fdm2["modes"], data["psi0_cpu"], chi_cpu, dx2)
    strats["Complex-freq"] = (sm2, cr2, fdm2)
    print(f"  spec_match={sm2:.4f}  corr_err={cr2:.4f}  poles={len(fdm2['E'])}")

    # Strategy 3: Krylov (geometric time sampling)
    print("Strategy 3: Krylov (geometric time sampling) ...")
    Ns = len(data["snap_times"])
    # geometric: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, ...
    krylov_idx = [0]
    step = 1
    while krylov_idx[-1] + step < Ns - 1:
        krylov_idx.append(krylov_idx[-1] + step)
        step = max(1, int(step * 1.5))
    krylov_idx = np.array(krylov_idx[:30])  # cap at 30
    fdm3 = run_fdm(data["snapshots"], data["snap_times"], chi_flat, chi_cpu,
                   dx2, dt_snap, None, eps_svd=best_eps, strategy="krylov",
                   krylov_indices=krylov_idx)
    sm3 = metric_spectral_match(fdm3["E"], fdm3["Gamma"], E_mp[:n_mp], G_mp[:n_mp])
    cr3 = metric_corr_recon(data["C_R"], data["corr_times"], fdm3["omega"],
                            fdm3["modes"], data["psi0_cpu"], chi_cpu, dx2)
    strats["Krylov"] = (sm3, cr3, fdm3)
    print(f"  spec_match={sm3:.4f}  corr_err={cr3:.4f}  poles={len(fdm3['E'])}")

    # Strategy 4: Oversampled + aggressive SVD
    print("Strategy 4: Oversampled (P=50, eps=0.05) ...")
    E_pr4 = np.linspace(0.5, 5.5, 50)
    fdm4 = run_fdm(data["snapshots"], data["snap_times"], chi_flat, chi_cpu,
                   dx2, dt_snap, E_pr4, eps_svd=0.05, strategy="uniform")
    sm4 = metric_spectral_match(fdm4["E"], fdm4["Gamma"], E_mp[:n_mp], G_mp[:n_mp])
    cr4 = metric_corr_recon(data["C_R"], data["corr_times"], fdm4["omega"],
                            fdm4["modes"], data["psi0_cpu"], chi_cpu, dx2)
    strats["Oversampled"] = (sm4, cr4, fdm4)
    print(f"  spec_match={sm4:.4f}  corr_err={cr4:.4f}  poles={len(fdm4['E'])}")

    # ── figure ──
    names = list(strats.keys())
    sms   = [strats[k][0] for k in names]
    crs   = [strats[k][1] for k in names]
    n_poles = [len(strats[k][2]["E"]) for k in names]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(names))
    ax1.bar(x, sms, color='C0'); ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("Spectral match ↑"); ax1.set_title("Score"); ax1.set_ylim(0, 1)

    ax2.bar(x, crs, color='C1'); ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("Corr recon error ↓"); ax2.set_title("Error")

    ax3.bar(x, n_poles, color='C2'); ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=8)
    ax3.set_ylabel("N poles"); ax3.set_title("Physical poles found")

    fig.suptitle("Part 4: Probe strategy comparison (m=0)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "part4_strategies.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure → {out}")

    return strats


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Main                                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝

def main():
    t_total = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")
    print(f"Grid: {n}×{n}, L={L}, T_max={n_steps*dt:.0f}, Δt_snap={dt*save_every:.3f}")
    print(f"Output: {OUT_DIR}/\n")

    # ── one-time grid setup ──
    x_np       = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx_np = L / n;  dx2 = dx_np**2

    chi_R, _ = build_analysis_mask(
        POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp)
    chi_cpu  = xp.to_numpy(chi_R)
    chi_flat = chi_cpu.ravel()

    # ── run all parts ──
    p1 = part1_massive(xp, X, Y, KX, KY, K, dx, X_np, Y_np, None,
                       chi_cpu, chi_flat, dx2)

    p2 = part2_scan(xp, X, Y, KX, KY, K, dx, X_np, Y_np,
                    chi_cpu, chi_flat, dx2)

    p3 = part3_metrics(p2, chi_cpu, dx2)

    p4 = part4_strategies(p2, chi_cpu, chi_flat, dx2)

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*70}")
    print(f"  TOTAL WALL TIME: {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"  Results in: {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
