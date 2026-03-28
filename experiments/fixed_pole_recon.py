#!/usr/bin/env python3
"""
Fixed-pole regularized mode reconstruction.

Given known complex frequencies ω_n from matrix pencil, reconstruct spatial
modes φ_n(r) by solving a constrained linear inverse problem:

    ψ(r, t_k) ≈ Σ_n φ_n(r) e^{-iω_n t_k}

Three routes:
  A: Direct Tikhonov-regularized Vandermonde (full spatial grid)
  B: POD-reduced fixed-pole fit (recommended — drops unknowns from 131k to ~40)
  C: Constrained DMD (fix eigenvalues, only solve for modes)

All routes exploit the key insight: poles are already accurately known,
so mode reconstruction is a LINEAR problem with known time basis functions.
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye as speye

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances, time_filter_mode, reconstruct_modes_lstsq

OUT = os.path.join(_root, "figures", "fixed_pole")
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Physics (same as dmd_extract.py)
# ═══════════════════════════════════════════════════════════════
n = 256; L = 10.0; vF = 1.0
dt = 0.01; n_steps = 15000; save_every = 10  # Δt_snap = 0.1
POTENTIAL = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5; cap_strength = 10.0; cap_power = 2
sigma0 = 0.6; k_mag = 3.0
Gamma_cut = 8.0


# ═══════════════════════════════════════════════════════════════
# Utilities (from dmd_extract.py)
# ═══════════════════════════════════════════════════════════════

def spinor_overlap(a, b, chi, dx2):
    """Normalized overlap |<a|χ|b>|² / (<a|χ|a><b|χ|b>)."""
    ab = (np.vdot(a[0]*chi, b[0]) + np.vdot(a[1]*chi, b[1])) * dx2
    aa = (np.vdot(a[0]*chi, a[0]) + np.vdot(a[1]*chi, a[1])) * dx2
    bb = (np.vdot(b[0]*chi, b[0]) + np.vdot(b[1]*chi, b[1])) * dx2
    denom = np.real(aa) * np.real(bb)
    return float(np.abs(ab)**2 / denom) if denom > 1e-30 else 0.0


def fft_flip(a, ax):
    return np.roll(np.flip(a, axis=ax), 1, axis=ax)


def c2v_err(rho):
    s = np.sum(rho) + 1e-30
    return (np.sum(np.abs(fft_flip(rho, 1) - rho)) / s,
            np.sum(np.abs(fft_flip(rho, 0) - rho)) / s)


def _normalize_mask(phi, chi, dx2):
    """Normalize spinor inside mask: ∫|φ|²χ dx² = 1."""
    rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
    norm = np.sqrt(np.sum(rho * chi) * dx2)
    return phi / norm if norm > 1e-30 else phi


# ═══════════════════════════════════════════════════════════════
# Propagation helper (from dmd_extract.py)
# ═══════════════════════════════════════════════════════════════

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
        max_record=n_steps // save_every + 2, measure_transmission=False)
    chi_cpu = xp.to_numpy(chi_R)
    return result, chi_cpu


# ═══════════════════════════════════════════════════════════════
# Adaptive time windowing (shared by Routes A/B)
# ═══════════════════════════════════════════════════════════════

def _adaptive_time_mask(snap_times, omega_arr, n_modes, quantile=0.5,
                        min_factor=3):
    """Select snapshots where signal is significant.

    Uses the *quantile* of Gamma (not minimum) to set cutoff.
    """
    Gammas = -2.0 * np.imag(omega_arr[:n_modes])
    pos = Gammas[Gammas > 0]
    if len(pos) == 0:
        return np.ones(len(snap_times), dtype=bool)
    Gamma_ref = np.quantile(pos, quantile)
    if Gamma_ref <= 0:
        return np.ones(len(snap_times), dtype=bool)
    t_cut = 2.0 * np.log(100) / (Gamma_ref / 2.0)  # exp(-Γ/2 · t) = 0.01
    mask_t = snap_times <= t_cut
    if np.sum(mask_t) < min_factor * n_modes:
        mask_t = np.ones(len(snap_times), dtype=bool)
    return mask_t


# ═══════════════════════════════════════════════════════════════
# Route A: Direct Tikhonov-regularized Vandermonde
# ═══════════════════════════════════════════════════════════════

def fixed_pole_recon_direct(snapshots, snap_times, omega_arr, chi_R,
                            n_modes=5, lambda_tik=1e-3, t_cut_quantile=0.5):
    """
    Reconstruct spatial modes using known poles + Tikhonov regularization.

    Solves per spinor component:
        (V†V + λI) φ_coeffs = V† data
    where V[k,j] = exp(-i ω_j t_k).
    """
    N = min(n_modes, len(omega_arr))
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    dx2 = (L / n) ** 2

    # Adaptive time window
    mask_t = _adaptive_time_mask(snap_times, omega_arr, N, t_cut_quantile)
    t_use = snap_times[mask_t]
    snap_use = snapshots[mask_t]
    Ns = len(t_use)

    # Vandermonde matrix
    V = np.exp(-1j * np.outer(t_use, omega_arr[:N]))  # (Ns, N)

    # Regularized normal equations
    VtV = V.conj().T @ V  # (N, N)
    reg = VtV + lambda_tik * np.eye(N)

    modes = []
    for j in range(N):
        modes.append(np.zeros((2, ny, nx), dtype=complex))

    for alpha in range(2):
        data = np.empty((Ns, ny * nx), dtype=complex)
        for k in range(Ns):
            data[k, :] = (snap_use[k, alpha] * chi_R).ravel()
        rhs = V.conj().T @ data  # (N, ny*nx)
        coeffs = np.linalg.solve(reg, rhs)  # (N, ny*nx)
        for j in range(N):
            modes[j][alpha] = coeffs[j].reshape(ny, nx)

    for j in range(N):
        modes[j] = _normalize_mask(modes[j], chi_R, dx2)

    return modes


# ═══════════════════════════════════════════════════════════════
# Route B: POD-reduced fixed-pole fit
# ═══════════════════════════════════════════════════════════════

def fixed_pole_recon_pod(snapshots, snap_times, omega_arr, chi_R,
                         n_modes=5, rank=40, lambda_tik=1e-3,
                         lambda_smooth=0.0, t_cut_quantile=0.5):
    """
    POD-reduced fixed-pole reconstruction.

    1. SVD of snapshot matrix → POD basis U_r (D, r)
    2. Project time series → C_pod (r, Ns)
    3. Fit known Vandermonde in reduced space
    4. Reconstruct: φ_n = U_r @ c_n
    """
    N = min(n_modes, len(omega_arr))
    Ns_all = len(snap_times)
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    dx2 = (L / n) ** 2
    D = 2 * ny * nx

    # Build masked snapshot matrix (same convention as DMD)
    X = np.zeros((D, Ns_all), dtype=complex)
    for k in range(Ns_all):
        X[:ny*nx, k] = (snapshots[k, 0] * chi_R).ravel()
        X[ny*nx:, k] = (snapshots[k, 1] * chi_R).ravel()

    # SVD
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    r = min(rank, len(S))
    U_r = U[:, :r]                           # (D, r)
    C_pod = np.diag(S[:r]) @ Vh[:r, :]       # (r, Ns_all)

    del X  # free memory

    # Adaptive time window
    mask_t = _adaptive_time_mask(snap_times, omega_arr, N, t_cut_quantile)
    t_use = snap_times[mask_t]
    C_pod_use = C_pod[:, mask_t]             # (r, Ns_use)
    Ns_use = len(t_use)

    # Vandermonde matrix from known poles
    V = np.exp(-1j * np.outer(t_use, omega_arr[:N]))  # (Ns_use, N)
    VtV = V.conj().T @ V  # (N, N)

    # Build regularization matrix
    if lambda_smooth > 0:
        # Laplacian penalty in POD space: G = (L U_r)† (L U_r)
        Lx = diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx), format='csr')
        Ly = diags([-1, 2, -1], [-1, 0, 1], shape=(ny, ny), format='csr')
        L2d = kron(speye(ny), Lx) + kron(Ly, speye(nx))
        L_block = kron(speye(2), L2d)  # (D, D) sparse — both spinor components
        LU = L_block @ U_r             # (D, r) — sparse @ dense
        G_smooth = LU.T.conj() @ LU    # (r, r)

        # Kronecker system: solve for vec(C_modes) of shape (N*r,)
        # (V†V ⊗ I_r + λ_tik I_{Nr} + λ_smooth I_N ⊗ G_smooth) vec = rhs
        I_r = np.eye(r)
        I_N = np.eye(N)
        M = np.kron(VtV, I_r) + lambda_tik * np.eye(N * r) + \
            lambda_smooth * np.kron(I_N, G_smooth)
        rhs_flat = (V.conj().T @ C_pod_use.T).ravel()  # vec(V† C_pod_use.T)
        c_flat = np.linalg.solve(M, rhs_flat)
        C_modes = c_flat.reshape(N, r)
    else:
        # Simple Tikhonov — per POD component (decoupled)
        reg = VtV + lambda_tik * np.eye(N)
        rhs = V.conj().T @ C_pod_use.T  # (N, r)
        C_modes = np.linalg.solve(reg, rhs)  # (N, r)

    # Reconstruct spatial modes
    modes = []
    for j in range(N):
        c_n = C_modes[j, :]              # (r,)
        phi_flat = U_r @ c_n             # (D,)
        phi = np.zeros((2, ny, nx), dtype=complex)
        phi[0] = phi_flat[:ny*nx].reshape(ny, nx)
        phi[1] = phi_flat[ny*nx:].reshape(ny, nx)
        phi = _normalize_mask(phi, chi_R, dx2)
        modes.append(phi)

    return modes, S  # return SVD spectrum for diagnostics


# ═══════════════════════════════════════════════════════════════
# Route C: Fixed-pole constrained DMD
# ═══════════════════════════════════════════════════════════════

def fixed_pole_constrained_dmd(snapshots, snap_times, omega_arr, chi_R,
                                n_modes=5, rank=40, n_eff=None):
    """
    DMD with fixed eigenvalues λ_n = exp(-iω_n Δt_snap).

    Instead of eigendecomposing Ã, project known λ_n onto its spectrum
    by finding the approximate null vector of (Ã - λ_n I).
    """
    N = min(n_modes, len(omega_arr))
    Ns_all = len(snap_times)
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    dx2 = (L / n) ** 2
    dt_snap = snap_times[1] - snap_times[0]

    if n_eff is not None:
        Ns_all = min(Ns_all, n_eff)

    D = 2 * ny * nx
    X = np.zeros((D, Ns_all), dtype=complex)
    for k in range(Ns_all):
        X[:ny*nx, k] = (snapshots[k, 0] * chi_R).ravel()
        X[ny*nx:, k] = (snapshots[k, 1] * chi_R).ravel()

    X0 = X[:, :-1]
    X1 = X[:, 1:]

    # SVD of X0
    U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    r = min(rank, len(S))
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    # Reduced dynamics
    A_tilde = (U_r.conj().T @ X1) @ (Vh_r.conj().T * (1.0 / S_r)[None, :])

    # For each known pole, find approximate null vector of (Ã - λI)
    lam_forced = np.exp(-1j * omega_arr[:N] * dt_snap)
    W_forced = np.zeros((r, N), dtype=complex)
    for j in range(N):
        M = A_tilde - lam_forced[j] * np.eye(r)
        u_svd, s_svd, vh_svd = np.linalg.svd(M)
        W_forced[:, j] = vh_svd[-1, :].conj()  # smallest singular value

    # Reconstruct modes in full space
    Phi = X1 @ (Vh_r.conj().T * (1.0 / S_r)[None, :]) @ W_forced  # (D, N)

    modes = []
    for j in range(N):
        phi_flat = Phi[:, j]
        phi = np.zeros((2, ny, nx), dtype=complex)
        phi[0] = phi_flat[:ny*nx].reshape(ny, nx)
        phi[1] = phi_flat[ny*nx:].reshape(ny, nx)
        phi = _normalize_mask(phi, chi_R, dx2)
        modes.append(phi)

    return modes


# ═══════════════════════════════════════════════════════════════
# DMD baseline (copied from dmd_extract.py for self-contained use)
# ═══════════════════════════════════════════════════════════════

def exact_dmd(snapshots, snap_times, chi_cpu, rank=None, n_eff=None):
    """Exact DMD — baseline comparison."""
    Ns = len(snap_times)
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    dt_snap = snap_times[1] - snap_times[0]
    dx2 = (L / n) ** 2

    if n_eff is not None:
        Ns = min(Ns, n_eff)

    D = 2 * ny * nx
    X = np.zeros((D, Ns), dtype=complex)
    for k in range(Ns):
        X[:ny*nx, k] = (snapshots[k, 0] * chi_cpu).ravel()
        X[ny*nx:, k] = (snapshots[k, 1] * chi_cpu).ravel()

    X0 = X[:, :-1]; X1 = X[:, 1:]
    U, S, Vh = np.linalg.svd(X0, full_matrices=False)

    if rank is None:
        rank = max(2, int(np.sum(S / S[0] > 1e-3)))
        rank = min(rank, len(S) // 2)
    r = min(rank, len(S))
    U_r = U[:, :r]; S_r = S[:r]; Vh_r = Vh[:r, :]

    A_tilde = (U_r.conj().T @ X1) @ (Vh_r.conj().T * (1.0 / S_r)[None, :])
    lam, w = np.linalg.eig(A_tilde)

    with np.errstate(invalid='ignore', divide='ignore'):
        omega = 1j * np.log(lam + 0j) / dt_snap
    E_all = np.real(omega)
    Gamma_all = -2.0 * np.imag(omega)

    ok = (np.isfinite(E_all) & np.isfinite(Gamma_all) &
          (Gamma_all > 0) & (Gamma_all < Gamma_cut) &
          (np.abs(lam) < 1.0) & (np.abs(lam) > 0.01))

    Phi = X1 @ (Vh_r.conj().T * (1.0 / S_r)[None, :]) @ w
    E_phys = E_all[ok]; Gamma_phys = Gamma_all[ok]; omega_phys = omega[ok]
    order = np.argsort(Gamma_phys)
    E_phys = E_phys[order]; Gamma_phys = Gamma_phys[order]
    omega_phys = omega_phys[order]

    modes = []
    for idx in np.where(ok)[0][order]:
        phi_flat = Phi[:, idx]
        phi = np.zeros((2, ny, nx), dtype=complex)
        phi[0] = phi_flat[:ny*nx].reshape(ny, nx)
        phi[1] = phi_flat[ny*nx:].reshape(ny, nx)
        phi = _normalize_mask(phi, chi_cpu, dx2)
        modes.append(phi)

    return dict(E=E_phys, Gamma=Gamma_phys, omega=omega_phys,
                modes=modes, svd_spectrum=S, rank_used=r)


# ═══════════════════════════════════════════════════════════════
# Comparison engine
# ═══════════════════════════════════════════════════════════════

def match_overlap(modes, E_target, E_modes, ref_modes, chi, dx2,
                  E_tol=0.5):
    """Compute overlap between extracted modes and reference, matched by E."""
    overlaps = []
    for i, e_ref in enumerate(E_target):
        best = 0.0
        for j, m in enumerate(modes):
            if abs(E_modes[j] - e_ref) < E_tol:
                ov = spinor_overlap(m, ref_modes[i], chi, dx2)
                best = max(best, ov)
        overlaps.append(best)
    return overlaps


def compare_all(snapshots, snap_times, omega_mp, E_mp, G_mp,
                chi_cpu, dx2, tf_modes, label=""):
    """Run all methods on same data, print comparison table."""
    n_mp = len(omega_mp)
    print(f"\n{'='*72}")
    print(f"  Comparison: {label} ({n_mp} poles)")
    print(f"{'='*72}")

    results = {}

    # --- Baseline: bare Vandermonde lstsq ---
    t0 = time.perf_counter()
    modes_bare = reconstruct_modes_lstsq(snapshots, snap_times, omega_mp,
                                          chi_cpu, n_modes=n_mp)
    t_bare = time.perf_counter() - t0
    results["Bare LS"] = (modes_bare, t_bare)

    # --- Route A: Tikhonov Vandermonde ---
    t0 = time.perf_counter()
    modes_A = fixed_pole_recon_direct(snapshots, snap_times, omega_mp, chi_cpu,
                                       n_modes=n_mp, lambda_tik=1e-3)
    t_A = time.perf_counter() - t0
    results["Route A (λ=1e-3)"] = (modes_A, t_A)

    # --- Route B: POD + fixed pole ---
    t0 = time.perf_counter()
    modes_B, svd_S = fixed_pole_recon_pod(snapshots, snap_times, omega_mp,
                                           chi_cpu, n_modes=n_mp, rank=40,
                                           lambda_tik=1e-3)
    t_B = time.perf_counter() - t0
    results["Route B (r=40)"] = (modes_B, t_B)

    # --- Route B with Laplacian ---
    t0 = time.perf_counter()
    modes_BL, _ = fixed_pole_recon_pod(snapshots, snap_times, omega_mp,
                                        chi_cpu, n_modes=n_mp, rank=40,
                                        lambda_tik=1e-3, lambda_smooth=1e-4)
    t_BL = time.perf_counter() - t0
    results["Route B+Lap"] = (modes_BL, t_BL)

    # --- Route C: constrained DMD ---
    t0 = time.perf_counter()
    modes_C = fixed_pole_constrained_dmd(snapshots, snap_times, omega_mp,
                                          chi_cpu, n_modes=n_mp, rank=40)
    t_C = time.perf_counter() - t0
    results["Route C (r=40)"] = (modes_C, t_C)

    # --- DMD baseline ---
    t0 = time.perf_counter()
    dmd = exact_dmd(snapshots, snap_times, chi_cpu, rank=30, n_eff=150)
    t_dmd = time.perf_counter() - t0
    results["DMD (r=30)"] = (dmd['modes'], t_dmd)

    # Print table header
    print(f"\n{'Method':<20} {'time':>6}", end="")
    for i in range(n_mp):
        print(f"  E={E_mp[i]:.2f}", end="")
    print(f"  {'avg':>6}  {'C₂v_x':>6} {'C₂v_y':>6}")
    print("-" * (30 + 10 * n_mp + 20))

    for name, (modes, elapsed) in results.items():
        overlaps = []
        c2v_x_list, c2v_y_list = [], []
        for i in range(n_mp):
            best = 0.0
            for j, m in enumerate(modes):
                # For DMD modes, match by energy
                if name.startswith("DMD"):
                    if j < len(dmd['E']) and abs(dmd['E'][j] - E_mp[i]) < 0.5:
                        ov = spinor_overlap(m, tf_modes[i], chi_cpu, dx2)
                        best = max(best, ov)
                else:
                    # Fixed-pole methods: mode j corresponds to pole j
                    if j == i and j < len(modes):
                        best = spinor_overlap(modes[j], tf_modes[i], chi_cpu, dx2)
                        break
            overlaps.append(best)

            # C₂v for the i-th mode
            if i < len(modes):
                rho = np.abs(modes[i][0])**2 + np.abs(modes[i][1])**2
                xe, ye = c2v_err(rho)
                c2v_x_list.append(xe)
                c2v_y_list.append(ye)

        avg_ov = np.mean(overlaps) if overlaps else 0
        avg_cx = np.mean(c2v_x_list) if c2v_x_list else 0
        avg_cy = np.mean(c2v_y_list) if c2v_y_list else 0

        print(f"{name:<20} {elapsed:5.1f}s", end="")
        for ov in overlaps:
            print(f"  {ov:7.4f}", end="")
        print(f"  {avg_ov:6.4f}  {avg_cx:6.4f} {avg_cy:6.4f}")

    return results, svd_S


# ═══════════════════════════════════════════════════════════════
# Parameter scans
# ═══════════════════════════════════════════════════════════════

def lambda_scan(snapshots, snap_times, omega_mp, chi_cpu, dx2, tf_modes,
                route='A', lambdas=None, rank=40):
    """Scan Tikhonov parameter for Route A or B."""
    if lambdas is None:
        lambdas = np.logspace(-6, 0, 15)
    n_mp = len(omega_mp)
    all_avg = []

    print(f"\n--- λ scan (Route {route}) ---")
    print(f"{'λ':>12} {'avg_overlap':>12}  per-mode")

    for lam in lambdas:
        if route == 'A':
            modes = fixed_pole_recon_direct(snapshots, snap_times, omega_mp,
                                             chi_cpu, n_modes=n_mp,
                                             lambda_tik=lam)
        else:
            modes, _ = fixed_pole_recon_pod(snapshots, snap_times, omega_mp,
                                             chi_cpu, n_modes=n_mp, rank=rank,
                                             lambda_tik=lam)
        overlaps = []
        for i in range(n_mp):
            ov = spinor_overlap(modes[i], tf_modes[i], chi_cpu, dx2)
            overlaps.append(ov)
        avg = np.mean(overlaps)
        all_avg.append(avg)
        print(f"{lam:12.2e} {avg:12.4f}  [{', '.join(f'{o:.3f}' for o in overlaps)}]")

    return lambdas, all_avg


def rank_scan(snapshots, snap_times, omega_mp, chi_cpu, dx2, tf_modes,
              ranks=None, lambda_tik=1e-3):
    """Scan POD rank for Route B."""
    if ranks is None:
        ranks = [4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80]
    n_mp = len(omega_mp)
    all_avg = []

    print(f"\n--- Rank scan (Route B, λ={lambda_tik:.0e}) ---")
    print(f"{'rank':>6} {'avg_overlap':>12}  per-mode")

    for r in ranks:
        modes, _ = fixed_pole_recon_pod(snapshots, snap_times, omega_mp,
                                         chi_cpu, n_modes=n_mp, rank=r,
                                         lambda_tik=lambda_tik)
        overlaps = []
        for i in range(n_mp):
            ov = spinor_overlap(modes[i], tf_modes[i], chi_cpu, dx2)
            overlaps.append(ov)
        avg = np.mean(overlaps)
        all_avg.append(avg)
        print(f"{r:6d} {avg:12.4f}  [{', '.join(f'{o:.3f}' for o in overlaps)}]")

    return ranks, all_avg


# ═══════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════

def plot_comparison(results, E_mp, tf_modes, chi_cpu, dx2, label, dmd_result=None):
    """Bar chart of per-mode overlap for all methods."""
    n_mp = len(E_mp)
    methods = []
    overlap_data = []

    for name, (modes, _) in results.items():
        overlaps = []
        for i in range(n_mp):
            if name.startswith("DMD") and dmd_result is not None:
                best = 0.0
                for j, m in enumerate(modes):
                    if j < len(dmd_result['E']) and \
                       abs(dmd_result['E'][j] - E_mp[i]) < 0.5:
                        ov = spinor_overlap(m, tf_modes[i], chi_cpu, dx2)
                        best = max(best, ov)
                overlaps.append(best)
            else:
                if i < len(modes):
                    overlaps.append(spinor_overlap(modes[i], tf_modes[i],
                                                    chi_cpu, dx2))
                else:
                    overlaps.append(0.0)
        methods.append(name)
        overlap_data.append(overlaps)

    fig, ax = plt.subplots(figsize=(max(8, 2*n_mp), 5))
    x = np.arange(n_mp)
    w = 0.8 / len(methods)
    for i, (name, ovs) in enumerate(zip(methods, overlap_data)):
        ax.bar(x + i * w - 0.4 + w/2, ovs, w, label=name, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"E={e:.2f}" for e in E_mp], fontsize=8)
    ax.set_ylabel("Spinor overlap")
    ax.set_title(f"Mode reconstruction comparison ({label})")
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    tag = label.replace("=", "").replace(" ", "_").replace(".", "")
    fig.savefig(os.path.join(OUT, f"comparison_{tag}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Figure → {OUT}/comparison_{tag}.png")


def plot_scans(lam_A, avg_A, lam_B, avg_B, ranks, avg_rank, label):
    """Lambda and rank scan figures."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Lambda scan
    if lam_A is not None and len(lam_A) > 0:
        ax1.semilogx(lam_A, avg_A, 'o-', label='Route A (direct)')
    if lam_B is not None and len(lam_B) > 0:
        ax1.semilogx(lam_B, avg_B, 's-', label='Route B (POD)')
    ax1.set_xlabel("λ (Tikhonov)")
    ax1.set_ylabel("Avg overlap")
    ax1.set_title(f"Regularization scan ({label})")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Rank scan
    if ranks is not None and len(ranks) > 0:
        ax2.plot(ranks, avg_rank, 'D-', color='C2')
        ax2.set_xlabel("POD rank r")
        ax2.set_ylabel("Avg overlap")
        ax2.set_title(f"POD rank scan ({label})")
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    tag = label.replace("=", "").replace(" ", "_").replace(".", "")
    fig.savefig(os.path.join(OUT, f"scans_{tag}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Figure → {OUT}/scans_{tag}.png")


def plot_modes_gallery(modes, E_arr, chi_cpu, label, method_name):
    """Density gallery for top modes from best method."""
    n_show = min(6, len(modes))
    if n_show == 0:
        return
    extent = [-L/2, L/2, -L/2, L/2]

    fig, axes = plt.subplots(1, n_show, figsize=(3.5*n_show, 3.5))
    if n_show == 1:
        axes = [axes]
    for j in range(n_show):
        rho = np.abs(modes[j][0])**2 + np.abs(modes[j][1])**2
        xe, ye = c2v_err(rho)
        ax = axes[j]
        ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=rho.max())
        ax.set_title(f"E={E_arr[j]:.3f}\nx={xe:.3f} y={ye:.3f}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{method_name} modes ({label})", fontsize=12)
    plt.tight_layout()
    tag = label.replace("=", "").replace(" ", "_").replace(".", "")
    fig.savefig(os.path.join(OUT, f"modes_gallery_{tag}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Figure → {OUT}/modes_gallery_{tag}.png")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    dx2 = (L / n) ** 2
    dt_snap = dt * save_every

    for mass, label, E_lo, E_hi, G_hi in [
        (1.0, "m=1", 0.0, 10.0, 1.0),
        (0.0, "m=0", 2.0, 6.0,  2.0),
    ]:
        print(f"\n{'#'*72}")
        print(f"#  {label} — mass={mass}")
        print(f"{'#'*72}")

        # 1. Propagate
        kx0 = k_mag if mass > 0 else k_mag / np.sqrt(2)
        ky0 = 0.0 if mass > 0 else k_mag / np.sqrt(2)
        t0 = time.perf_counter()
        result, chi_cpu = run_prop(xp, mass, kx0, ky0,
                                    X, Y, KX, KY, K, dx, X_np, Y_np)
        print(f"Propagation: {time.perf_counter()-t0:.1f}s")

        snapshots = result["snapshots"]
        snap_times = result["snap_times"]
        print(f"Snapshots: {snapshots.shape}, T_max={snap_times[-1]:.1f}")

        # 2. Extract poles via matrix pencil
        spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
                                  Gamma_cut=Gamma_cut, pencil_fraction=4,
                                  L_pencil_max=400)
        good = (spec["E"] > E_lo) & (spec["E"] < E_hi) & (spec["Gamma"] < G_hi)
        E_mp = spec["E"][good]
        G_mp = spec["Gamma"][good]
        omega_mp = spec["omega"][good]
        n_mp = min(5, len(E_mp))
        E_mp = E_mp[:n_mp]; G_mp = G_mp[:n_mp]; omega_mp = omega_mp[:n_mp]

        print(f"\nMatrix Pencil: {n_mp} poles in [{E_lo},{E_hi}]")
        for j in range(n_mp):
            print(f"  E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}  Γ/ΔE~{G_mp[j]*3*pot_params['a']*pot_params['b']*E_mp[j]**2/(2*pot_params['k_harm']):.1f}")

        if n_mp == 0:
            print("No poles found, skipping")
            continue

        # 3. Time-filter reference modes
        tf_modes = [time_filter_mode(snapshots, snap_times, omega_mp[j], chi_cpu)
                    for j in range(n_mp)]

        # 4. Run all methods + comparison
        results, svd_S = compare_all(snapshots, snap_times, omega_mp, E_mp,
                                      G_mp, chi_cpu, dx2, tf_modes, label)

        # 5. Plot comparison
        dmd_r = exact_dmd(snapshots, snap_times, chi_cpu, rank=30, n_eff=150)
        plot_comparison(results, E_mp, tf_modes, chi_cpu, dx2, label, dmd_r)

        # 6. Parameter scans
        lam_A, avg_A = lambda_scan(snapshots, snap_times, omega_mp, chi_cpu,
                                    dx2, tf_modes, route='A')
        lam_B, avg_B = lambda_scan(snapshots, snap_times, omega_mp, chi_cpu,
                                    dx2, tf_modes, route='B', rank=40)
        ranks, avg_rank = rank_scan(snapshots, snap_times, omega_mp, chi_cpu,
                                     dx2, tf_modes)

        plot_scans(lam_A, avg_A, lam_B, avg_B, ranks, avg_rank, label)

        # 7. Modes gallery for best method (Route B)
        best_modes, _ = fixed_pole_recon_pod(snapshots, snap_times, omega_mp,
                                              chi_cpu, n_modes=n_mp, rank=40,
                                              lambda_tik=1e-3)
        plot_modes_gallery(best_modes, E_mp, chi_cpu, label, "Route B (POD)")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*72}")
    print(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
