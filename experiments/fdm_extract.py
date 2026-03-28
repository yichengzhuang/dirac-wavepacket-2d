#!/usr/bin/env python3
"""
Filter Diagonalization Method (FDM) for spatial mode extraction.

Reference: Mandelshtam & Taylor, J. Chem. Phys. 107, 6756 (1997)

Why FDM?
--------
Time filtering fails when Γ ≳ ΔE (massless Dirac): the Fourier bandwidth
is limited by Γ, so overlapping Lorentzians cannot be separated.
FDM uses the full spatial inner product B[j,k] = <Φ_j|Φ_k>: even when
two modes share the same energy, they can be separated if their spatial
structures differ.

Algorithm (multi-propagation, Option 2 in FDM_METHOD.md)
---------------------------------------------------------
1. Run M propagations from different launch directions.
2. From each propagation, build P_per energy-windowed probes:
       Φ_j(r) = Σ_k e^{iE_j t_k} χ_R(r) ψ(r, t_k)
3. Assemble P = M × P_per total probes into overlap matrix B (P×P)
   and evolution matrix A (P×P):
       B[j,k] = <Φ_j|Φ_k> dx²
       A[j,k] = <Φ_j|U Φ_k> dx²   (U shifts time by one snap step)
4. SVD-regularise B, solve generalised eigenvalue problem A c = z B c.
5. Eigenvalues z → ω = i log(z)/Δt_snap → E, Γ.
   Eigenvectors c → spatial modes φ_n = Σ_j c_j Φ_j.
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import svd as la_svd, eig as la_eig

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances


# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Grid
n  = 256
L  = 10.0

# Physics
vF = 1.0
m  = 0.0

# Time
dt         = 0.01
n_steps    = 15000    # T_max = 150  (longer → better freq resolution ΔE~0.04)
save_every = 10       # Δt_snap = 0.1  →  1501 snapshots
                      # Nyquist: E_max < π/Δt ≈ 31  ✓

# Potential
POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

# CAP & mask
cap_radius         = 3.5
cap_strength       = 10.0
cap_power          = 2
mask_safety_factor = 0.95
mask_margin        = 1.30

# Wavepacket
sigma0 = 0.6
k_mag  = 3.0

# ── FDM parameters ──
M_prop  = 4      # propagations (4 cardinal + diagonal directions)
P_per   = 8      # energy-windowed probes per propagation
P_total = M_prop * P_per  # 32 total probes

E_fdm_min = 1.0
E_fdm_max = 5.0

# SVD regularisation: drop singular values < eps_svd * sigma_max
eps_svd  = 1e-6

# Physical mode filter
Gamma_max = 8.0

# ── Launch directions ──
angles_prop = np.linspace(0, 2 * np.pi, M_prop, endpoint=False)

# ── Output ──
OUT_DIR = os.path.join(_root, "figures", "fdm")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def fft_flip(arr, axis):
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)


def measure_c2v_error(rho):
    rx = fft_flip(rho, axis=1)
    ry = fft_flip(rho, axis=0)
    x_err = np.sum(np.abs(rx - rho)) / (np.sum(rho) + 1e-30)
    y_err = np.sum(np.abs(ry - rho)) / (np.sum(rho) + 1e-30)
    return x_err, y_err


def build_probes(snapshots, snap_times, E_probes, chi_flat, dx2):
    """
    Build energy-windowed probes and their time-shifted versions.

    Returns
    -------
    probes         : (P, 2, n*n)  complex  —  Σ_m e^{iE_j t_m} χ ψ(t_m)
    probes_shifted : (P, 2, n*n)  complex  —  Σ_m e^{iE_j t_m} χ ψ(t_{m+1})
    """
    Ns = len(snap_times)
    P  = len(E_probes)
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    N_sp = ny * nx

    # phases[m, j] = e^{i E_j t_m}
    phases = np.exp(1j * np.outer(snap_times,       E_probes))  # (Ns, P)
    phases_minus1 = np.exp(1j * np.outer(snap_times[:-1], E_probes))  # (Ns-1, P)

    # Flatten + mask snapshots
    snap_flat = snapshots.reshape(Ns, 2, N_sp)          # (Ns, 2, N_sp)
    snap_chi  = snap_flat * chi_flat[None, None, :]      # (Ns, 2, N_sp)

    # probes[j, c, i] = Σ_m phases[m, j] * snap_chi[m, c, i]
    # = (phases.T) @ snap_chi.reshape(Ns, 2*N_sp)
    S = snap_chi.reshape(Ns, 2 * N_sp)                  # (Ns, 2N_sp)
    probes_flat = (phases.T @ S).reshape(P, 2, N_sp)    # (P, 2, N_sp)

    S_shift = snap_chi[1:].reshape(Ns - 1, 2 * N_sp)
    probes_shifted_flat = (phases_minus1.T @ S_shift).reshape(P, 2, N_sp)

    return probes_flat, probes_shifted_flat


def build_BA(all_probes, all_probes_shifted, dx2):
    """
    Assemble total B and A matrices from all propagations.

    all_probes         : list of (P_per, 2, N_sp)
    all_probes_shifted : list of (P_per, 2, N_sp)
    """
    probes_full  = np.concatenate(all_probes,         axis=0)  # (P, 2, N_sp)
    shifted_full = np.concatenate(all_probes_shifted, axis=0)

    # B[j,k] = dx² Σ_{c,i} conj(Φ_j[c,i]) Φ_k[c,i]
    pf = probes_full.reshape(len(probes_full), -1)    # (P, 2*N_sp)
    sf = shifted_full.reshape(len(shifted_full), -1)

    B = (pf.conj() @ pf.T) * dx2
    A = (pf.conj() @ sf.T) * dx2

    return B, A


def solve_fdm(B, A, dt_snap, eps_svd=1e-6, Gamma_max=8.0):
    """
    Solve generalised eigenvalue problem with SVD regularisation.

    Returns
    -------
    E_fdm    : real frequencies of physical poles
    Gamma_fdm: decay widths (> 0)
    vecs     : eigenvectors (columns), corresponding to physical poles
    ok       : boolean mask into original eigenvalue array
    """
    # SVD regularise B
    U_b, s_b, Vh_b = la_svd(B)
    s_max = s_b[0]
    keep  = s_b > eps_svd * s_max
    n_keep = keep.sum()
    print(f"  SVD: {B.shape[0]} probes, keeping {n_keep} singular vectors "
          f"(eps={eps_svd:.0e}, s_min/s_max={s_b[keep][-1]/s_max:.2e})")

    # Build B^{-1/2}
    s_inv_sqrt = 1.0 / np.sqrt(s_b[keep])
    # B = U S V†  →  B^{-1/2} = U S^{-1/2} U†  (B is hermitian → U=V)
    B_inv_half = (U_b[:, keep] * s_inv_sqrt[None, :]) @ U_b[:, keep].conj().T
    B_half     = (U_b[:, keep] * np.sqrt(s_b[keep])[None, :]) @ U_b[:, keep].conj().T

    # Reduce to standard eigenvalue problem: (B^{-1/2} A B^{-1/2}) c' = z c'
    A_red = B_inv_half @ A @ B_inv_half

    z_all, vecs_red = np.linalg.eig(A_red)

    # Back-transform eigenvectors: c = B^{-1/2} c'
    vecs_all = B_inv_half @ vecs_red

    # z → ω = i log(z) / Δt_snap
    with np.errstate(invalid='ignore'):
        omega_all = 1j * np.log(z_all + 0j) / dt_snap

    E_all     = np.real(omega_all)
    Gamma_all = -2.0 * np.imag(omega_all)

    # Physical filter
    ok = (
        np.isfinite(E_all) &
        np.isfinite(Gamma_all) &
        (Gamma_all > 0) &
        (Gamma_all < Gamma_max) &
        (np.abs(z_all) < 1.0)
    )
    return E_all[ok], Gamma_all[ok], vecs_all[:, ok], ok


def reconstruct_modes(vecs, all_probes, n, chi_cpu):
    """
    Build spatial modes from FDM eigenvectors.

    vecs      : (P, n_modes)
    all_probes: list of (P_per, 2, N_sp)

    Returns list of (2, n, n) arrays (normalised inside chi_R mask).
    """
    probes_full = np.concatenate(all_probes, axis=0)    # (P, 2, N_sp)
    N_sp = probes_full.shape[2]
    modes = []
    for col in range(vecs.shape[1]):
        c   = vecs[:, col]                              # (P,)
        phi = np.tensordot(c, probes_full, axes=[[0], [0]])  # (2, N_sp)
        phi = phi.reshape(2, n, n)
        # normalise inside mask
        rho  = np.abs(phi[0])**2 + np.abs(phi[1])**2
        norm = np.sqrt(np.sum(rho * chi_cpu) * (L / n)**2)
        if norm > 0:
            phi = phi / norm
        modes.append(phi)
    return modes


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(E_fdm, Gamma_fdm, modes, chi_cpu, E_mp=None, Gamma_mp=None):
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]
    n_modes = len(modes)

    # ── spectrum ──
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.vlines(E_fdm, 0, Gamma_fdm, colors="C0", lw=2, label="FDM poles")
    if E_mp is not None:
        ax.vlines(E_mp, 0, Gamma_mp, colors="C1", lw=2, linestyle="--",
                  label="Matrix pencil")
    ax.set_xlabel("E (code units)")
    ax.set_ylabel("Γ")
    ax.set_title("FDM extracted resonances — massless harmonic ellipse")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fdm_spectrum.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  spectrum  → {out}")

    # ── spatial modes ──
    ncols = min(n_modes, 4)
    nrows = (n_modes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).ravel()
    for j, phi in enumerate(modes):
        ax  = axes[j]
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        rho_masked = rho * chi_cpu
        im  = ax.imshow(rho_masked, extent=extent, origin="lower",
                        cmap="inferno", vmin=0, vmax=rho_masked.max())
        x_e, y_e = measure_c2v_error(rho_masked)
        ax.set_title(
            f"E={E_fdm[j]:.3f}  Γ={Gamma_fdm[j]:.3f}\n"
            f"x_err={x_e:.3f}  y_err={y_e:.3f}",
            fontsize=9,
        )
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
    for j in range(n_modes, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"FDM spatial modes — massless harmonic ellipse, M={M_prop} prop.",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fdm_modes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  modes     → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    xp = get_backend("auto")
    print(f"Backend : {xp.name}")
    print(f"Grid    : {n}×{n},  L={L}")
    print(f"Propagations: {M_prop},  P_per={P_per},  P_total={P_total}")
    print(f"T_max = {n_steps*dt:.0f},  Δt_snap = {dt*save_every:.3f}")
    print(f"FDM E range: [{E_fdm_min:.2f}, {E_fdm_max:.2f}]  (P_per={P_per} probes)")
    print(f"Out dir : {OUT_DIR}\n")

    # ── one-time setup ──
    x_np       = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    dx_np = float(L / n)
    dx2   = dx_np**2

    V      = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W      = radial_CAP(X_np, Y_np,
                        cap_radius=cap_radius,
                        cap_strength=cap_strength,
                        cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(
        POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius,
        safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp,
    )
    chi_cpu  = xp.to_numpy(chi_R)
    chi_flat = chi_cpu.ravel()
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)

    # Energy grid for probes
    E_probes = np.linspace(E_fdm_min, E_fdm_max, P_per)

    # ── propagation loop ──
    all_probes         = []
    all_probes_shifted = []

    # also collect correlation for matrix-pencil comparison
    all_CR        = []
    all_CR_times  = []
    dt_corr = dt * save_every

    t_wall_start = time.perf_counter()

    for ip, angle in enumerate(angles_prop):
        kx0 = k_mag * np.cos(angle)
        ky0 = k_mag * np.sin(angle)
        print(f"\n[{ip+1}/{M_prop}]  k=({kx0:+.2f},{ky0:+.2f})", flush=True)
        t0 = time.perf_counter()

        psi0_up, psi0_dn = gaussian_wavepacket(
            X, Y, x0=0.0, y0=0.0, kx0=kx0, ky0=ky0,
            sigma=sigma0, theta=None, band="conduction",
            dx=dx, xp=xp,
        )

        result = propagate(
            psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
            n_steps=n_steps, dt=dt,
            save_every=save_every,
            corr_every=save_every,
            max_record=n_steps // save_every + 2,
            measure_transmission=False,
        )

        snapshots  = result["snapshots"]
        snap_times = result["snap_times"]

        # collect correlation for matrix pencil
        all_CR.append(result["C_R"])
        all_CR_times.append(result["corr_times"])

        # build probes
        probes, probes_shifted = build_probes(
            snapshots, snap_times, E_probes, chi_flat, dx2)
        all_probes.append(probes)
        all_probes_shifted.append(probes_shifted)

        del snapshots, result, psi0_up, psi0_dn

        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f} s", flush=True)

    # ── matrix pencil on first propagation's C_R (for comparison) ──
    print("\nMatrix pencil (propagation 1) …")
    spec_mp = extract_resonances(
        all_CR[0], all_CR_times[0], dt_corr,
        Gamma_cut=Gamma_max, pencil_fraction=4, L_pencil_max=400)
    E_mp, Gamma_mp = spec_mp["E"], spec_mp["Gamma"]
    print("  Matrix pencil results:")
    for j in range(min(6, len(E_mp))):
        print(f"    E={E_mp[j]:.4f}  Γ={Gamma_mp[j]:.4f}")

    # ── FDM ──
    print(f"\nBuilding B, A matrices (P={P_total}) …")
    B, A = build_BA(all_probes, all_probes_shifted, dx2)

    print("Solving FDM eigenvalue problem …")
    E_fdm, Gamma_fdm, vecs, ok = solve_fdm(B, A, dt_corr,
                                            eps_svd=eps_svd,
                                            Gamma_max=Gamma_max)

    # Sort by |Γ| (narrowest first = best-confined modes)
    order     = np.argsort(Gamma_fdm)
    E_fdm     = E_fdm[order]
    Gamma_fdm = Gamma_fdm[order]
    vecs      = vecs[:, order]

    print(f"\nFDM extracted {len(E_fdm)} physical poles:")
    print(f"  {'E':>8}  {'Γ':>8}")
    for j in range(min(10, len(E_fdm))):
        print(f"  {E_fdm[j]:8.4f}  {Gamma_fdm[j]:8.4f}")

    # ── save arrays ──
    np.save(os.path.join(OUT_DIR, "fdm_E.npy"),     E_fdm)
    np.save(os.path.join(OUT_DIR, "fdm_Gamma.npy"), Gamma_fdm)
    np.save(os.path.join(OUT_DIR, "mp_E.npy"),      E_mp)
    np.save(os.path.join(OUT_DIR, "mp_Gamma.npy"),  Gamma_mp)

    # ── reconstruct top-N spatial modes ──
    N_show = min(8, len(E_fdm))
    print(f"\nReconstructing {N_show} spatial modes …")
    modes = reconstruct_modes(vecs[:, :N_show], all_probes, n, chi_cpu)

    # ── C₂v report ──
    print("\nC₂v symmetry of FDM modes (narrowest first):")
    print(f"  {'E':>8}  {'Γ':>8}  {'x_err':>8}  {'y_err':>8}")
    for j, phi in enumerate(modes):
        rho = (np.abs(phi[0])**2 + np.abs(phi[1])**2) * chi_cpu
        x_e, y_e = measure_c2v_error(rho)
        print(f"  {E_fdm[j]:8.4f}  {Gamma_fdm[j]:8.4f}  {x_e:8.4f}  {y_e:8.4f}")

    # ── figures ──
    print("\nGenerating figures …")
    make_figures(E_fdm[:N_show], Gamma_fdm[:N_show], modes,
                 chi_cpu, E_mp, Gamma_mp)

    np.save(os.path.join(OUT_DIR, "fdm_modes.npy"),
            np.stack([(np.abs(m[0])**2 + np.abs(m[1])**2) * chi_cpu
                      for m in modes]))

    total = time.perf_counter() - t_wall_start
    print(f"\nTotal wall time: {total:.1f} s  ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
