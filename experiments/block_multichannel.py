#!/usr/bin/env python3
"""
Multi-channel block matrix pencil and block FDM for spatial mode reconstruction.

Core idea: instead of compressing ψ(x,y,t) into a single scalar C_R(t),
record M linearly independent time traces:

    a_μ(t) = <g_μ|ψ(t)>    μ = 1, ..., M

then solve the block problem:

    a(t) = Σ_n v_n e^{-iω_n t}

where v_n ∈ C^M is the spatial fingerprint of mode n.  Even when ω_n are
close, modes can be separated if their v_n differ.

Probe strategies tested:
  (A) Spatial patches: divide billiard into ~20 regions, A/B sublattice separate
  (B) SVD reduced basis: POD from sparse snapshots, project onto top-M modes

Methods tested:
  (1) Block matrix pencil (block Hankel)
  (2) Block FDM (generalized eigenvalue with spatial inner products)
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

OUT_DIR = os.path.join(_root, "figures", "fdm_debug")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters ──
n = 256; L = 10.0; vF = 1.0
dt = 0.01; n_steps = 15000; save_every = 10   # T=150, Δt=0.1, 1501 snaps
POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5; cap_strength = 10.0; cap_power = 2
mask_safety_factor = 0.95; mask_margin = 1.30
sigma0 = 0.6; k_mag = 3.0
Gamma_cut = 8.0; pencil_fraction = 4; L_pencil_max = 400


def fft_flip(arr, axis):
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)

def c2v_err(rho):
    s = np.sum(rho) + 1e-30
    return (np.sum(np.abs(fft_flip(rho, 1) - rho)) / s,
            np.sum(np.abs(fft_flip(rho, 0) - rho)) / s)

def spinor_overlap(a, b, chi, dx2):
    ab = (np.vdot(a[0]*chi, b[0]) + np.vdot(a[1]*chi, b[1])) * dx2
    aa = (np.vdot(a[0]*chi, a[0]) + np.vdot(a[1]*chi, a[1])) * dx2
    bb = (np.vdot(b[0]*chi, b[0]) + np.vdot(b[1]*chi, b[1])) * dx2
    d = np.real(aa) * np.real(bb)
    return float(np.abs(ab)**2 / d) if d > 1e-30 else 0.0


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Probe construction                                              ║
# ╚═══════════════════════════════════════════════════════════════════╝

def make_patch_probes(chi_cpu, nn, n_rings=3, n_angular=4):
    """
    Spatial patch probes: divide the billiard interior into regions.
    Returns list of (2*n_patches) probe masks — A and B sublattice separate.
    """
    ny, nx = nn, nn
    x = np.linspace(-L/2, L/2, nx, endpoint=False)
    y = np.linspace(-L/2, L/2, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Elliptical coordinates
    a, b = pot_params["a"], pot_params["b"]
    rho = np.sqrt((X/a)**2 + (Y/b)**2)
    theta = np.arctan2(Y/b, X/a)

    patches = []
    r_edges = np.linspace(0, 1.3, n_rings + 1)  # up to mask boundary
    a_edges = np.linspace(-np.pi, np.pi, n_angular + 1)

    for ir in range(n_rings):
        for ia in range(n_angular):
            mask = ((rho >= r_edges[ir]) & (rho < r_edges[ir+1]) &
                    (theta >= a_edges[ia]) & (theta < a_edges[ia+1]) &
                    (chi_cpu > 0.5))
            if mask.sum() > 10:
                patches.append(mask.astype(float))

    # Also add whole-mask probes for A and B separately (important!)
    patches.append(chi_cpu.copy())  # full mask

    return patches


def make_svd_probes(snapshots, chi_cpu, n_basis=30, n_sample=50):
    """
    SVD/POD reduced basis from sparse snapshots.
    Returns list of n_basis probe functions (2, ny, nx).
    """
    Ns = len(snapshots)
    idx = np.linspace(0, min(Ns-1, 200), n_sample, dtype=int)

    # Stack masked snapshots: (n_sample, 2*ny*nx)
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    data = np.zeros((len(idx), 2 * ny * nx), dtype=complex)
    for k, i in enumerate(idx):
        s = snapshots[i] * chi_cpu[None]
        data[k] = s.reshape(-1)

    # SVD
    U, S, Vh = np.linalg.svd(data, full_matrices=False)
    n_use = min(n_basis, len(S))

    print(f"  SVD: top singular values: {np.array2string(S[:5]/S[0], precision=4)}")
    print(f"  Keeping {n_use} basis vectors (explains "
          f"{np.sum(S[:n_use]**2)/np.sum(S**2)*100:.1f}% of variance)")

    # Each row of Vh is a spatial basis function
    basis = []
    for j in range(n_use):
        u = Vh[j].reshape(2, ny, nx)
        basis.append(u)

    return basis, S


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Project snapshots onto probes → time traces                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

def project_patches(snapshots, snap_times, patches, chi_cpu, dx2):
    """
    Project snapshots onto spatial patches.
    Returns C_block: (M, Ns) array where M = 2*len(patches) (A and B separate).
    """
    Ns = len(snap_times)
    M = 2 * len(patches)  # A and B sublattice
    C_block = np.zeros((M, Ns), dtype=complex)

    for k in range(Ns):
        psi_A = snapshots[k, 0]
        psi_B = snapshots[k, 1]
        for p, patch in enumerate(patches):
            C_block[2*p,   k] = np.sum(patch * psi_A) * dx2
            C_block[2*p+1, k] = np.sum(patch * psi_B) * dx2

    return C_block


def project_svd(snapshots, snap_times, basis, chi_cpu, dx2):
    """
    Project snapshots onto SVD basis functions.
    Returns C_block: (M, Ns) complex array.
    """
    Ns = len(snap_times)
    M = len(basis)
    C_block = np.zeros((M, Ns), dtype=complex)

    for k in range(Ns):
        s = snapshots[k] * chi_cpu[None]  # (2, ny, nx)
        s_flat = s.reshape(-1)
        for mu in range(M):
            C_block[mu, k] = np.vdot(basis[mu].reshape(-1), s_flat) * dx2

    return C_block


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Block matrix pencil                                             ║
# ╚═══════════════════════════════════════════════════════════════════╝

def block_matrix_pencil(C_block, dt_snap, L_pencil=None, eps_svd=1e-4):
    """
    Correct block matrix pencil for vector signal C_block: (M, Ns).

    The signal decomposes as:
        a_μ(t_k) = Σ_n v_{n,μ} z_n^k    where z_n = e^{-iω_n Δt}

    Build block Hankel with M×M blocks from the signal vectors directly:
        H0[i*M:(i+1)*M, j] = a(t_{i+j})    (M-vector at lag i+j)

    This is the "trajectory matrix" approach (equivalent to multivariate SSA).
    The key: H0 has shape (nL*M, nR) where each column j contains the
    trajectory [a(j), a(j+1), ..., a(j+nL-1)] stacked as M-vectors.

    SVD of H0 captures the signal subspace; the shift H0→H1 encodes z_n.
    Solve in the rank-r subspace for stability.
    """
    M, Ns = C_block.shape
    K = Ns - 1

    if L_pencil is None:
        L_pencil = min(K // 4, 100)  # smaller L for stability

    nL = min(L_pencil, (K - 1) // 2)
    nR = K - nL

    # ── Build trajectory matrix (block Hankel) ──
    # H0[i*M:(i+1)*M, j] = C_block[:, i+j] for i=0..nL-1, j=0..nR-1
    # H1 is the time-shifted version: H1[i*M:(i+1)*M, j] = C_block[:, i+j+1]
    H0 = np.zeros((nL * M, nR), dtype=complex)
    H1 = np.zeros((nL * M, nR), dtype=complex)

    for i in range(nL):
        H0[i*M:(i+1)*M, :] = C_block[:, i:i+nR]
        H1[i*M:(i+1)*M, :] = C_block[:, i+1:i+1+nR]

    # ── SVD of H0 ──
    U, S, Vh = np.linalg.svd(H0, full_matrices=False)
    tol = eps_svd * S[0]
    r = int(np.sum(S > tol))
    r = max(r, 2)
    print(f"  Block MP: H0 size {H0.shape}, SVD rank {r}/{min(H0.shape)} "
          f"(eps={eps_svd:.0e}, s_min/s_max={S[r-1]/S[0]:.2e})")

    U_r = U[:, :r]; S_r = S[:r]; Vh_r = Vh[:r, :]

    # ── Solve in rank-r subspace: Z_red = S^{-1} U^H H1 V^H ──
    Z_red = np.diag(1.0 / S_r) @ (U_r.conj().T @ H1 @ Vh_r.conj().T)
    z, w = np.linalg.eig(Z_red)

    # z → ω
    omega = 1j * np.log(z + 0j) / dt_snap
    E_all = np.real(omega)
    Gamma_all = -2.0 * np.imag(omega)

    ok = (np.isfinite(E_all) & np.isfinite(Gamma_all) &
          (Gamma_all > 0) & (Gamma_all < Gamma_cut) & (np.abs(z) < 1.0))

    # ── Channel fingerprints: first M rows of U_r @ w ──
    W_full = U_r @ w  # (nL*M, r)
    V_modes = W_full[:M, ok]

    return dict(E=E_all[ok], Gamma=Gamma_all[ok], omega=omega[ok],
                V_modes=V_modes, z_all=z, svd_spectrum=S, rank=r)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Reconstruct spatial modes from channel fingerprints             ║
# ╚═══════════════════════════════════════════════════════════════════╝

def reconstruct_from_patches(V_modes, patches, nn, chi_cpu, dx2):
    """Reconstruct 2D modes from patch fingerprints."""
    n_modes = V_modes.shape[1]
    modes = []
    for col in range(n_modes):
        v = V_modes[:, col]  # (2*n_patches,) — A and B for each patch
        phi = np.zeros((2, nn, nn), dtype=complex)
        for p in range(len(patches)):
            if 2*p < len(v):
                phi[0] += v[2*p] * patches[p]
            if 2*p+1 < len(v):
                phi[1] += v[2*p+1] * patches[p]
        # normalize
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        norm = np.sqrt(np.sum(rho * chi_cpu) * dx2)
        if norm > 1e-30:
            phi /= norm
        modes.append(phi)
    return modes


def reconstruct_from_svd(V_modes, basis, nn, chi_cpu, dx2):
    """Reconstruct 2D modes from SVD basis coefficients."""
    n_modes = V_modes.shape[1]
    modes = []
    for col in range(n_modes):
        v = V_modes[:, col]  # (M,) coefficients
        phi = np.zeros((2, nn, nn), dtype=complex)
        for mu in range(min(len(v), len(basis))):
            phi += v[mu] * basis[mu]
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        norm = np.sqrt(np.sum(rho * chi_cpu) * dx2)
        if norm > 1e-30:
            phi /= norm
        modes.append(phi)
    return modes


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Main                                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝

def main():
    t_total = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx_np = L / n; dx2 = dx_np**2

    chi_R, _ = build_analysis_mask(POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)

    dt_snap = dt * save_every

    # ══════════════════════════════════════════════════════════════
    # Test on both m=0 and m=0.5
    # ══════════════════════════════════════════════════════════════

    for mass_val in [0.5, 0.0]:
        print(f"\n{'='*70}")
        print(f"  MASS = {mass_val}")
        print(f"{'='*70}")

        # ── propagate ──
        kx0 = k_mag / np.sqrt(2); ky0 = k_mag / np.sqrt(2)
        print(f"Propagating (45° launch) ...")
        V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
        W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                       cap_strength=cap_strength, cap_power=cap_power, xp=xp)
        sqrt_chi = xp.sqrt(chi_R)
        halfV, kin = build_split_operator(vF, dt, mass_val, V, W, KX, KY, K, xp)
        psi0_up, psi0_dn = gaussian_wavepacket(
            X, Y, x0=0.0, y0=0.0, kx0=kx0, ky0=ky0,
            sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
        result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
            n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
            max_record=n_steps//save_every+2, measure_transmission=False)

        snapshots = result["snapshots"]
        snap_times = result["snap_times"]
        psi0_cpu = np.stack([xp.to_numpy(psi0_up), xp.to_numpy(psi0_dn)])

        # ── scalar MP (reference) ──
        print("Scalar matrix pencil (reference) ...")
        spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
            Gamma_cut=Gamma_cut, pencil_fraction=pencil_fraction,
            L_pencil_max=L_pencil_max)
        E_mp = spec["E"]; G_mp = spec["Gamma"]; omega_mp = spec["omega"]
        good = (E_mp > 0) & (G_mp < 2.0)
        E_mp, G_mp, omega_mp = E_mp[good], G_mp[good], omega_mp[good]
        n_mp = min(8, len(E_mp))
        print(f"  Scalar MP: {len(E_mp)} poles. Top {n_mp}:")
        for j in range(n_mp):
            print(f"    E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

        # ── time-filter reference modes ──
        print("Time filtering reference modes ...")
        tf_modes = [time_filter_mode(snapshots, snap_times, omega_mp[j], chi_cpu)
                    for j in range(n_mp)]

        # ═════════════════════════════════════════════
        # Strategy A: Spatial patches
        # ═════════════════════════════════════════════
        print("\n--- Strategy A: Spatial patches ---")
        patches = make_patch_probes(chi_cpu, n, n_rings=3, n_angular=6)
        print(f"  {len(patches)} patches, {2*len(patches)} channels (A+B)")

        C_patch = project_patches(snapshots, snap_times, patches, chi_cpu, dx2)
        print(f"  C_block shape: {C_patch.shape}")

        res_patch = block_matrix_pencil(C_patch, dt_snap, eps_svd=1e-4)
        E_bp = res_patch["E"]; G_bp = res_patch["Gamma"]

        # Filter to physical energy range first, then sort by Γ
        E_lo = max(mass_val + 0.5, 1.5); E_hi = 6.0
        phys = (E_bp > E_lo) & (E_bp < E_hi) & (G_bp < 2.0)
        if phys.sum() < 2:
            phys = (E_bp > 0) & (E_bp < 8.0) & (G_bp < 2.0)
        E_bp = E_bp[phys]; G_bp = G_bp[phys]
        res_patch["V_modes"] = res_patch["V_modes"][:, phys] if res_patch["V_modes"].shape[1] > 0 else res_patch["V_modes"]
        n_bp = min(10, len(E_bp))

        order = np.argsort(G_bp)[:n_bp]
        E_bp, G_bp = E_bp[order], G_bp[order]
        print(f"  Block MP (patches): {len(res_patch['E'])} poles. Top {n_bp}:")
        for j in range(n_bp):
            print(f"    E={E_bp[j]:.4f}  Γ={G_bp[j]:.4f}")

        # reconstruct modes
        modes_patch = reconstruct_from_patches(
            res_patch["V_modes"][:, order], patches, n, chi_cpu, dx2)

        # compare with TF
        print("  Overlap with time-filtered modes:")
        for j in range(min(n_bp, n_mp)):
            # find nearest MP pole
            dists = np.abs(E_bp[j] - E_mp[:n_mp])
            i_best = np.argmin(dists)
            if j < len(modes_patch) and i_best < len(tf_modes):
                ovlp = spinor_overlap(modes_patch[j], tf_modes[i_best], chi_cpu, dx2)
            else:
                ovlp = 0.0
            print(f"    Block E={E_bp[j]:.4f} ↔ MP E={E_mp[i_best]:.4f}  "
                  f"ΔE={dists[i_best]:.4f}  overlap={ovlp:.4f}")

        # ═════════════════════════════════════════════
        # Strategy B: SVD reduced basis
        # ═════════════════════════════════════════════
        print("\n--- Strategy B: SVD reduced basis ---")
        basis, S_svd = make_svd_probes(snapshots, chi_cpu, n_basis=30, n_sample=60)

        C_svd = project_svd(snapshots, snap_times, basis, chi_cpu, dx2)
        print(f"  C_block shape: {C_svd.shape}")

        res_svd = block_matrix_pencil(C_svd, dt_snap, eps_svd=1e-4)
        E_bs = res_svd["E"]; G_bs = res_svd["Gamma"]

        phys_s = (E_bs > E_lo) & (E_bs < E_hi) & (G_bs < 2.0)
        if phys_s.sum() < 2:
            phys_s = (E_bs > 0) & (E_bs < 8.0) & (G_bs < 2.0)
        E_bs = E_bs[phys_s]; G_bs = G_bs[phys_s]
        res_svd["V_modes"] = res_svd["V_modes"][:, phys_s] if res_svd["V_modes"].shape[1] > 0 else res_svd["V_modes"]
        n_bs = min(10, len(E_bs))

        order_s = np.argsort(G_bs)[:n_bs]
        E_bs, G_bs = E_bs[order_s], G_bs[order_s]
        print(f"  Block MP (SVD): {len(res_svd['E'])} poles. Top {n_bs}:")
        for j in range(n_bs):
            print(f"    E={E_bs[j]:.4f}  Γ={G_bs[j]:.4f}")

        modes_svd = reconstruct_from_svd(
            res_svd["V_modes"][:, order_s], basis, n, chi_cpu, dx2)

        print("  Overlap with time-filtered modes:")
        for j in range(min(n_bs, n_mp)):
            dists = np.abs(E_bs[j] - E_mp[:n_mp])
            i_best = np.argmin(dists)
            if j < len(modes_svd) and i_best < len(tf_modes):
                ovlp = spinor_overlap(modes_svd[j], tf_modes[i_best], chi_cpu, dx2)
            else:
                ovlp = 0.0
            print(f"    Block E={E_bs[j]:.4f} ↔ MP E={E_mp[i_best]:.4f}  "
                  f"ΔE={dists[i_best]:.4f}  overlap={ovlp:.4f}")

        # ═════════════════════════════════════════════
        # Figures
        # ═════════════════════════════════════════════
        tag = f"m{mass_val:.1f}".replace(".", "p")
        half_L = L / 2
        extent = [-half_L, half_L, -half_L, half_L]

        # Pole comparison
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(E_mp[:n_mp], G_mp[:n_mp], s=80, marker='o',
                   facecolors='none', edgecolors='C0', lw=2,
                   label='Scalar MP', zorder=3)
        if len(E_bp) > 0:
            ax.scatter(E_bp[:n_bp], G_bp[:n_bp], s=40, marker='x',
                       color='C1', lw=2, label='Block MP (patches)', zorder=3)
        if len(E_bs) > 0:
            ax.scatter(E_bs[:n_bs], G_bs[:n_bs], s=40, marker='+',
                       color='C2', lw=2, label='Block MP (SVD)', zorder=3)
        ax.set_xlabel("E"); ax.set_ylabel(r"$\Gamma$")
        ax.set_title(f"Pole comparison — m={mass_val}")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"block_{tag}_poles.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Poles → {out}")

        # Mode comparison: top-4 block MP (patches) vs TF
        n_show = min(4, len(modes_patch), n_mp)
        if n_show > 0:
            fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
            if n_show == 1:
                axes = axes[:, np.newaxis]
            for j in range(n_show):
                # block mode
                rho_b = np.abs(modes_patch[j][0])**2 + np.abs(modes_patch[j][1])**2
                axes[0, j].imshow(rho_b * chi_cpu, extent=extent,
                                  origin="lower", cmap="inferno")
                xe, ye = c2v_err(rho_b * chi_cpu)
                axes[0, j].set_title(f"Block patch\nE={E_bp[j]:.3f} Γ={G_bp[j]:.3f}\n"
                                     f"x={xe:.3f} y={ye:.3f}", fontsize=8)
                axes[0, j].set_aspect("equal")

                # TF reference
                i_best = np.argmin(np.abs(E_bp[j] - E_mp[:n_mp]))
                rho_t = np.abs(tf_modes[i_best][0])**2 + np.abs(tf_modes[i_best][1])**2
                axes[1, j].imshow(rho_t, extent=extent,
                                  origin="lower", cmap="inferno")
                axes[1, j].set_title(f"Time filter\nE={E_mp[i_best]:.3f} "
                                     f"Γ={G_mp[i_best]:.3f}", fontsize=8)
                axes[1, j].set_aspect("equal")

            fig.suptitle(f"Block MP (patches) vs Time filter — m={mass_val}",
                         fontsize=12)
            plt.tight_layout()
            out = os.path.join(OUT_DIR, f"block_{tag}_modes_patch.png")
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  Patch modes → {out}")

        # SVD modes
        n_show_s = min(4, len(modes_svd), n_mp)
        if n_show_s > 0:
            fig, axes = plt.subplots(2, n_show_s, figsize=(4*n_show_s, 8))
            if n_show_s == 1:
                axes = axes[:, np.newaxis]
            for j in range(n_show_s):
                rho_b = np.abs(modes_svd[j][0])**2 + np.abs(modes_svd[j][1])**2
                axes[0, j].imshow(rho_b * chi_cpu, extent=extent,
                                  origin="lower", cmap="inferno")
                xe, ye = c2v_err(rho_b * chi_cpu)
                axes[0, j].set_title(f"Block SVD\nE={E_bs[j]:.3f} Γ={G_bs[j]:.3f}\n"
                                     f"x={xe:.3f} y={ye:.3f}", fontsize=8)
                axes[0, j].set_aspect("equal")

                i_best = np.argmin(np.abs(E_bs[j] - E_mp[:n_mp]))
                rho_t = np.abs(tf_modes[i_best][0])**2 + np.abs(tf_modes[i_best][1])**2
                axes[1, j].imshow(rho_t, extent=extent,
                                  origin="lower", cmap="inferno")
                axes[1, j].set_title(f"Time filter\nE={E_mp[i_best]:.3f} "
                                     f"Γ={G_mp[i_best]:.3f}", fontsize=8)
                axes[1, j].set_aspect("equal")

            fig.suptitle(f"Block MP (SVD) vs Time filter — m={mass_val}",
                         fontsize=12)
            plt.tight_layout()
            out = os.path.join(OUT_DIR, f"block_{tag}_modes_svd.png")
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  SVD modes → {out}")

        del snapshots, result, psi0_up, psi0_dn

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*70}")
    print(f"  TOTAL: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
