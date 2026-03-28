#!/usr/bin/env python3
"""
Two tasks in one script:

Task 1 — Fix the normalization bug in FDM overlap metric and re-evaluate
          Part 1 (m=1.0) spatial overlap.

Task 2 — Extract spatial modes for narrow resonances (Γ < 0.25) using
          time filtering from matrix pencil frequencies.  These modes
          have Γ/ΔE < 1 and should be spatially reliable.

Both tasks run a single propagation each (~2 min CPU per propagation).
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

# ── reuse FDM core from fdm_debug ──
from fdm_debug import (
    build_probes_fixed, build_BA, solve_fdm, reconstruct_modes,
)

OUT_DIR = os.path.join(_root, "figures", "fdm_debug")
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed parameters (same as fdm_debug.py)
n = 256; L = 10.0; vF = 1.0
dt = 0.01; n_steps = 15000; save_every = 10
POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5; cap_strength = 10.0; cap_power = 2
mask_safety_factor = 0.95; mask_margin = 1.30
sigma0 = 0.6; k_mag = 3.0
Gamma_cut = 8.0; pencil_fraction = 4; L_pencil_max = 400


def fft_flip(arr, axis):
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)


def measure_c2v(rho):
    rx = fft_flip(rho, axis=1)
    ry = fft_flip(rho, axis=0)
    s = np.sum(rho) + 1e-30
    return np.sum(np.abs(rx - rho)) / s, np.sum(np.abs(ry - rho)) / s


def spinor_overlap(a, b, chi, dx2):
    """Properly normalized overlap |<a|chi|b>|^2 / (<a|chi|a> <b|chi|b>)."""
    ab = (np.vdot(a[0] * chi, b[0]) + np.vdot(a[1] * chi, b[1])) * dx2
    aa = (np.vdot(a[0] * chi, a[0]) + np.vdot(a[1] * chi, a[1])) * dx2
    bb = (np.vdot(b[0] * chi, b[0]) + np.vdot(b[1] * chi, b[1])) * dx2
    denom = np.real(aa) * np.real(bb)
    if denom < 1e-30:
        return 0.0
    return float(np.abs(ab)**2 / denom)


def run_propagation(xp, mass, kx0, ky0, X, Y, KX, KY, K, dx,
                    X_np, Y_np, chi_cpu):
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
    return dict(snapshots=result["snapshots"], snap_times=result["snap_times"],
                C_R=result["C_R"], corr_times=result["corr_times"],
                psi0_cpu=psi0_cpu)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Task 1: Fix overlap normalization + re-evaluate m=1             ║
# ╚═══════════════════════════════════════════════════════════════════╝

def task1_fix_overlap(xp, X, Y, KX, KY, K, dx, X_np, Y_np, chi_cpu, chi_flat, dx2):
    print("=" * 70)
    print("  TASK 1: Fix overlap normalization (m=1.0)")
    print("=" * 70)

    dt_snap = dt * save_every

    print("Propagating m=1.0 ...")
    data = run_propagation(xp, 1.0, k_mag, 0.0, X, Y, KX, KY, K, dx,
                           X_np, Y_np, chi_cpu)

    print("Matrix pencil ...")
    spec = extract_resonances(data["C_R"], data["corr_times"], dt_snap,
                              Gamma_cut=Gamma_cut, pencil_fraction=pencil_fraction,
                              L_pencil_max=L_pencil_max)
    E_mp, G_mp, omega_mp = spec["E"], spec["Gamma"], spec["omega"]
    n_mp = min(5, len(E_mp))

    # Filter to conduction band (E > 0) and reasonable Γ
    good = (E_mp > 0) & (G_mp < 1.0)
    E_mp, G_mp, omega_mp = E_mp[good], G_mp[good], omega_mp[good]
    n_mp = min(5, len(E_mp))

    print(f"  MP: {n_mp} good poles:")
    for j in range(n_mp):
        print(f"    E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

    # Time-filter modes
    print("Time filtering ...")
    tf_modes = [time_filter_mode(data["snapshots"], data["snap_times"],
                                  omega_mp[j], chi_cpu)
                for j in range(n_mp)]

    # FDM with MP-guided probes
    print("FDM (MP-guided, eps=1e-2) ...")
    E_probes = E_mp[:n_mp].copy()
    if len(E_probes) < 5:
        E_probes = np.sort(np.concatenate([
            E_probes, np.linspace(E_mp[0]-0.5, E_mp[-1]+0.5, 5-len(E_probes))
        ]))

    probes, shifted = build_probes_fixed(data["snapshots"], data["snap_times"],
                                          E_probes, chi_flat)
    B, A = build_BA(probes, shifted, dx2)
    fdm = solve_fdm(B, A, dt_snap, eps_svd=1e-2, Gamma_max=Gamma_cut)

    n_fdm = min(10, fdm["vecs"].shape[1])
    if n_fdm > 0:
        order = np.argsort(fdm["Gamma"])[:n_fdm]
        fdm["E"] = fdm["E"][order]
        fdm["Gamma"] = fdm["Gamma"][order]
        fdm["omega"] = fdm["omega"][order]
        fdm["vecs"] = fdm["vecs"][:, order]
    fdm_modes = reconstruct_modes(fdm["vecs"], probes, n, chi_cpu, dx2)

    # Match and compute FIXED overlap
    print("\nMatching with FIXED overlap normalization:")
    print(f"  {'MP#':>4} {'E_MP':>8} {'Γ_MP':>8} {'E_FDM':>8} {'Γ_FDM':>8} "
          f"{'ΔΓ/Γ':>8} {'overlap_old':>12} {'overlap_fix':>12}")

    matches = []
    for i in range(n_mp):
        if len(fdm["E"]) == 0:
            break
        j = np.argmin(np.abs(fdm["E"] - E_mp[i]))
        relG = abs(fdm["Gamma"][j] - G_mp[i]) / G_mp[i]

        # OLD overlap (buggy — inconsistent normalization)
        if j < len(fdm_modes) and i < len(tf_modes):
            ovlp_old = (np.vdot(fdm_modes[j][0]*chi_cpu, tf_modes[i][0]) +
                        np.vdot(fdm_modes[j][1]*chi_cpu, tf_modes[i][1])) * dx2
            overlap_old = float(np.abs(ovlp_old)**2)
        else:
            overlap_old = 0.0

        # FIXED overlap (properly normalized)
        if j < len(fdm_modes) and i < len(tf_modes):
            overlap_fix = spinor_overlap(fdm_modes[j], tf_modes[i], chi_cpu, dx2)
        else:
            overlap_fix = 0.0

        print(f"  {i+1:>4} {E_mp[i]:8.4f} {G_mp[i]:8.4f} "
              f"{fdm['E'][j]:8.4f} {fdm['Gamma'][j]:8.4f} "
              f"{relG:8.3f} {overlap_old:12.6f} {overlap_fix:12.6f}")
        matches.append(dict(E_mp=E_mp[i], G_mp=G_mp[i],
                            E_fdm=fdm["E"][j], G_fdm=fdm["Gamma"][j],
                            relG=relG, overlap_old=overlap_old,
                            overlap_fix=overlap_fix))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(E_mp[:n_mp], G_mp[:n_mp], s=80, marker='o', facecolors='none',
               edgecolors='C0', lw=2, label='MP', zorder=3)
    if len(fdm["E"]) > 0:
        ax.scatter(fdm["E"], fdm["Gamma"], s=40, marker='x', color='C1',
                   lw=2, label='FDM', zorder=3)
    ax.set_xlabel("E"); ax.set_ylabel(r"$\Gamma$")
    ax.set_title("Pole comparison (m=1)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    x = range(len(matches))
    ax.bar(x, [m["overlap_old"] for m in matches], width=0.35, label="old (buggy)",
           color='C3', alpha=0.6)
    ax.bar([xi+0.35 for xi in x], [m["overlap_fix"] for m in matches],
           width=0.35, label="fixed", color='C0')
    ax.set_xlabel("mode"); ax.set_ylabel(r"$|\langle\phi_{FDM}|\phi_{TF}\rangle|^2$")
    ax.set_title("Overlap: old vs fixed"); ax.legend()
    ax.set_ylim(0, 1.1)

    ax = axes[2]
    ax.bar(x, [m["relG"] for m in matches], color='C2')
    ax.axhline(0.15, ls='--', color='r', label='15%')
    ax.set_xlabel("mode"); ax.set_ylabel(r"$|\Delta\Gamma/\Gamma|$")
    ax.set_title(r"Relative $\Gamma$ error"); ax.legend()

    fig.suptitle("Task 1: FDM validation with FIXED overlap (m=1.0)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "task1_fixed_overlap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Figure -> {out}")

    return matches


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Task 2: Narrow-Γ modes from time filtering (m=0)               ║
# ╚═══════════════════════════════════════════════════════════════════╝

def task2_narrow_modes(xp, X, Y, KX, KY, K, dx, X_np, Y_np, chi_cpu, dx2):
    print("\n" + "=" * 70)
    print("  TASK 2: Extract narrow-Γ modes (Γ < 0.25) via time filtering (m=0)")
    print("=" * 70)

    dt_snap = dt * save_every
    Gamma_threshold = 0.25

    # Multiple propagation directions for better coverage
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    all_E, all_G, all_omega, all_a = [], [], [], []
    all_snapshots = []
    all_snap_times = []

    for ia, angle in enumerate(angles):
        kx0 = k_mag * np.cos(angle)
        ky0 = k_mag * np.sin(angle)
        print(f"\nPropagation {ia+1}/4: k=({kx0:+.2f},{ky0:+.2f})")
        data = run_propagation(xp, 0.0, kx0, ky0, X, Y, KX, KY, K, dx,
                               X_np, Y_np, chi_cpu)

        spec = extract_resonances(data["C_R"], data["corr_times"], dt_snap,
                                  Gamma_cut=Gamma_cut,
                                  pencil_fraction=pencil_fraction,
                                  L_pencil_max=L_pencil_max)
        all_E.append(spec["E"])
        all_G.append(spec["Gamma"])
        all_omega.append(spec["omega"])
        all_a.append(spec["a"])
        all_snapshots.append(data["snapshots"])
        all_snap_times.append(data["snap_times"])

        n_narrow = np.sum(spec["Gamma"] < Gamma_threshold)
        print(f"  MP: {len(spec['E'])} poles, {n_narrow} with Γ<{Gamma_threshold}")

    # Collect all narrow modes across propagations, deduplicate by energy
    print(f"\nCollecting narrow modes (Γ < {Gamma_threshold}) ...")
    narrow_modes = []  # list of (E, Gamma, omega, prop_idx, mode_idx)
    for ia in range(len(angles)):
        for j in range(len(all_E[ia])):
            if all_E[ia][j] > 0 and all_G[ia][j] < Gamma_threshold:
                narrow_modes.append((all_E[ia][j], all_G[ia][j],
                                     all_omega[ia][j], ia, j))

    # Deduplicate: merge modes within ΔE < 0.1
    narrow_modes.sort(key=lambda x: x[0])
    deduped = []
    used = set()
    for i, (E, G, omega, ia, j) in enumerate(narrow_modes):
        if i in used:
            continue
        # find cluster
        cluster = [(E, G, omega, ia, j)]
        used.add(i)
        for k in range(i+1, len(narrow_modes)):
            if k in used:
                continue
            if abs(narrow_modes[k][0] - E) < 0.1:
                cluster.append(narrow_modes[k])
                used.add(k)
            else:
                break
        # pick the one with smallest Γ
        best = min(cluster, key=lambda x: x[1])
        deduped.append(best)

    print(f"  {len(narrow_modes)} raw narrow poles -> {len(deduped)} unique modes:")
    for E, G, omega, ia, j in deduped:
        print(f"    E={E:.4f}  Γ={G:.4f}  (from propagation {ia+1})")

    if len(deduped) == 0:
        print("  No narrow modes found!")
        return

    # Time-filter each narrow mode from its best propagation
    print("\nTime filtering narrow modes ...")
    filtered_modes = []
    mode_info = []
    for E, G, omega, ia, j in deduped:
        phi = time_filter_mode(all_snapshots[ia], all_snap_times[ia],
                                omega, chi_cpu)
        filtered_modes.append(phi)
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        x_err, y_err = measure_c2v(rho)
        mode_info.append(dict(E=E, G=G, x_err=x_err, y_err=y_err, prop=ia))

    # Report
    print(f"\n  {'E':>8} {'Γ':>8} {'Γ/ΔE':>8} {'x_err':>8} {'y_err':>8} {'prop':>5}")
    for k, m in enumerate(mode_info):
        # compute Γ/ΔE to nearest neighbor
        if len(mode_info) > 1:
            dEs = [abs(m["E"] - mode_info[j]["E"]) for j in range(len(mode_info)) if j != k]
            dE_min = min(dEs) if dEs else 1.0
        else:
            dE_min = 1.0
        ratio = m["G"] / dE_min if dE_min > 0 else float('inf')
        print(f"  {m['E']:8.4f} {m['G']:8.4f} {ratio:8.3f} "
              f"{m['x_err']:8.4f} {m['y_err']:8.4f} {m['prop']+1:5d}")

    # Check pairwise overlap between modes
    print("\nPairwise overlap matrix (should be ~0 for orthogonal modes):")
    nm = len(filtered_modes)
    O = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            O[i, j] = spinor_overlap(filtered_modes[i], filtered_modes[j],
                                      chi_cpu, dx2)
    print("  " + "  ".join([f"m{k+1:d}" for k in range(nm)]))
    for i in range(nm):
        row = "  ".join([f"{O[i,j]:.3f}" for j in range(nm)])
        print(f"  m{i+1}: {row}")

    # ── Figures ──
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]

    # Figure 1: mode densities
    ncols = min(len(filtered_modes), 4)
    nrows = (len(filtered_modes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for k, (phi, mi) in enumerate(zip(filtered_modes, mode_info)):
        ax = axes[k // ncols, k % ncols]
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                       vmin=0, vmax=rho.max())
        ax.set_title(f"E={mi['E']:.3f}  $\\Gamma$={mi['G']:.3f}\n"
                     f"x_err={mi['x_err']:.4f}  y_err={mi['y_err']:.4f}",
                     fontsize=9)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
    for k in range(len(filtered_modes), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle(f"Narrow modes ($\\Gamma < {Gamma_threshold}$) from time filtering, "
                 f"massless harmonic ellipse", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "task2_narrow_modes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Modes figure -> {out}")

    # Figure 2: spectrum with narrow modes highlighted
    fig, ax = plt.subplots(figsize=(10, 4))
    for ia in range(len(angles)):
        mask_narrow = all_G[ia] < Gamma_threshold
        mask_broad  = ~mask_narrow & (all_E[ia] > 0)
        ax.scatter(all_E[ia][mask_broad], all_G[ia][mask_broad],
                   s=15, alpha=0.3, color='gray', zorder=1)
        ax.scatter(all_E[ia][mask_narrow], all_G[ia][mask_narrow],
                   s=50, marker='*', color=f'C{ia}', zorder=3,
                   label=f'prop {ia+1} narrow')
    ax.axhline(Gamma_threshold, ls='--', color='red', lw=1.5,
               label=f'$\\Gamma = {Gamma_threshold}$')
    ax.set_xlabel("E"); ax.set_ylabel(r"$\Gamma$")
    ax.set_title("Matrix pencil poles: narrow modes highlighted")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 6); ax.set_ylim(0, 2)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "task2_narrow_spectrum.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Spectrum figure -> {out}")

    return dict(modes=filtered_modes, info=mode_info, overlap=O)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Main                                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝

def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx_np = L / n; dx2 = dx_np**2

    chi_R, _ = build_analysis_mask(
        POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)
    chi_flat = chi_cpu.ravel()

    t1 = task1_fix_overlap(xp, X, Y, KX, KY, K, dx, X_np, Y_np,
                           chi_cpu, chi_flat, dx2)
    t2 = task2_narrow_modes(xp, X, Y, KX, KY, K, dx, X_np, Y_np,
                            chi_cpu, dx2)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"  TOTAL: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
