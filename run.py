#!/usr/bin/env python3
"""
Complete pipeline for 2D Dirac billiard simulations.

Propagation → LDOS (spectral density map)
            → Pole extraction (complex resonance frequencies)
            → Mode reconstruction (spatial quasi-eigenstates)

Usage:
    python run.py
    python run.py --potential soft_stadium --a_half 1.0 --b 1.0
    python run.py --mass 1.0 --n_prop 48 --backend torch_cuda
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances, time_filter_mode, compute_ldos


def fixed_pole_svd_modes(snapshots, snap_times, omega_arr, chi_R,
                         n_modes=5, rank=40, L=10.0):
    """
    Fixed-pole mode extraction via SVD subspace projection.

    Combines matrix-pencil poles (super-resolution) with SVD spatial
    noise suppression.  For each known pole omega_n:
      1. Build snapshot matrix, compute shift matrices X0, X1.
      2. Truncated SVD of X0 → reduced dynamics operator A_tilde (r x r).
      3. Find approximate null vector of (A_tilde - lambda_n I).
      4. Project back to full spatial grid.
    """
    N = min(n_modes, len(omega_arr))
    ny, nx = snapshots.shape[2], snapshots.shape[3]
    Ns = len(snap_times)
    dt_snap = snap_times[1] - snap_times[0]
    D = 2 * ny * nx

    X = np.zeros((D, Ns), dtype=complex)
    for k in range(Ns):
        X[:ny*nx, k] = (snapshots[k, 0] * chi_R).ravel()
        X[ny*nx:, k] = (snapshots[k, 1] * chi_R).ravel()

    X0 = X[:, :-1]
    X1 = X[:, 1:]

    U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    r = min(rank, len(S))
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]

    A_tilde = (U_r.conj().T @ X1) @ (Vh_r.conj().T * (1.0 / S_r)[None, :])

    lam_forced = np.exp(-1j * omega_arr[:N] * dt_snap)
    W_forced = np.zeros((r, N), dtype=complex)
    for j in range(N):
        M = A_tilde - lam_forced[j] * np.eye(r)
        _, _, vh = np.linalg.svd(M)
        W_forced[:, j] = vh[-1, :].conj()

    Phi = X1 @ (Vh_r.conj().T * (1.0 / S_r)[None, :]) @ W_forced

    dx_val = L / nx
    dx2 = dx_val ** 2
    modes = []
    for j in range(N):
        phi = np.zeros((2, ny, nx), dtype=complex)
        phi[0] = Phi[:ny*nx, j].reshape(ny, nx)
        phi[1] = Phi[ny*nx:, j].reshape(ny, nx)
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        norm = np.sqrt(np.sum(rho * chi_R) * dx2)
        if norm > 1e-30:
            phi = phi / norm
        modes.append(phi)

    return modes


def run_pipeline(args):
    t_start = time.perf_counter()
    os.makedirs(args.outdir, exist_ok=True)

    xp = get_backend(args.backend)

    # ── Grid ──
    x_np = np.linspace(-args.L/2, args.L/2, args.n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(args.n, args.L, xp)
    dx2 = (args.L / args.n)**2

    # ── Potential, CAP, mask ──
    pot_params = {}
    if "ellipse" in args.potential:
        pot_params = dict(k_harm=args.k_harm, a=args.a, b=args.b)
    elif "circle" in args.potential:
        pot_params = dict(k_harm=args.k_harm, r0=args.a)
    elif "stadium" in args.potential:
        pot_params = dict(a_half=args.a_half, b=args.b,
                          V0=args.V0, lam=args.lam, mu=args.mu)

    V = build_potential(args.potential, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, args.cap_radius, args.cap_strength,
                   args.cap_power, xp=xp)
    chi_R, mask_info = build_analysis_mask(
        args.potential, X_np, Y_np, pot_params,
        cap_radius=args.cap_radius, safety_factor=0.95, mask_margin=1.30,
        xp=xp)
    sqrt_chi = xp.sqrt(chi_R)
    chi_cpu = xp.to_numpy(chi_R)

    halfV, kin = build_split_operator(
        args.vF, args.dt, args.mass, V, W, KX, KY, K, xp)

    # ── Multi-angle propagation + LDOS ──
    angles = np.linspace(0, 2 * np.pi, args.n_prop, endpoint=False)
    E_values = np.linspace(args.E_lo, args.E_hi, args.n_E)
    dt_snap = args.dt * args.save_every
    ldos_sum = None
    first_result = None

    for i, angle in enumerate(angles):
        kx0 = args.k_mag * np.cos(angle)
        ky0 = args.k_mag * np.sin(angle)

        psi0_up, psi0_dn = gaussian_wavepacket(
            X, Y, x0=0.0, y0=0.0, kx0=kx0, ky0=ky0,
            sigma=args.sigma, band="conduction", dx=dx, xp=xp)

        t0 = time.perf_counter()
        result = propagate(
            psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
            n_steps=args.n_steps, dt=args.dt,
            save_every=args.save_every, corr_every=args.save_every,
            max_record=args.n_steps // args.save_every + 2,
            measure_transmission=False)

        snapshots = result["snapshots"]
        snap_times = result["snap_times"]
        print(f"  [{i+1}/{args.n_prop}] "
              f"angle={np.degrees(angle):5.1f} deg  "
              f"{time.perf_counter()-t0:.1f}s  "
              f"{len(snap_times)} snapshots")

        ldos_one = compute_ldos(snapshots, snap_times, E_values, chi_cpu)
        ldos_sum = ldos_one if ldos_sum is None else ldos_sum + ldos_one

        if i == 0:
            first_result = result
        del snapshots

    ldos_avg = ldos_sum / args.n_prop

    # ── Pole extraction ──
    spec = extract_resonances(
        first_result["C_R"], first_result["corr_times"], dt_snap,
        Gamma_cut=args.Gamma_cut, pencil_fraction=4, L_pencil_max=400)

    good = ((spec["E"] > args.E_lo) & (spec["E"] < args.E_hi)
            & (spec["Gamma"] < args.Gamma_cut))
    E_mp = spec["E"][good][:args.n_modes]
    G_mp = spec["Gamma"][good][:args.n_modes]
    omega_mp = spec["omega"][good][:args.n_modes]
    n_poles = len(E_mp)

    print(f"\n  Poles: {n_poles}")
    for j in range(n_poles):
        print(f"    E={E_mp[j]:.4f}  Gamma={G_mp[j]:.4f}")

    # ── Mode extraction ──
    snapshots = first_result["snapshots"]
    snap_times = first_result["snap_times"]

    # Method A: Time filtering (per-pixel Fourier filter)
    modes_tf = [time_filter_mode(snapshots, snap_times, omega_mp[j], chi_cpu)
                for j in range(n_poles)]

    # Method B: Fixed-pole SVD extraction (spatial noise suppression)
    print("\n  SVD mode extraction (rank=40)...")
    t0 = time.perf_counter()
    modes_svd = fixed_pole_svd_modes(snapshots, snap_times, omega_mp,
                                      chi_cpu, n_modes=n_poles, rank=40,
                                      L=args.L)
    print(f"    {time.perf_counter()-t0:.1f}s")

    # ── Save ──
    C_R = first_result["C_R"]
    corr_times = first_result["corr_times"]

    np.save(os.path.join(args.outdir, "ldos.npy"), ldos_avg)
    np.save(os.path.join(args.outdir, "E_values.npy"), E_values)
    np.save(os.path.join(args.outdir, "chi_R.npy"), chi_cpu)
    np.save(os.path.join(args.outdir, "C_R.npy"), C_R)
    np.save(os.path.join(args.outdir, "corr_times.npy"), corr_times)
    np.save(os.path.join(args.outdir, "poles.npy"),
            np.column_stack([E_mp, G_mp]) if n_poles > 0 else np.array([]))
    for j in range(n_poles):
        np.save(os.path.join(args.outdir, f"mode_tf_{j}.npy"), modes_tf[j])
        np.save(os.path.join(args.outdir, f"mode_svd_{j}.npy"), modes_svd[j])

    # ── Figures ──
    extent = [-args.L/2, args.L/2, -args.L/2, args.L/2]

    # Autocorrelation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(corr_times, np.abs(C_R), 'k-', lw=0.6)
    ax1.set_xlabel("t"); ax1.set_ylabel("|C_R(t)|")
    ax1.set_title("Autocorrelation magnitude")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3)
    ax2.plot(corr_times, np.real(C_R), 'C0-', lw=0.4, label="Re")
    ax2.plot(corr_times, np.imag(C_R), 'C1-', lw=0.4, label="Im")
    ax2.set_xlabel("t"); ax2.set_ylabel("C_R(t)")
    ax2.set_title("Autocorrelation (Re/Im)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "autocorrelation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # LDOS spectrum
    ldos_spec = np.array([np.sum(ldos_avg[i] * chi_cpu) * dx2
                          for i in range(len(E_values))])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(E_values, ldos_spec, 'k-', lw=0.8)
    for j in range(n_poles):
        ax.axvline(E_mp[j], color='C0', ls='--', alpha=0.5, lw=0.7)
    ax.set_xlabel("E"); ax.set_ylabel("Integrated LDOS")
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "ldos_spectrum.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # LDOS + mode maps (3 rows: LDOS, TF mode, SVD mode)
    n_show = min(n_poles, 6)
    if n_show > 0:
        fig, axes = plt.subplots(3, n_show, figsize=(3.5*n_show, 10))
        if n_show == 1:
            axes = axes[:, np.newaxis]
        for j in range(n_show):
            idx = np.argmin(np.abs(E_values - E_mp[j]))
            # Row 0: LDOS
            im = ldos_avg[idx]
            axes[0, j].imshow(im, extent=extent, origin="lower",
                              cmap="inferno", vmin=0,
                              vmax=np.percentile(im, 99.5))
            axes[0, j].set_title(f"E={E_mp[j]:.3f}", fontsize=9)
            axes[0, j].set_aspect("equal")
            axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
            # Row 1: Time-filter mode
            rho_tf = np.abs(modes_tf[j][0])**2 + np.abs(modes_tf[j][1])**2
            axes[1, j].imshow(rho_tf, extent=extent, origin="lower",
                              cmap="inferno", vmin=0,
                              vmax=np.percentile(rho_tf, 99.5))
            axes[1, j].set_aspect("equal")
            axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
            # Row 2: SVD mode
            rho_svd = np.abs(modes_svd[j][0])**2 + np.abs(modes_svd[j][1])**2
            axes[2, j].imshow(rho_svd, extent=extent, origin="lower",
                              cmap="inferno", vmin=0,
                              vmax=np.percentile(rho_svd, 99.5))
            axes[2, j].set_aspect("equal")
            axes[2, j].set_xticks([]); axes[2, j].set_yticks([])
        axes[0, 0].set_ylabel("LDOS")
        axes[1, 0].set_ylabel("Mode (TF)")
        axes[2, 0].set_ylabel("Mode (SVD)")
        plt.tight_layout()
        fig.savefig(os.path.join(args.outdir, "ldos_and_modes.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  Total: {time.perf_counter()-t_start:.0f}s")
    print(f"  Output: {args.outdir}/")


def main():
    p = argparse.ArgumentParser(description="2D Dirac billiard pipeline")
    p.add_argument("--backend", default="auto")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--L", type=float, default=10.0)
    p.add_argument("--vF", type=float, default=1.0)
    p.add_argument("--mass", type=float, default=0.0)
    p.add_argument("--potential", default="harmonic_ellipse")
    p.add_argument("--k_harm", type=float, default=3.0)
    p.add_argument("--a", type=float, default=2.0)
    p.add_argument("--b", type=float, default=1.2)
    p.add_argument("--a_half", type=float, default=2.0)
    p.add_argument("--V0", type=float, default=70.0)
    p.add_argument("--lam", type=float, default=25.0)
    p.add_argument("--mu", type=float, default=5.0)
    p.add_argument("--cap_radius", type=float, default=3.5)
    p.add_argument("--cap_strength", type=float, default=10.0)
    p.add_argument("--cap_power", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--k_mag", type=float, default=3.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--n_steps", type=int, default=15000)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--n_prop", type=int, default=8)
    p.add_argument("--E_lo", type=float, default=1.0)
    p.add_argument("--E_hi", type=float, default=6.0)
    p.add_argument("--n_E", type=int, default=300)
    p.add_argument("--Gamma_cut", type=float, default=8.0)
    p.add_argument("--n_modes", type=int, default=5)
    p.add_argument("--outdir", default="output")
    run_pipeline(p.parse_args())


if __name__ == "__main__":
    main()
