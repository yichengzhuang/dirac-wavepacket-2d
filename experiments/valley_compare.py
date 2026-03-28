#!/usr/bin/env python3
"""
Verify the valley (K vs K') x-flip relation used for STM density.

Physics:
  H_K  = v_F ( σ_x k_x + σ_y k_y) + V(x,y)·I    (valley = +1)
  H_K' = v_F (-σ_x k_x + σ_y k_y) + V(x,y)·I    (valley = -1)

P_x intertwiner:  P_x H_K P_x⁻¹ = H_K'  (pure spatial x-reflection)
  → If ψ_K(t) solves H_K, then P_x ψ_K(t) solves H_K' with initial
    condition P_x ψ_K(0).
  → ρ_K'(x,y,t) = |P_x ψ_K(x,y,t)|² = |ψ_K(-x,y,t)|² = ρ_K(-x,y,t)

This script verifies:
  1. Run K valley → get ψ_K(t)
  2. Run K' valley with initial condition P_x ψ_K₀ = ψ_K₀(-x,y)
  3. Check ρ_K'(x,y,t) = ρ_K(-x,y,t) at every snapshot
  4. Compare K' resonance modes with x-flipped K modes
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances, time_filter_mode

# ── parameters (same as kramers_test.py) ──
n  = 512
L  = 10.0
vF = 1.0
m  = 0.0
dt = 0.01
n_steps = 10000
decim = 10
save_every = 10
max_record = 2500

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

mask_safety_factor = 0.95
mask_margin        = 1.30

x0, y0   = 0.0, 0.0
kx0, ky0 = -1.0, 0.0
sigma0   = 0.6
theta0   = None
band0    = "conduction"

Gamma_cut       = 8.0
pencil_fraction = 4
L_pencil_max    = 400
n_modes         = 3


def density(phi):
    return np.abs(phi[0])**2 + np.abs(phi[1])**2

def normalize(phi, dx):
    norm2 = np.sum(density(phi)) * dx**2
    return phi / np.sqrt(norm2) if norm2 > 0 else phi

def overlap(a, b, dx):
    return (np.vdot(a[0].ravel(), b[0].ravel()) +
            np.vdot(a[1].ravel(), b[1].ravel())) * dx**2

def fft_xflip(arr):
    """FFT-consistent x → -x: index j → (n-j) % n along axis=1 (x-axis)."""
    return np.roll(np.flip(arr, axis=1), 1, axis=1)

def apply_Px(phi):
    """P_x: pure spatial x-reflection (no spinor operation).
    (φ_up(x,y), φ_dn(x,y)) → (φ_up(-x,y), φ_dn(-x,y))
    Uses FFT-consistent reflection: j → (n-j) % n.
    """
    return np.stack([fft_xflip(phi[0]), fft_xflip(phi[1])], axis=0)


def main():
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    os.makedirs(out_dir, exist_ok=True)

    # ── shared setup ──
    x_np = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx_val = L / n

    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(
        POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp,
    )
    sqrt_chi = xp.sqrt(chi_R)
    chi_cpu = xp.to_numpy(chi_R)

    # ══════════════════════════════════════════════
    # K valley propagation
    # ══════════════════════════════════════════════
    print(f"\n{'='*50}")
    print(f"  Valley K (valley=+1)")
    print(f"{'='*50}")

    halfV_K, kin_K = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp, valley=+1)
    psi0_K_up, psi0_K_dn = gaussian_wavepacket(
        X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=theta0, band=band0, dx=dx, xp=xp, valley=+1,
    )

    res_K = propagate(
        psi0_K_up, psi0_K_dn, halfV_K, kin_K, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt,
        save_every=save_every, corr_every=decim, max_record=max_record,
        measure_transmission=True,
        transmission_every=max(n_steps // 10, 1),
    )
    print(f"  Recorded {len(res_K['C_R'])} corr samples, {len(res_K['snapshots'])} snapshots")

    print(f"  Harmonic inversion ...")
    spec_K = extract_resonances(res_K["C_R"], res_K["corr_times"], dt * decim,
                                Gamma_cut=Gamma_cut,
                                pencil_fraction=pencil_fraction,
                                L_pencil_max=L_pencil_max)
    nm = min(n_modes, len(spec_K["omega"]))
    for j in range(nm):
        print(f"    mode {j+1}:  E={spec_K['E'][j]:.6f}  G={spec_K['Gamma'][j]:.6f}")

    modes_K = [time_filter_mode(res_K["snapshots"], res_K["snap_times"],
                                spec_K["omega"][j], chi_cpu)
               for j in range(nm)]

    # ══════════════════════════════════════════════
    # K' valley propagation with P_x initial condition
    # ══════════════════════════════════════════════
    print(f"\n{'='*50}")
    print(f"  Valley K' (valley=-1), init = P_x psi_K0")
    print(f"{'='*50}")

    halfV_Kp, kin_Kp = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp, valley=-1)

    # P_x ψ_K₀: FFT-consistent x-reflection of K initial wavepacket
    psi0_Kp_up = xp.array(fft_xflip(xp.to_numpy(psi0_K_up)))
    psi0_Kp_dn = xp.array(fft_xflip(xp.to_numpy(psi0_K_dn)))

    res_Kp = propagate(
        psi0_Kp_up, psi0_Kp_dn, halfV_Kp, kin_Kp, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt,
        save_every=save_every, corr_every=decim, max_record=max_record,
        measure_transmission=True,
        transmission_every=max(n_steps // 10, 1),
    )
    print(f"  Recorded {len(res_Kp['C_R'])} corr samples, {len(res_Kp['snapshots'])} snapshots")

    print(f"  Harmonic inversion ...")
    spec_Kp = extract_resonances(res_Kp["C_R"], res_Kp["corr_times"], dt * decim,
                                 Gamma_cut=Gamma_cut,
                                 pencil_fraction=pencil_fraction,
                                 L_pencil_max=L_pencil_max)
    nm = min(nm, len(spec_Kp["omega"]))
    for j in range(nm):
        print(f"    mode {j+1}:  E={spec_Kp['E'][j]:.6f}  G={spec_Kp['Gamma'][j]:.6f}")

    modes_Kp = [time_filter_mode(res_Kp["snapshots"], res_Kp["snap_times"],
                                 spec_Kp["omega"][j], chi_cpu)
                for j in range(nm)]

    # ══════════════════════════════════════════════
    # Verification
    # ══════════════════════════════════════════════
    print("\n" + "="*70)
    print("  Verification: rho_K'(x,y) = rho_K(-x,y)")
    print("="*70)

    # 1. Energy comparison
    print("\n--- Energy comparison ---")
    print("  mode       E_K      E_K'      |dE|")
    for j in range(nm):
        dE = abs(spec_K["E"][j] - spec_Kp["E"][j])
        print(f"  {j+1:4d}  {spec_K['E'][j]:10.6f}  {spec_Kp['E'][j]:10.6f}  {dE:10.2e}")

    # 2. Snapshot-level density: rho_K'(x,y,t) vs rho_K(-x,y,t)
    print("\n--- Snapshot density: rho_K'(x,y,t) vs rho_K(-x,y,t) ---")
    snaps_K  = res_K["snapshots"]
    snaps_Kp = res_Kp["snapshots"]
    n_check = min(5, len(snaps_K), len(snaps_Kp))
    indices = np.linspace(0, min(len(snaps_K), len(snaps_Kp))-1, n_check, dtype=int)
    for idx in indices:
        rho_K  = np.abs(snaps_K[idx][0])**2 + np.abs(snaps_K[idx][1])**2
        rho_Kp = np.abs(snaps_Kp[idx][0])**2 + np.abs(snaps_Kp[idx][1])**2
        rho_K_flip = fft_xflip(rho_K)
        diff = rho_Kp - rho_K_flip
        rms = np.sqrt(np.mean(diff**2))
        rel = rms / (np.sqrt(np.mean(rho_K**2)) + 1e-30)
        t_snap = res_K["snap_times"][idx]
        print(f"  t={t_snap:6.1f}:  |rho_K' - rho_K(-x,y)| rel = {rel:.2e}  (should be ~0)")

    # 3. Mode density comparison
    print("\n--- Mode density: rho_K'(x,y) vs rho_K(-x,y) ---")
    for j in range(nm):
        phi_K  = normalize(modes_K[j], dx_val)
        phi_Kp = normalize(modes_Kp[j], dx_val)
        rho_K  = density(phi_K)
        rho_Kp = density(phi_Kp)
        rho_K_flip = fft_xflip(rho_K)
        diff = rho_Kp - rho_K_flip
        rel = np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(rho_K**2)) + 1e-30)
        print(f"  mode {j+1}:  rel RMS = {rel:.2e}  (should be ~0)")

    # ══════════════════════════════════════════════
    # Plot: rho_K, rho_K(-x,y), rho_K'(sim), difference
    # ══════════════════════════════════════════════
    extent = [-L/2, L/2, -L/2, L/2]
    fig, axes = plt.subplots(nm, 4, figsize=(16, 4 * nm))
    if nm == 1:
        axes = axes[np.newaxis, :]

    for j in range(nm):
        phi_K  = normalize(modes_K[j], dx_val)
        phi_Kp = normalize(modes_Kp[j], dx_val)

        rho_K      = density(phi_K)
        rho_K_flip = fft_xflip(rho_K)
        rho_Kp_sim = density(phi_Kp)

        diff = rho_Kp_sim - rho_K_flip
        rel = np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(rho_K**2)) + 1e-30)

        vmax = max(rho_K.max(), 1e-10)

        ax = axes[j, 0]
        ax.imshow(rho_K, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"Mode {j+1}: $\\rho_K(x,y)$\nE={spec_K['E'][j]:.4f}", fontsize=10)
        ax.set_ylabel(f"mode {j+1}", fontsize=12)

        ax = axes[j, 1]
        ax.imshow(rho_K_flip, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"$\\rho_K(-x,y)$  (x-flip shortcut)", fontsize=10)

        ax = axes[j, 2]
        ax.imshow(rho_Kp_sim, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"$\\rho_{{K'}}(x,y)$  (explicit simulation)", fontsize=10)

        ax = axes[j, 3]
        dmax = max(np.abs(diff).max(), 1e-10)
        ax.imshow(diff, extent=extent, origin="lower", cmap="RdBu_r",
                  vmin=-dmax, vmax=dmax)
        ax.set_title(f"$\\rho_{{K'}} - \\rho_K(-x,y)$\nrel={rel:.1e}", fontsize=10)

    for ax in axes.flat:
        ax.set_aspect("equal")

    fig.suptitle("Valley verification:  $P_x H_K P_x^{-1} = H_{K'}$  "
                 "$\\Rightarrow$  $\\rho_{K'}(x,y) = \\rho_K(-x,y)$",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "valley_compare.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # ── STM density ──
    print("\n" + "="*70)
    print("  STM density: rho_STM = rho_K + rho_K' = rho_K(x,y) + rho_K(-x,y)")
    print("="*70)

    fig2, axes2 = plt.subplots(nm, 3, figsize=(14, 4 * nm))
    if nm == 1:
        axes2 = axes2[np.newaxis, :]

    for j in range(nm):
        phi_K  = normalize(modes_K[j], dx_val)
        phi_Kp = normalize(modes_Kp[j], dx_val)
        rho_K  = density(phi_K)
        rho_Kp = density(phi_Kp)

        stm_sim   = rho_K + rho_Kp               # from explicit simulation
        stm_flip  = rho_K + fft_xflip(rho_K)       # x-flip shortcut

        diff = stm_sim - stm_flip
        rel = np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(stm_sim**2)) + 1e-30)
        print(f"  mode {j+1}:  STM sim vs x-flip rel = {rel:.2e}")

        vmax = max(stm_sim.max(), stm_flip.max(), 1e-10)

        ax = axes2[j, 0]
        ax.imshow(stm_sim, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax)
        ax.set_title(f"Mode {j+1}: $\\rho_K + \\rho_{{K'}}$  (simulation)", fontsize=10)
        ax.set_ylabel(f"mode {j+1}", fontsize=12)

        ax = axes2[j, 1]
        ax.imshow(stm_flip, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax)
        ax.set_title(f"$\\rho_K(x,y) + \\rho_K(-x,y)$  (shortcut)", fontsize=10)

        ax = axes2[j, 2]
        dmax = max(np.abs(diff).max(), 1e-10)
        ax.imshow(diff, extent=extent, origin="lower", cmap="RdBu_r",
                  vmin=-dmax, vmax=dmax)
        ax.set_title(f"difference\nrel={rel:.1e}", fontsize=10)

    for ax in axes2.flat:
        ax.set_aspect("equal")

    fig2.suptitle("STM:  explicit $\\rho_K + \\rho_{K'}$  vs  "
                  "x-flip shortcut  $\\rho_K(x,y) + \\rho_K(-x,y)$",
                  fontsize=13, y=1.01)
    plt.tight_layout()

    out_path2 = os.path.join(out_dir, "valley_STM_compare.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot saved: {out_path2}")


if __name__ == "__main__":
    main()
