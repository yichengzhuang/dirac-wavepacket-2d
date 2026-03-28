#!/usr/bin/env python3
"""
Verify Kramers degeneracy and mirror anticommutation for 2D Dirac
in an elliptical potential.

Two mirror operations for C₂ᵥ:
  M_x = σ_y P_x   (x → -x,  commutes with σ_y, anticommutes with σ_x)
  M_y = σ_x P_y   (y → -y,  commutes with σ_x, anticommutes with σ_y)

Key prediction:  M_x M_y = -M_y M_x  (anticommute!)
  → Cannot simultaneously diagonalise both mirrors
  → Projecting onto M_x eigenstates breaks M_y symmetry, and vice versa
  → This is the spin-1/2 double group C̄₂ᵥ; irreps are 2D Kramers pairs
"""

import sys, os, time
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

# ── L=10 harmonic_ellipse parameters ──
n  = 512
L  = 10.0
vF = 1.0
m  = 0.0
dt = 0.01
n_steps = 10000   # T=100, converged for L=10
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


# ═══════════════════════════════════════════════
# Symmetry operations
# ═══════════════════════════════════════════════

def _fft_flip(arr, axis):
    """FFT-consistent reflection: index j → (n-j) % n along given axis."""
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)

def apply_Mx(phi):
    """M_x = σ_y P_x:  (φ_up, φ_dn) → (-i φ_dn(-x,y), i φ_up(-x,y))"""
    up, dn = phi[0], phi[1]
    return np.stack([-1j * _fft_flip(dn, axis=1),
                      1j * _fft_flip(up, axis=1)], axis=0)

def apply_My(phi):
    """M_y = σ_x P_y:  (φ_up, φ_dn) → (φ_dn(x,-y), φ_up(x,-y))"""
    up, dn = phi[0], phi[1]
    return np.stack([_fft_flip(dn, axis=0),
                     _fft_flip(up, axis=0)], axis=0)

def apply_T(phi):
    """Time-reversal T = σ_y K:  (φ_up, φ_dn) → (-i φ_dn*, i φ_up*)"""
    up, dn = phi[0], phi[1]
    return np.stack([-1j * np.conj(dn),
                      1j * np.conj(up)], axis=0)

def density(phi):
    return np.abs(phi[0])**2 + np.abs(phi[1])**2

def normalize(phi, dx):
    norm2 = np.sum(density(phi)) * dx**2
    return phi / np.sqrt(norm2) if norm2 > 0 else phi

def overlap(a, b, dx):
    return (np.vdot(a[0].ravel(), b[0].ravel()) +
            np.vdot(a[1].ravel(), b[1].ravel())) * dx**2

def project_and_normalize(phi, M_phi, dx):
    """Return (φ+Mφ, φ-Mφ) normalised, plus their pre-normalisation norms."""
    p = phi + M_phi
    m = phi - M_phi
    norm_p = np.sqrt(np.sum(density(p)) * dx**2)
    norm_m = np.sqrt(np.sum(density(m)) * dx**2)
    p_n = p / norm_p if norm_p > 1e-10 else p * 0
    m_n = m / norm_m if norm_m > 1e-10 else m * 0
    return p_n, m_n, norm_p, norm_m


def main():
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    os.makedirs(out_dir, exist_ok=True)

    # ── setup ──
    x_np = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx_val = L / n

    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, mask_info = build_analysis_mask(
        POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp,
    )
    chi_cpu = xp.to_numpy(chi_R)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)

    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=theta0, band=band0, dx=dx, xp=xp,
    )

    # ── propagate ──
    print(f"\nPropagating {n_steps} steps (T={n_steps*dt:.0f}) …")
    result = propagate(
        psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt,
        save_every=save_every, corr_every=decim, max_record=max_record,
        measure_transmission=True,
        transmission_every=max(n_steps // 10, 1),
    )

    C_R        = result["C_R"]
    corr_times = result["corr_times"]
    snapshots  = result["snapshots"]
    snap_times = result["snap_times"]
    print(f"Recorded {len(C_R)} corr samples, {len(snapshots)} snapshots")

    # ── extract resonances ──
    print("\nHarmonic inversion …")
    spec = extract_resonances(C_R, corr_times, dt * decim,
                              Gamma_cut=Gamma_cut,
                              pencil_fraction=pencil_fraction,
                              L_pencil_max=L_pencil_max)
    E, Gamma, omega, a = spec["E"], spec["Gamma"], spec["omega"], spec["a"]
    nm = min(n_modes, len(omega))
    for j in range(nm):
        print(f"  mode {j+1}:  E={E[j]:.6f}  Γ={Gamma[j]:.6f}")

    # ── filter modes ──
    print(f"\nFiltering {nm} modes …")
    modes = [time_filter_mode(snapshots, snap_times, omega[j], chi_cpu)
             for j in range(nm)]

    # ═══════════════════════════════════════════════
    # Numerical verification
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print("  Symmetry verification")
    print("="*70)

    # 1. Verify anticommutation: M_x M_y = -M_y M_x on each mode
    print("\n--- Anticommutation test: M_x M_y φ  vs  -M_y M_x φ ---")
    for j in range(nm):
        phi = normalize(modes[j], dx_val)
        MxMy = apply_Mx(apply_My(phi))
        MyMx = apply_My(apply_Mx(phi))
        # Should have MxMy = -MyMx, i.e. MxMy + MyMx = 0
        diff = np.max(np.abs(MxMy + MyMx))
        print(f"  mode {j+1}:  max|M_x M_y φ + M_y M_x φ| = {diff:.2e}  (should be ~0)")

    # 2. Kramers & overlap analysis
    print("\n--- Kramers & mirror overlaps ---")
    for j in range(nm):
        phi = normalize(modes[j], dx_val)
        print(f"\n  Mode {j+1} (E={E[j]:.6f}):")
        print(f"    <φ|Tφ>    = {overlap(phi, apply_T(phi), dx_val):.6f}  (Kramers → 0)")
        print(f"    <φ|M_x φ> = {overlap(phi, apply_Mx(phi), dx_val):.6f}")
        print(f"    <φ|M_y φ> = {overlap(phi, apply_My(phi), dx_val):.6f}")

    # 3. Cross-check: M_y maps between M_x eigenstates
    print("\n--- M_y maps M_x(+) ↔ M_x(-) ---")
    for j in range(nm):
        phi = normalize(modes[j], dx_val)
        Mx_phi = apply_Mx(phi)
        phi_xp, phi_xm, _, _ = project_and_normalize(phi, Mx_phi, dx_val)

        My_xp = apply_My(phi_xp)  # M_y acting on M_x(+) eigenstate
        # This should be proportional to M_x(-) eigenstate
        ov_pp = overlap(phi_xp, My_xp, dx_val)
        ov_mp = overlap(phi_xm, My_xp, dx_val)
        print(f"  mode {j+1}:  <M_x(+)|M_y|M_x(+)> = {ov_pp:.6f}  (should ~0)")
        print(f"            <M_x(-)|M_y|M_x(+)> = {abs(ov_mp):.6f}  (should ~1)")

    # ═══════════════════════════════════════════════
    # Plot: M_x vs M_y projections side by side
    # ═══════════════════════════════════════════════
    fig, axes = plt.subplots(nm, 5, figsize=(20, 4 * nm))
    if nm == 1:
        axes = axes[np.newaxis, :]

    extent = [-L/2, L/2, -L/2, L/2]

    for j in range(nm):
        phi = normalize(modes[j], dx_val)

        Mx_phi = apply_Mx(phi)
        My_phi = apply_My(phi)

        phi_xp, phi_xm, nxp, nxm = project_and_normalize(phi, Mx_phi, dx_val)
        phi_yp, phi_ym, nyp, nym = project_and_normalize(phi, My_phi, dx_val)

        rho_orig = density(phi)
        rho_xp   = density(phi_xp)
        rho_xm   = density(phi_xm)
        rho_yp   = density(phi_yp)
        rho_ym   = density(phi_ym)

        vmax = max(rho_orig.max(), 1e-10)

        ax = axes[j, 0]
        ax.imshow(rho_orig, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax)
        ax.set_title(f"Mode {j+1}: $|\\phi|^2$\nE={E[j]:.4f}", fontsize=11)
        ax.set_ylabel(f"mode {j+1}", fontsize=12)

        ax = axes[j, 1]
        ax.imshow(rho_xp, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax)
        ax.set_title(f"$M_x$-even: $\\phi + M_x\\phi$\nx-symmetric", fontsize=11)

        ax = axes[j, 2]
        ax.imshow(rho_xm, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax)
        ax.set_title(f"$M_x$-odd: $\\phi - M_x\\phi$\nx-symmetric", fontsize=11)

        ax = axes[j, 3]
        ax.imshow(rho_yp, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax)
        ax.set_title(f"$M_y$-even: $\\phi + M_y\\phi$\ny-symmetric", fontsize=11)

        ax = axes[j, 4]
        ax.imshow(rho_ym, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax)
        ax.set_title(f"$M_y$-odd: $\\phi - M_y\\phi$\ny-symmetric", fontsize=11)

    for ax in axes.flat:
        ax.set_aspect("equal")

    fig.suptitle("$M_x = \\sigma_y P_x$  vs  $M_y = \\sigma_x P_y$  projections  "
                 "—  $M_x M_y = -M_y M_x$  (cannot diagonalise both)",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "kramers_Mx_vs_My.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # ═══════════════════════════════════════════════
    # STM image: incoherent valley sum
    #   ρ_STM = ρ_K(x,y) + ρ_K'(x,y)
    #         = ρ_K(x,y) + ρ_K(-x,y)
    # Automatically has full C₂ᵥ symmetry.
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print("  STM image: ρ_STM = ρ_K + ρ_K'  (incoherent valley sum)")
    print("="*70)

    fig, axes = plt.subplots(nm, 3, figsize=(14, 4 * nm))
    if nm == 1:
        axes = axes[np.newaxis, :]

    for j in range(nm):
        phi = normalize(modes[j], dx_val)
        rho_K  = density(phi)                 # K valley
        rho_Kp = np.roll(np.flip(rho_K, axis=1), 1, axis=1)  # K' = ρ_K(-x, y)
        rho_STM = rho_K + rho_Kp             # incoherent sum

        vmax_K   = rho_K.max()
        vmax_STM = rho_STM.max()

        ax = axes[j, 0]
        ax.imshow(rho_K, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax_K)
        ax.set_title(f"Mode {j+1}:  $\\rho_K$  (single valley)\n"
                     f"E={E[j]:.4f}, $\\Gamma$={Gamma[j]:.4f}", fontsize=11)
        ax.set_ylabel(f"mode {j+1}", fontsize=12)

        ax = axes[j, 1]
        ax.imshow(rho_Kp, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax_K)
        ax.set_title(f"$\\rho_{{K'}}(x,y) = \\rho_K(-x,y)$", fontsize=11)

        ax = axes[j, 2]
        im = ax.imshow(rho_STM, extent=extent, origin="lower", cmap="inferno",
                       vmin=0, vmax=vmax_STM)
        ax.set_title(f"$\\rho_{{\\mathrm{{STM}}}} = \\rho_K + \\rho_{{K'}}$\n"
                     f"(full $C_{{2v}}$ symmetry)", fontsize=11)

    for ax in axes.flat:
        ax.set_aspect("equal")

    fig.suptitle("Simulated STM:  incoherent valley sum  "
                 "$\\rho_{\\mathrm{STM}} = \\rho_K(x,y) + \\rho_K(-x,y)$",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "STM_valley_sum.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
