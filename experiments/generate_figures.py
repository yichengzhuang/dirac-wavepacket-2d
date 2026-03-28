#!/usr/bin/env python3
"""
Generate publication-quality figures for symmetry_analysis.tex.

Runs one L=10 propagation, extracts modes, and outputs 5 targeted figures
into figures/ directory.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances, time_filter_mode
from potentials_2d import harmonic_ellipse

# ── parameters (L=10 harmonic_ellipse) ──
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

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ═══════════════════════════════════════════════
# Symmetry operations (from kramers_test.py)
# ═══════════════════════════════════════════════

def apply_Mx(phi):
    up, dn = phi[0], phi[1]
    return np.stack([-1j * dn[:, ::-1], 1j * up[:, ::-1]], axis=0)

def apply_My(phi):
    up, dn = phi[0], phi[1]
    return np.stack([dn[::-1, :], up[::-1, :]], axis=0)

def apply_T(phi):
    up, dn = phi[0], phi[1]
    return np.stack([-1j * np.conj(dn), 1j * np.conj(up)], axis=0)

def density(phi):
    return np.abs(phi[0])**2 + np.abs(phi[1])**2

def normalize(phi, dx):
    norm2 = np.sum(density(phi)) * dx**2
    return phi / np.sqrt(norm2) if norm2 > 0 else phi

def overlap(a, b, dx):
    return (np.vdot(a[0].ravel(), b[0].ravel()) +
            np.vdot(a[1].ravel(), b[1].ravel())) * dx**2


# ═══════════════════════════════════════════════
# Figure generators
# ═══════════════════════════════════════════════

def fig_potential_axes(X_np, Y_np, V_np):
    """Fig 1: Elliptical potential with C₂ᵥ mirror axes."""
    fig, ax = plt.subplots(figsize=(5, 4))
    extent = [-L/2, L/2, -L/2, L/2]

    im = ax.imshow(V_np, extent=extent, origin="lower", cmap="viridis",
                   vmin=0, vmax=10)
    # Equipotential contours
    ax.contour(X_np, Y_np, V_np, levels=[1, 3, 5, 8], colors="white",
               linewidths=0.8, alpha=0.6)

    # Mirror axes
    ax.axhline(0, color="red", ls="--", lw=1.5, alpha=0.8)
    ax.axvline(0, color="cyan", ls="--", lw=1.5, alpha=0.8)
    ax.text(4.3, 0.2, "$P_y$", color="red", fontsize=14, fontweight="bold")
    ax.text(0.2, 4.0, "$P_x$", color="cyan", fontsize=14, fontweight="bold")

    ax.set_xlabel("$x$", fontsize=13)
    ax.set_ylabel("$y$", fontsize=13)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="$V(x,y)$", shrink=0.85)
    ax.set_title("Elliptical potential  $V = \\frac{1}{2}k_h(x^2/a^2 + y^2/b^2)$",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_potential_axes.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [1/5] fig_potential_axes.png")


def fig_kramers_Tphi(phi, E, Gamma, dx_val):
    """Fig 2: |φ|² vs |Tφ|² — Kramers partners have identical density."""
    phi_n = normalize(phi, dx_val)
    T_phi = apply_T(phi_n)
    ov = overlap(phi_n, T_phi, dx_val)

    rho = density(phi_n)
    rho_T = density(T_phi)
    extent = [-L/2, L/2, -L/2, L/2]
    vmax = rho.max()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    axes[0].imshow(rho, extent=extent, origin="lower", cmap="inferno",
                   vmin=0, vmax=vmax)
    axes[0].set_title("$|\\phi|^2$", fontsize=13)
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$y$")

    axes[1].imshow(rho_T, extent=extent, origin="lower", cmap="inferno",
                   vmin=0, vmax=vmax)
    axes[1].set_title("$|T\\phi|^2 = |\\sigma_y \\phi^*|^2$", fontsize=13)
    axes[1].set_xlabel("$x$")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-2.5, 2.5)

    fig.suptitle(f"Kramers partners:  $E = {E:.4f}$,  "
                 f"$\\langle\\phi|T\\phi\\rangle = {ov:.1e}$  (orthogonal)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_kramers_Tphi.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [2/5] fig_kramers_Tphi.png")


def fig_Mx_vs_My(phi, E, dx_val):
    """Fig 3: M_x vs M_y projections — cannot diagonalise both."""
    phi_n = normalize(phi, dx_val)
    Mx_phi = apply_Mx(phi_n)
    My_phi = apply_My(phi_n)

    def proj(phi_, M_phi_):
        p = phi_ + M_phi_
        m = phi_ - M_phi_
        np_ = np.sqrt(np.sum(density(p)) * dx_val**2)
        nm_ = np.sqrt(np.sum(density(m)) * dx_val**2)
        return (p / np_ if np_ > 1e-10 else p * 0,
                m / nm_ if nm_ > 1e-10 else m * 0)

    xp, xm = proj(phi_n, Mx_phi)
    yp, ym = proj(phi_n, My_phi)

    extent = [-L/2, L/2, -L/2, L/2]
    xlim, ylim = (-3.5, 3.5), (-2.5, 2.5)
    rhos = [density(phi_n), density(xp), density(xm), density(yp), density(ym)]
    titles = ["$|\\phi|^2$ (raw)",
              "$M_x$-even\n($x$-symmetric)",
              "$M_x$-odd\n($x$-symmetric)",
              "$M_y$-even\n($y$-symmetric)",
              "$M_y$-odd\n($y$-symmetric)"]

    # Use per-panel vmax so dim panels are visible
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
    for i, (ax, rho, title) in enumerate(zip(axes, rhos, titles)):
        vm = rho.max() if rho.max() > 1e-15 else 1e-10
        ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vm)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        if i == 0:
            ax.set_ylabel("$y$")
        ax.set_xlabel("$x$")

    fig.suptitle(f"$M_x$ vs $M_y$ projections  ($E = {E:.4f}$)  —  "
                 "$\\{M_x, M_y\\} = 0$: cannot diagonalise both",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_Mx_vs_My.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [3/5] fig_Mx_vs_My.png")


def fig_Mx_projection(phi, E, dx_val):
    """Fig 4: M_x projection — raw φ → even φ⁺ + odd φ⁻."""
    phi_n = normalize(phi, dx_val)
    Mx_phi = apply_Mx(phi_n)

    p = phi_n + Mx_phi
    m = phi_n - Mx_phi
    np_ = np.sqrt(np.sum(density(p)) * dx_val**2)
    nm_ = np.sqrt(np.sum(density(m)) * dx_val**2)
    if np_ > 1e-10: p = p / np_
    if nm_ > 1e-10: m = m / nm_

    extent = [-L/2, L/2, -L/2, L/2]
    xlim, ylim = (-3.5, 3.5), (-2.5, 2.5)
    rhos = [density(phi_n), density(p), density(m)]
    titles = [
        "$|\\phi|^2$  (filtered mode)",
        "$|\\phi^{(+)}|^2 = |\\phi + \\sigma_y\\phi(-x,y)|^2$\n$M_x$-even",
        "$|\\phi^{(-)}|^2 = |\\phi - \\sigma_y\\phi(-x,y)|^2$\n$M_x$-odd",
    ]

    vmax = max(r.max() for r in rhos)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, (ax, rho, title) in enumerate(zip(axes, rhos, titles)):
        ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        if i == 0:
            ax.set_ylabel("$y$")

    fig.suptitle(f"Symmetry projection from single propagation  ($E = {E:.4f}$)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_Mx_projection.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [4/5] fig_Mx_projection.png")


def fig_STM_valley_sum(phi, E, dx_val):
    """Fig 5: STM = ρ_K + ρ_K' = ρ_K(x,y) + ρ_K(-x,y)."""
    phi_n = normalize(phi, dx_val)
    rho_K  = density(phi_n)
    rho_Kp = rho_K[:, ::-1]
    rho_STM = rho_K + rho_Kp

    extent = [-L/2, L/2, -L/2, L/2]
    xlim, ylim = (-3.5, 3.5), (-2.5, 2.5)
    vmax_K = rho_K.max()
    vmax_S = rho_STM.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].imshow(rho_K, extent=extent, origin="lower", cmap="inferno",
                   vmin=0, vmax=vmax_K)
    axes[0].set_title("$\\rho_K(x,y)$  (single valley)", fontsize=11)
    axes[0].set_ylabel("$y$")

    axes[1].imshow(rho_Kp, extent=extent, origin="lower", cmap="inferno",
                   vmin=0, vmax=vmax_K)
    axes[1].set_title("$\\rho_{K'}(x,y) = \\rho_K(-x,y)$", fontsize=11)

    axes[2].imshow(rho_STM, extent=extent, origin="lower", cmap="inferno",
                   vmin=0, vmax=vmax_S)
    axes[2].set_title("$\\rho_{\\mathrm{STM}} = \\rho_K + \\rho_{K'}$\n"
                      "(full $C_{2v}$ symmetry)", fontsize=11)

    for ax in axes:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")

    fig.suptitle(f"STM reconstruction from single-valley simulation  ($E = {E:.4f}$)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_STM_valley_sum.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [5/5] fig_STM_valley_sum.png")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    os.makedirs(FIGDIR, exist_ok=True)

    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    # ── setup ──
    x_np = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx_val = L / n

    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    V_np = xp.to_numpy(V)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(
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

    # ── Fig 1: potential (no propagation needed) ──
    fig_potential_axes(X_np, Y_np, V_np)

    # ── propagate ──
    print(f"\nPropagating {n_steps} steps (T={n_steps*dt:.0f}) ...")
    result = propagate(
        psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt,
        save_every=save_every, corr_every=decim, max_record=max_record,
    )

    C_R = result["C_R"]
    corr_times = result["corr_times"]
    snapshots = result["snapshots"]
    snap_times = result["snap_times"]

    # ── extract resonances ──
    print("Harmonic inversion ...")
    spec = extract_resonances(C_R, corr_times, dt * decim,
                              Gamma_cut=Gamma_cut,
                              pencil_fraction=pencil_fraction,
                              L_pencil_max=L_pencil_max)
    E, Gamma, omega = spec["E"], spec["Gamma"], spec["omega"]
    print(f"  mode 1: E={E[0]:.6f}  Gamma={Gamma[0]:.6f}")

    # ── filter mode 1 ──
    phi = time_filter_mode(snapshots, snap_times, omega[0], chi_cpu)

    # ── generate remaining figures ──
    print("\nGenerating figures ...")
    fig_kramers_Tphi(phi, E[0], Gamma[0], dx_val)
    fig_Mx_vs_My(phi, E[0], dx_val)
    fig_Mx_projection(phi, E[0], dx_val)
    fig_STM_valley_sum(phi, E[0], dx_val)

    print(f"\nAll figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
