#!/usr/bin/env python3
"""
Pseudospin texture S(r) = psi^ sigma psi / psi^ psi
for conduction vs valence filtered modes.

Analysis includes:
  - S_z sublattice polarization map
  - In-plane (S_x, S_y) vector field
  - Wavefunction vortices: zeros of psi_A and psi_B (well-defined topology)
  - Winding number of the relative phase arg(psi_A* psi_B)
  - Skyrmion density q(r) = S . (dS/dx x dS/dy) / (4pi)
  - k-space winding of arg(psi_B_k / psi_A_k) around k=0 (Berry phase probe)

Note on k-space winding:
  For free Dirac eigenstates, W_k = +1 for both bands (the valence spinor
  -e^{itheta} has the same winding as e^{itheta}).  For bound states in a
  scalar potential V*I, k-mixing via convolution distorts the spinor structure,
  and for non-circular geometries angular momentum mixing further disrupts
  the clean vortex at k=0.  The real-space winding numbers W[psi_A]=-1,
  W[psi_B]=+1 remain the robust topological invariants.
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

# ── parameters ──
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

Gamma_cut       = 8.0
pencil_fraction = 4
L_pencil_max    = 400
n_modes         = 3

# plot limits
xlim = (-3.2, 3.2)
ylim = (-3.2, 3.2)
quiver_step = 8


# ══════════════════════════════════════════════
# Pseudospin computation
# ══════════════════════════════════════════════

def pseudospin_field(phi):
    """S(r) = psi^ sigma psi / psi^ psi. Returns Sx, Sy, Sz, rho."""
    pA, pB = phi[0], phi[1]
    rho = np.abs(pA)**2 + np.abs(pB)**2
    rho_safe = np.where(rho > 1e-30, rho, 1e-30)

    Sx = 2.0 * np.real(np.conj(pA) * pB) / rho_safe
    Sy = 2.0 * np.imag(np.conj(pA) * pB) / rho_safe
    Sz = (np.abs(pA)**2 - np.abs(pB)**2) / rho_safe

    return Sx, Sy, Sz, rho


def density(phi):
    return np.abs(phi[0])**2 + np.abs(phi[1])**2

def normalize(phi, dx):
    norm2 = np.sum(density(phi)) * dx**2
    return phi / np.sqrt(norm2) if norm2 > 0 else phi


# ══════════════════════════════════════════════
# Topological analysis
# ══════════════════════════════════════════════

def winding_number_contour(phase_field, x_np, y_np, cx, cy, radius, n_pts=720):
    """
    Winding number of a phase field around a circular contour.

    W = (1/2pi) oint d(phase).  The phase_field should be a 2D array of angles.
    """
    from scipy.interpolate import RegularGridInterpolator
    # interpolate cos and sin separately to avoid branch-cut artifacts
    cos_f = RegularGridInterpolator((y_np, x_np), np.cos(phase_field),
                                     method="linear", bounds_error=False, fill_value=0.0)
    sin_f = RegularGridInterpolator((y_np, x_np), np.sin(phase_field),
                                     method="linear", bounds_error=False, fill_value=0.0)

    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    pts = np.stack([cy + radius * np.sin(theta),
                    cx + radius * np.cos(theta)], axis=-1)

    phi_vals = np.arctan2(sin_f(pts), cos_f(pts))
    dphi = np.diff(phi_vals, append=phi_vals[0])
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    return np.sum(dphi) / (2*np.pi)


def detect_vortices_plaquette(phase_field, x_np, y_np, rho, rho_thresh=0.02):
    """
    Detect vortices using plaquette method on a phase field.

    Returns list of (x, y, charge).
    """
    rho_norm = rho / max(rho.max(), 1e-30)
    ny, nx = phase_field.shape
    vortices = []

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            rho_local = 0.25 * (rho_norm[iy, ix] + rho_norm[iy, ix+1] +
                                rho_norm[iy+1, ix] + rho_norm[iy+1, ix+1])
            if rho_local < rho_thresh:
                continue

            p = [phase_field[iy, ix], phase_field[iy, ix+1],
                 phase_field[iy+1, ix+1], phase_field[iy+1, ix]]

            total = 0.0
            for k in range(4):
                dp = p[(k+1) % 4] - p[k]
                dp = (dp + np.pi) % (2*np.pi) - np.pi
                total += dp

            charge = round(total / (2*np.pi))
            if abs(charge) >= 1:
                vx = 0.5 * (x_np[ix] + x_np[ix+1])
                vy = 0.5 * (y_np[iy] + y_np[iy+1])
                vortices.append((vx, vy, charge))

    return vortices


def kspace_winding(phi, dx, n, radius_k, n_pts=720):
    """
    W_k = (1/2pi) oint d arg(psi_B_k / psi_A_k) around k=0.

    This is the true chirality topological invariant (Berry phase).
    Expected: +1 for conduction, -1 for valence.
    """
    from scipy.interpolate import RegularGridInterpolator

    pA_k = np.fft.fftshift(np.fft.fft2(phi[0]))
    pB_k = np.fft.fftshift(np.fft.fft2(phi[1]))

    kx = np.fft.fftshift(np.fft.fftfreq(n, d=dx)) * 2 * np.pi

    ratio = pB_k / (pA_k + 1e-30)
    phase = np.angle(ratio)

    cos_f = RegularGridInterpolator((kx, kx), np.cos(phase),
                                     bounds_error=False, fill_value=0)
    sin_f = RegularGridInterpolator((kx, kx), np.sin(phase),
                                     bounds_error=False, fill_value=0)

    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = np.stack([radius_k * np.sin(theta),
                    radius_k * np.cos(theta)], axis=-1)

    phi_vals = np.arctan2(sin_f(pts), cos_f(pts))
    dphi = np.diff(phi_vals, append=phi_vals[0])
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return np.sum(dphi) / (2 * np.pi)


def skyrmion_density(Sx, Sy, Sz, dx):
    """
    Topological charge density q(r) = (1/4pi) S . (dS/dx x dS/dy).
    """
    dSx_dx = np.gradient(Sx, dx, axis=1)
    dSx_dy = np.gradient(Sx, dx, axis=0)
    dSy_dx = np.gradient(Sy, dx, axis=1)
    dSy_dy = np.gradient(Sy, dx, axis=0)
    dSz_dx = np.gradient(Sz, dx, axis=1)
    dSz_dy = np.gradient(Sz, dx, axis=0)

    cross_x = dSy_dx * dSz_dy - dSz_dx * dSy_dy
    cross_y = dSz_dx * dSx_dy - dSx_dx * dSz_dy
    cross_z = dSx_dx * dSy_dy - dSy_dx * dSx_dy

    q = (Sx * cross_x + Sy * cross_y + Sz * cross_z) / (4.0 * np.pi)
    return q


def run_band(band, xp, X, Y, KX, KY, K, dx, X_np, Y_np, V, W, chi_R, sqrt_chi, chi_cpu):
    """Propagate and extract modes for a given band."""
    print(f"\n{'='*50}")
    print(f"  {band} band")
    print(f"{'='*50}")

    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)
    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=theta0, band=band, dx=dx, xp=xp,
    )

    result = propagate(
        psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt,
        save_every=save_every, corr_every=decim, max_record=max_record,
        measure_transmission=False,
    )
    print(f"  Recorded {len(result['C_R'])} corr samples, {len(result['snapshots'])} snapshots")

    print(f"  Harmonic inversion ...")
    spec = extract_resonances(result["C_R"], result["corr_times"], dt * decim,
                              Gamma_cut=Gamma_cut,
                              pencil_fraction=pencil_fraction,
                              L_pencil_max=L_pencil_max)
    nm = min(n_modes, len(spec["omega"]))
    for j in range(nm):
        print(f"    mode {j+1}:  E={spec['E'][j]:.6f}  G={spec['Gamma'][j]:.6f}")

    modes = [time_filter_mode(result["snapshots"], result["snap_times"],
                              spec["omega"][j], chi_cpu)
             for j in range(nm)]

    return modes, spec["E"][:nm], spec["Gamma"][:nm]


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

    shared = dict(xp=xp, X=X, Y=Y, KX=KX, KY=KY, K=K, dx=dx,
                  X_np=X_np, Y_np=Y_np, V=V, W=W, chi_R=chi_R,
                  sqrt_chi=sqrt_chi, chi_cpu=chi_cpu)

    # ── run both bands ──
    modes_c, E_c, G_c = run_band("conduction", **shared)
    modes_v, E_v, G_v = run_band("valence", **shared)

    n_show = min(len(modes_c), len(modes_v))
    extent = [-L/2, L/2, -L/2, L/2]
    a_ell, b_ell = pot_params["a"], pot_params["b"]
    theta_arr = np.linspace(0, 2*np.pi, 200)

    # quiver subsampling
    step = quiver_step
    ix = np.arange(0, n, step)
    iy = np.arange(0, n, step)
    Xq, Yq = np.meshgrid(x_np[ix], x_np[iy], indexing="xy")

    # ══════════════════════════════════════════════
    # Figure 1: full pseudospin texture (4 columns x 2*n_show rows)
    # ══════════════════════════════════════════════
    print("\nPlotting pseudospin texture...")

    fig, axes = plt.subplots(n_show * 2, 4, figsize=(20, 5.0 * n_show * 2))
    if n_show * 2 == 1:
        axes = axes[np.newaxis, :]

    row = 0
    for j in range(n_show):
        for label, phi, Ej, Gj in [("conduction", modes_c[j], E_c[j], G_c[j]),
                                    ("valence",    modes_v[j], E_v[j], G_v[j])]:
            phi = normalize(phi, dx_val)
            Sx, Sy, Sz, rho = pseudospin_field(phi)
            rho_norm = rho / max(rho.max(), 1e-30)
            mask = rho_norm > 0.02

            # Col 0: total density
            ax = axes[row, 0]
            im = ax.imshow(rho_norm, extent=extent, cmap="inferno", origin="lower",
                           vmin=0, vmax=1)
            ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                    "w--", lw=0.8, alpha=0.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            ax.set_ylabel(f"{label} #{j+1}\n$E={Ej:+.3f}$", fontsize=10)
            if row == 0:
                ax.set_title(r"$\rho = |\psi_A|^2 + |\psi_B|^2$", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Col 1: Sz
            ax = axes[row, 1]
            Sz_masked = np.where(mask, Sz, np.nan)
            im = ax.imshow(Sz_masked, extent=extent, cmap="RdBu_r", origin="lower",
                           vmin=-1, vmax=1)
            ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                    "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            if row == 0:
                ax.set_title(r"$S_z = (|\psi_A|^2 - |\psi_B|^2)/\rho$", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Col 2: sublattice RGB overlay
            ax = axes[row, 2]
            rhoA = np.abs(phi[0])**2
            rhoB = np.abs(phi[1])**2
            scale = max(rhoA.max(), rhoB.max(), 1e-30)
            rgb = np.zeros((*rhoA.shape, 3))
            rgb[:, :, 0] = rhoA / scale
            rgb[:, :, 2] = rhoB / scale
            rgb = np.clip(rgb, 0, 1)
            ax.imshow(rgb, extent=extent, origin="lower")
            ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                    "w--", lw=0.8, alpha=0.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            if row == 0:
                ax.set_title(r"Red=$|\psi_A|^2$, Blue=$|\psi_B|^2$", fontsize=11)

            # Col 3: in-plane pseudospin quiver
            ax = axes[row, 3]
            ax.imshow(rho_norm, extent=extent, cmap="gray_r", origin="lower",
                      vmin=0, vmax=1, alpha=0.3)
            Sx_q = Sx[np.ix_(iy, ix)]
            Sy_q = Sy[np.ix_(iy, ix)]
            rho_q = rho_norm[np.ix_(iy, ix)]
            good = rho_q > 0.05
            Sx_show = np.where(good, Sx_q, 0)
            Sy_show = np.where(good, Sy_q, 0)
            S_mag = np.sqrt(Sx_show**2 + Sy_show**2)
            ax.quiver(Xq, Yq, Sx_show, Sy_show,
                      S_mag, cmap="cool", scale=25, width=0.004,
                      headwidth=3, headlength=3, alpha=0.9)
            ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                    "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            if row == 0:
                ax.set_title(r"In-plane $(S_x, S_y)$ texture", fontsize=11)

            row += 1

    fig.suptitle("Pseudospin texture: conduction vs valence", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pseudospin_texture.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ══════════════════════════════════════════════
    # Figure 2: Sz comparison (mode 1 only)
    # ══════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))

    for i, (label, phi, Ej) in enumerate([
        ("conduction", modes_c[0], E_c[0]),
        ("valence",    modes_v[0], E_v[0]),
    ]):
        phi = normalize(phi, dx_val)
        Sx, Sy, Sz, rho = pseudospin_field(phi)
        rho_norm = rho / max(rho.max(), 1e-30)
        mask = rho_norm > 0.02

        ax = axes2[i, 0]
        im = ax.imshow(rho_norm, extent=extent, cmap="inferno", origin="lower",
                       vmin=0, vmax=1)
        ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                "w--", lw=0.8, alpha=0.5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_ylabel(f"{label}\n$E={Ej:+.3f}$", fontsize=12)
        if i == 0:
            ax.set_title(r"Total density $\rho$", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes2[i, 1]
        Sz_masked = np.where(mask, Sz, np.nan)
        im = ax.imshow(Sz_masked, extent=extent, cmap="RdBu_r", origin="lower",
                       vmin=-1, vmax=1)
        ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        if i == 0:
            ax.set_title(r"$S_z(r)$: red=$\psi_A$ dom., blue=$\psi_B$ dom.", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes2[i, 2]
        iy0 = n // 2
        ax.plot(x_np, Sz[iy0, :], "b-", lw=2, label=r"$S_z(x, 0)$")
        ax.plot(x_np, rho_norm[iy0, :], "k--", lw=1, alpha=0.5,
                label=r"$\rho(x,0)$ (scaled)")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("x"); ax.set_ylabel(r"$S_z$")
        ax.set_xlim(*xlim); ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=9)
        if i == 0:
            ax.set_title(r"$S_z$ cross-section at $y=0$", fontsize=12)

    fig2.suptitle(r"Chirality texture: $S_z$ flips between conduction and valence",
                  fontsize=14, y=1.01)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "Sz_comparison.png")
    fig2.savefig(out_path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path2}")

    # ══════════════════════════════════════════════
    # Figure 3: Topological analysis — wavefunction vortices & winding
    #
    # Key insight: the in-plane pseudospin angle phi_S = arg(psi_A* psi_B)
    # has singularities where |S_perp| -> 0 (i.e. S_z -> +-1), making
    # its winding number ill-defined when contours cross these lines.
    #
    # Better approach: analyze the phase of each spinor component separately.
    # arg(psi_A) and arg(psi_B) have vortices at their respective zeros.
    # The relative phase arg(psi_A* psi_B) = arg(psi_B) - arg(psi_A)
    # has vortices where EITHER component vanishes.
    #
    # For the in-plane winding, we also track min(|S_perp|) along the contour
    # to flag when the integral path crosses a singularity.
    # ══════════════════════════════════════════════
    print("\nTopological analysis...")

    fig3, axes3 = plt.subplots(n_show * 2, 4, figsize=(22, 5.0 * n_show * 2))
    if n_show * 2 == 1:
        axes3 = axes3[np.newaxis, :]

    radii = np.linspace(0.15, 2.8, 50)

    row = 0
    for j in range(n_show):
        for label, phi, Ej in [("conduction", modes_c[j], E_c[j]),
                                ("valence",    modes_v[j], E_v[j])]:
            phi = normalize(phi, dx_val)
            pA, pB = phi[0], phi[1]
            Sx, Sy, Sz, rho = pseudospin_field(phi)
            rho_norm = rho / max(rho.max(), 1e-30)

            phase_A = np.angle(pA)
            phase_B = np.angle(pB)
            # relative phase = arg(psi_A* psi_B) = phase_B - phase_A
            phase_rel = np.angle(np.conj(pA) * pB)

            rhoA = np.abs(pA)**2
            rhoB = np.abs(pB)**2

            # ── Col 0: wavefunction vortices (psi_A zeros + psi_B zeros) ──
            ax = axes3[row, 0]
            # background: density
            ax.imshow(rho_norm, extent=extent, cmap="gray_r", origin="lower",
                      vmin=0, vmax=1, alpha=0.4)

            # detect vortices of psi_A (in regions where psi_A has amplitude)
            vort_A = detect_vortices_plaquette(phase_A, x_np, x_np, rhoA,
                                               rho_thresh=0.001 * rhoA.max())
            vort_B = detect_vortices_plaquette(phase_B, x_np, x_np, rhoB,
                                               rho_thresh=0.001 * rhoB.max())

            n_A = len(vort_A)
            n_B = len(vort_B)
            net_A = sum(c for _, _, c in vort_A)
            net_B = sum(c for _, _, c in vort_B)

            for vx, vy, vc in vort_A:
                if abs(vx) < xlim[1] and abs(vy) < ylim[1]:
                    marker = "^" if vc > 0 else "v"
                    ax.plot(vx, vy, marker, color="red", ms=5, mew=0.8,
                            mec="darkred", zorder=5, alpha=0.8)
            for vx, vy, vc in vort_B:
                if abs(vx) < xlim[1] and abs(vy) < ylim[1]:
                    marker = "^" if vc > 0 else "v"
                    ax.plot(vx, vy, marker, color="blue", ms=5, mew=0.8,
                            mec="darkblue", zorder=5, alpha=0.8)

            ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                    "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            ax.set_ylabel(f"{label} #{j+1}\n$E={Ej:+.3f}$", fontsize=10)
            if row == 0:
                ax.set_title("Wavefunction vortices\n"
                             r"red: $\psi_A$ zeros, blue: $\psi_B$ zeros",
                             fontsize=10)
            ax.text(0.05, 0.95,
                    f"$\\psi_A$: {n_A} (net {net_A:+d})\n"
                    f"$\\psi_B$: {n_B} (net {net_B:+d})",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

            # ── Col 1: winding number of arg(psi_A), arg(psi_B), and relative phase ──
            ax = axes3[row, 1]

            from scipy.interpolate import RegularGridInterpolator
            interp_Sx = RegularGridInterpolator((x_np, x_np), Sx,
                                                 method="linear", bounds_error=False, fill_value=0.0)
            interp_Sy = RegularGridInterpolator((x_np, x_np), Sy,
                                                 method="linear", bounds_error=False, fill_value=0.0)

            W_A = []
            W_B = []
            W_rel = []
            Sperp_min = []
            for r in radii:
                W_A.append(winding_number_contour(phase_A, x_np, x_np, 0, 0, r))
                W_B.append(winding_number_contour(phase_B, x_np, x_np, 0, 0, r))
                W_rel.append(winding_number_contour(phase_rel, x_np, x_np, 0, 0, r))

                # track min |S_perp| along contour
                theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
                pts = np.stack([r * np.sin(theta), r * np.cos(theta)], axis=-1)
                sx_v = interp_Sx(pts)
                sy_v = interp_Sy(pts)
                Sperp_min.append(np.min(np.sqrt(sx_v**2 + sy_v**2)))

            ax.plot(radii, W_A, "r-", lw=1.5, label=r"$W[\psi_A]$")
            ax.plot(radii, W_B, "b-", lw=1.5, label=r"$W[\psi_B]$")
            ax.plot(radii, W_rel, "k--", lw=1.5, label=r"$W[\psi_A^*\psi_B]$")

            # shade regions where |S_perp| is small (winding unreliable)
            ax2_twin = ax.twinx()
            ax2_twin.fill_between(radii, 0, Sperp_min, alpha=0.15, color="orange")
            ax2_twin.set_ylabel(r"min $|S_\perp|$ on contour", fontsize=8, color="orange")
            ax2_twin.set_ylim(0, 1.1)
            ax2_twin.tick_params(axis="y", labelcolor="orange", labelsize=7)

            ax.axhline(0, color="gray", lw=0.5, ls=":")
            for w_int in range(-4, 5):
                if w_int != 0:
                    ax.axhline(w_int, color="gray", lw=0.2, ls="--", alpha=0.3)
            ax.set_xlabel("contour radius")
            ax.set_ylabel("winding number $W$")
            ax.set_ylim(-4.5, 4.5)
            ax.legend(fontsize=7, loc="upper left")
            if row == 0:
                ax.set_title("Winding numbers\naround origin", fontsize=10)

            # ── Col 2: skyrmion density ──
            q = skyrmion_density(Sx, Sy, Sz, dx_val)
            q_masked = np.where(rho_norm > 0.02, q, np.nan)
            Q_total = np.nansum(q * (rho_norm > 0.02)) * dx_val**2

            ax = axes3[row, 2]
            qmax = max(np.nanmax(np.abs(q_masked)), 1e-10)
            im = ax.imshow(q_masked, extent=extent, cmap="RdBu_r", origin="lower",
                           vmin=-qmax, vmax=qmax)
            ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                    "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            if row == 0:
                ax.set_title(r"Skyrmion density $q(\mathbf{r})$"
                             "\n" r"$= \frac{1}{4\pi}\mathbf{S}\cdot"
                             r"(\partial_x\mathbf{S}\times\partial_y\mathbf{S})$",
                             fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.text(0.05, 0.95, f"$Q = {Q_total:.3f}$",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, color="k",
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

            # ── Col 3: relative phase map with |S_perp| = 0 contour ──
            ax = axes3[row, 3]
            phase_rel_masked = np.where(rho_norm > 0.02, phase_rel, np.nan)
            im = ax.imshow(phase_rel_masked, extent=extent, cmap="hsv", origin="lower",
                           vmin=-np.pi, vmax=np.pi)
            # contour of |S_perp| ~ 0 (these are the singular lines)
            S_perp = np.sqrt(Sx**2 + Sy**2)
            try:
                ax.contour(x_np, x_np, S_perp, levels=[0.1], colors="white",
                           linewidths=1.0, linestyles="--")
            except Exception:
                pass
            ax.plot(a_ell*np.cos(theta_arr), b_ell*np.sin(theta_arr),
                    "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            if row == 0:
                ax.set_title(r"$\arg(\psi_A^*\psi_B)$" + "\n"
                             r"white dashed: $|S_\perp| = 0.1$", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase [rad]")

            row += 1

    fig3.suptitle("Topological analysis: wavefunction vortices & winding numbers",
                  fontsize=14, y=1.01)
    plt.tight_layout()
    out_path3 = os.path.join(out_dir, "pseudospin_topology.png")
    fig3.savefig(out_path3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {out_path3}")

    # ══════════════════════════════════════════════
    # Figure 4: k-space winding number W_k vs radius_k
    # W_k = winding of arg(psi_B_k / psi_A_k) around k=0
    # True chirality topological invariant (Berry phase)
    # ══════════════════════════════════════════════
    print("\nk-space winding number analysis...")

    dk = 2 * np.pi / (n * dx_val)
    k_radii = np.linspace(0.5 * dk, 8.0, 100)

    fig4, axes4 = plt.subplots(2, n_show, figsize=(6 * n_show, 9), squeeze=False)

    for j in range(n_show):
        # top row: W_k vs radius
        ax = axes4[0, j]
        for label, phi, Ej, color in [
            ("conduction", modes_c[j], E_c[j], "tab:red"),
            ("valence",    modes_v[j], E_v[j], "tab:blue"),
        ]:
            phi = normalize(phi, dx_val)
            Wk_vals = [kspace_winding(phi, dx_val, n, rk) for rk in k_radii]
            ax.plot(k_radii, Wk_vals, "-", color=color, lw=1.5, label=f"{label} $E={Ej:+.3f}$")

        # mark |k| = |E|/vF for reference
        ax.axvline(abs(E_c[j]) / vF, color="tab:red", lw=0.5, ls=":", alpha=0.5)
        ax.axvline(abs(E_v[j]) / vF, color="tab:blue", lw=0.5, ls=":", alpha=0.5)
        ax.axhline(+1, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(-1, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel(r"$k$ radius")
        ax.set_ylabel(r"$W_k$")
        ax.set_ylim(-3.5, 3.5)
        ax.set_title(f"Mode #{j+1}", fontsize=12)
        ax.legend(fontsize=9)

        # bottom row: k-space amplitude along contour
        ax2 = axes4[1, j]
        for label, phi, Ej, color in [
            ("conduction", modes_c[j], E_c[j], "tab:red"),
            ("valence",    modes_v[j], E_v[j], "tab:blue"),
        ]:
            phi = normalize(phi, dx_val)
            pA_k = np.fft.fftshift(np.fft.fft2(phi[0]))
            pB_k = np.fft.fftshift(np.fft.fft2(phi[1]))
            amp = np.abs(pA_k)**2 + np.abs(pB_k)**2
            kx_arr = np.fft.fftshift(np.fft.fftfreq(n, d=dx_val)) * 2 * np.pi
            from scipy.interpolate import RegularGridInterpolator
            amp_f = RegularGridInterpolator((kx_arr, kx_arr), amp,
                                             bounds_error=False, fill_value=0)
            avg_amp = []
            for rk in k_radii:
                theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
                pts = np.stack([rk * np.sin(theta), rk * np.cos(theta)], axis=-1)
                avg_amp.append(np.mean(amp_f(pts)))
            ax2.plot(k_radii, avg_amp, "-", color=color, lw=1.5, label=f"{label}")

        ax2.axvline(abs(E_c[j]) / vF, color="tab:red", lw=0.5, ls=":", alpha=0.5)
        ax2.axvline(abs(E_v[j]) / vF, color="tab:blue", lw=0.5, ls=":", alpha=0.5)
        ax2.set_xlabel(r"$k$ radius")
        ax2.set_ylabel(r"mean $|\psi_k|^2$ on contour")
        ax2.set_yscale("log")
        ax2.legend(fontsize=9)
        ax2.set_title(r"k-space amplitude (dotted: $|k|=|E|/v_F$)", fontsize=10)

    fig4.suptitle(r"k-space winding: $W_k = \frac{1}{2\pi}\oint d\,\arg(\psi_{B,k}/\psi_{A,k})$"
                  "\n(Berry phase / chirality invariant)", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path4 = os.path.join(out_dir, "kspace_winding.png")
    fig4.savefig(out_path4, dpi=180, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {out_path4}")

    # ── summary ──
    print("\n" + "="*70)
    print("  Pseudospin & topology summary")
    print("="*70)
    print(f"  {'':4s} {'':4s}  {'E':>8s}  {'<Sz>':>7s}  {'Q_sky':>7s}  "
          f"{'vort_A':>8s}  {'vort_B':>8s}  {'W_A(1.5)':>9s}  {'W_B(1.5)':>9s}  "
          f"{'W_k(mode)':>10s}")
    # sample W_k over many radii and take mode (most frequent integer)
    wk_sample_radii = np.linspace(0.3, 6.0, 60)
    for j in range(n_show):
        for label, phi, Ej in [("cond", modes_c[j], E_c[j]),
                                ("val",  modes_v[j], E_v[j])]:
            phi = normalize(phi, dx_val)
            pA, pB = phi[0], phi[1]
            Sx, Sy, Sz, rho = pseudospin_field(phi)
            rho_norm = rho / max(rho.max(), 1e-30)

            mask = rho_norm > 0.05
            Sz_avg = np.sum(Sz * rho * mask) / (np.sum(rho * mask) + 1e-30)

            q = skyrmion_density(Sx, Sy, Sz, dx_val)
            Q = np.sum(q * (rho_norm > 0.02)) * dx_val**2

            phase_A = np.angle(pA)
            phase_B = np.angle(pB)
            rhoA = np.abs(pA)**2
            rhoB = np.abs(pB)**2

            vort_A = detect_vortices_plaquette(phase_A, x_np, x_np, rhoA,
                                                rho_thresh=0.001 * rhoA.max())
            vort_B = detect_vortices_plaquette(phase_B, x_np, x_np, rhoB,
                                                rho_thresh=0.001 * rhoB.max())
            net_A = sum(c for _, _, c in vort_A)
            net_B = sum(c for _, _, c in vort_B)

            W_A = winding_number_contour(phase_A, x_np, x_np, 0, 0, 1.5)
            W_B = winding_number_contour(phase_B, x_np, x_np, 0, 0, 1.5)

            # W_k: sample at many radii, round to nearest integer, take mode
            wk_raw = [kspace_winding(phi, dx_val, n, rk) for rk in wk_sample_radii]
            wk_ints = [round(w) for w in wk_raw]
            from collections import Counter
            wk_counts = Counter(wk_ints)
            wk_mode_val, wk_mode_cnt = wk_counts.most_common(1)[0]
            wk_frac = wk_mode_cnt / len(wk_ints)

            print(f"  {j+1:4d} {label:4s}  {Ej:+8.4f}  {Sz_avg:+7.4f}  {Q:+7.3f}  "
                  f"{len(vort_A):3d}({net_A:+d})  {len(vort_B):3d}({net_B:+d})  "
                  f"{W_A:+9.2f}  {W_B:+9.2f}  "
                  f"{wk_mode_val:+d}({wk_frac:.0%})")


if __name__ == "__main__":
    main()
