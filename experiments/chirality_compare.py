#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chirality comparison: conduction vs valence band Dirac propagation.

Runs the same simulation twice with opposite helicity initial wavepackets,
extracts resonance spectra for both, and generates comparison plots.

Uses the accelerated backend (accel/) for CPU/GPU speed.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances, time_filter_mode
from visualization import _save


# ╔═══════════════════════════════════════════════════════════════╗
# ║                   BACKEND & PARAMETERS                       ║
# ╚═══════════════════════════════════════════════════════════════╝

BACKEND = "auto"

# -- grid --
n  = 512
L  = 10.0

# -- Dirac --
vF = 1.0
m  = 0.0

# -- time --
dt      = 0.01
n_steps = 10000

# -- recording cadence --
decim      = 10
save_every = 10
max_record = 2500

# -- potential --
POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

# -- CAP --
cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

# -- analysis mask --
mask_safety_factor = 0.95
mask_margin        = 1.30

# -- initial wavepacket --
x0, y0   = 0.0, 0.0
kx0, ky0 = -1.0, 0.0
sigma0   = 0.6
theta0   = None

# -- spectral extraction --
Gamma_cut       = 8.0
pencil_fraction = 4
L_pencil_max    = 400
n_modes         = 5

# -- comparison --
N_COMPARE_MODES = 5


# ╔═══════════════════════════════════════════════════════════════╗
# ║                     HELPERS                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

def make_output_folder():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    here = os.path.dirname(os.path.abspath(__file__))
    name = os.path.join(here, "Results",
        f"{ts}__chirality_compare__"
        f"{POTENTIAL_OPTION}__"
        f"kx={kx0}_ky={ky0}__"
        f"n={n}_dt={dt}_T={n_steps*dt:.0f}"
    )
    os.makedirs(name, exist_ok=True)
    return name


def density(phi):
    return np.abs(phi[0])**2 + np.abs(phi[1])**2


def apply_sigma_z(phi):
    """sigma_z (psi_A, psi_B) = (psi_A, -psi_B)"""
    return np.stack([phi[0], -phi[1]], axis=0)


def overlap(a, b, dx):
    """Spinor inner product <a|b> with dx^2 measure."""
    return (np.vdot(a[0], b[0]) + np.vdot(a[1], b[1])) * dx**2


def normalize(phi, dx):
    norm2 = np.sum(density(phi)) * dx**2
    return phi / np.sqrt(norm2) if norm2 > 0 else phi


def pair_modes_by_abs_energy(E_ref, E_target):
    """Match modes by closest |E|. Returns index array into E_target."""
    indices = []
    used = set()
    for e in E_ref:
        dists = np.abs(np.abs(E_target) - np.abs(e))
        for idx in np.argsort(dists):
            if idx not in used:
                indices.append(idx)
                used.add(idx)
                break
    return np.array(indices)


# ╔═══════════════════════════════════════════════════════════════╗
# ║                  SINGLE-BAND RUN                             ║
# ╚═══════════════════════════════════════════════════════════════╝

def run_band(band, xp, X, Y, X_np, Y_np, KX, KY, K, dx, V, W,
             chi_R, chi_cpu, sqrt_chi, halfV, kinetic, folder):
    sub = os.path.join(folder, band)
    os.makedirs(sub, exist_ok=True)

    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=theta0, band=band, dx=dx, xp=xp,
    )

    print(f"\n  [{band}] Propagating {n_steps} steps ...")
    result = propagate(
        psi0_up, psi0_dn, halfV, kinetic, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt,
        save_every=save_every, corr_every=decim, max_record=max_record,
        measure_transmission=True,
        transmission_every=max(n_steps // 10, 1),
    )

    C_R        = result["C_R"]
    corr_times = result["corr_times"]
    dt_corr    = dt * decim
    snapshots  = result["snapshots"]
    snap_times = result["snap_times"]

    print(f"  [{band}] Recorded {len(C_R)} correlation samples, "
          f"{len(snapshots)} snapshots")

    print(f"  [{band}] Harmonic inversion ...")
    spec = extract_resonances(
        C_R, corr_times, dt_corr,
        Gamma_cut=Gamma_cut,
        pencil_fraction=pencil_fraction,
        L_pencil_max=L_pencil_max,
    )

    E, Gamma, omega, a = spec["E"], spec["Gamma"], spec["omega"], spec["a"]

    nm = min(n_modes, len(omega))
    modes = []
    if nm > 0:
        print(f"  [{band}] Filtering {nm} modes ...")
        modes = [time_filter_mode(snapshots, snap_times, omega[j], chi_cpu)
                 for j in range(nm)]

    for j in range(min(n_modes, len(E))):
        print(f"    mode {j+1}:  E={E[j]:.6f}  "
              f"Gamma={Gamma[j]:.6f}  |a|={abs(a[j]):.4e}")

    return {
        "E": E, "Gamma": Gamma, "omega": omega, "a": a, "modes": modes,
        "C_R": C_R, "corr_times": corr_times,
    }


# ╔═══════════════════════════════════════════════════════════════╗
# ║                 COMPARISON PLOTS                             ║
# ╚═══════════════════════════════════════════════════════════════╝

def plot_spectrum_comparison(res_c, res_v, folder):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_c = min(N_COMPARE_MODES, len(res_c["E"]))
    n_v = min(N_COMPARE_MODES, len(res_v["E"]))
    if n_c > 0:
        sizes_c = np.abs(res_c["a"][:n_c])
        sizes_c = 50 + 200 * sizes_c / (sizes_c.max() + 1e-30)
        ax.scatter(res_c["E"][:n_c], res_c["Gamma"][:n_c] / 2,
                   s=sizes_c, c="tab:blue", alpha=0.7,
                   edgecolors="k", linewidths=0.5, label="conduction", zorder=3)
    if n_v > 0:
        sizes_v = np.abs(res_v["a"][:n_v])
        sizes_v = 50 + 200 * sizes_v / (sizes_v.max() + 1e-30)
        ax.scatter(res_v["E"][:n_v], res_v["Gamma"][:n_v] / 2,
                   s=sizes_v, c="tab:red", alpha=0.7,
                   edgecolors="k", linewidths=0.5, label="valence", zorder=3)
    ax.set_xlabel("Re(E)")
    ax.set_ylabel(r"$\Gamma/2$")
    ax.set_title("Resonance spectrum: conduction vs valence")
    ax.legend()
    fig.tight_layout()
    _save(fig, folder, "spectrum_comparison.png")


def plot_correlation_comparison(res_c, res_v, folder):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(res_c["corr_times"], np.abs(res_c["C_R"]),
                 label="conduction", alpha=0.8)
    axes[0].plot(res_v["corr_times"], np.abs(res_v["C_R"]),
                 label="valence", alpha=0.8)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("|C_R(t)|")
    axes[0].set_title("Correlation magnitude"); axes[0].legend()

    axes[1].plot(res_c["corr_times"], np.real(res_c["C_R"]),
                 label="conduction", alpha=0.8)
    axes[1].plot(res_v["corr_times"], np.real(res_v["C_R"]),
                 label="valence", alpha=0.8)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("Re C_R(t)")
    axes[1].set_title("Correlation real part"); axes[1].legend()
    fig.tight_layout()
    _save(fig, folder, "correlation_comparison.png")


def plot_mode_comparison(res_c, res_v, folder):
    n_c = min(N_COMPARE_MODES, len(res_c["modes"]))
    n_v = min(N_COMPARE_MODES, len(res_v["modes"]))
    n_show = min(n_c, n_v)
    if n_show == 0:
        return
    pair_idx = pair_modes_by_abs_energy(res_c["E"][:n_show], res_v["E"][:n_v])

    ext = [-L / 2, L / 2, -L / 2, L / 2]
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 7))
    if n_show == 1:
        axes = axes.reshape(2, 1)
    for j in range(n_show):
        rho_c = density(res_c["modes"][j])
        jv = pair_idx[j]
        rho_v = density(res_v["modes"][jv])
        vmax = max(rho_c.max(), rho_v.max())
        axes[0, j].imshow(rho_c, extent=ext, origin="lower", vmin=0, vmax=vmax)
        axes[0, j].set_title(f"cond #{j+1}\nE={res_c['E'][j]:.4f}\n"
                             r"$\Gamma$" + f"={res_c['Gamma'][j]:.4f}", fontsize=9)
        axes[1, j].imshow(rho_v, extent=ext, origin="lower", vmin=0, vmax=vmax)
        axes[1, j].set_title(f"val #{jv+1}\nE={res_v['E'][jv]:.4f}\n"
                             r"$\Gamma$" + f"={res_v['Gamma'][jv]:.4f}", fontsize=9)
    axes[0, 0].set_ylabel("conduction")
    axes[1, 0].set_ylabel("valence")
    fig.suptitle("Mode density comparison", fontsize=13)
    fig.tight_layout()
    _save(fig, folder, "mode_comparison.png")


def plot_sublattice_polarization(res_c, res_v, folder):
    n_c = min(N_COMPARE_MODES, len(res_c["modes"]))
    n_v = min(N_COMPARE_MODES, len(res_v["modes"]))
    n_show = min(n_c, n_v)
    if n_show == 0:
        return
    pair_idx = pair_modes_by_abs_energy(res_c["E"][:n_show], res_v["E"][:n_v])

    ext = [-L / 2, L / 2, -L / 2, L / 2]
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 7))
    if n_show == 1:
        axes = axes.reshape(2, 1)
    eps = 1e-15
    for j in range(n_show):
        phi_c = res_c["modes"][j]
        rho_c = density(phi_c)
        P_c = (np.abs(phi_c[0])**2 - np.abs(phi_c[1])**2) / (rho_c + eps)
        jv = pair_idx[j]
        phi_v = res_v["modes"][jv]
        rho_v = density(phi_v)
        P_v = (np.abs(phi_v[0])**2 - np.abs(phi_v[1])**2) / (rho_v + eps)

        im0 = axes[0, j].imshow(P_c, extent=ext, origin="lower",
                                 cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0, j].set_title(f"cond #{j+1}\nE={res_c['E'][j]:.4f}", fontsize=9)
        im1 = axes[1, j].imshow(P_v, extent=ext, origin="lower",
                                 cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, j].set_title(f"val #{jv+1}\nE={res_v['E'][jv]:.4f}", fontsize=9)
    axes[0, 0].set_ylabel("conduction")
    axes[1, 0].set_ylabel("valence")
    fig.suptitle(r"Sublattice polarization $P = (\rho_A - \rho_B)/(\rho_A + \rho_B)$",
                 fontsize=13)
    fig.colorbar(im1, ax=axes, shrink=0.6, label="P")
    fig.tight_layout()
    _save(fig, folder, "sublattice_polarization.png")


# ╔═══════════════════════════════════════════════════════════════╗
# ║              EIGENSTATE ANALYSIS: sigma_z test               ║
# ╚═══════════════════════════════════════════════════════════════╝

def pair_pos_neg_modes(E_c, E_v):
    """Pair E>0 from cond with E<0 from val by closest |E|."""
    pos = [(j, E_c[j]) for j in range(len(E_c)) if E_c[j] > 0]
    neg = [(j, E_v[j]) for j in range(len(E_v)) if E_v[j] < 0]
    if not pos or not neg:
        return []
    E_neg_abs = np.array([abs(x[1]) for x in neg])
    pairs = []
    used = set()
    for jc, Ec in pos:
        dists = np.abs(E_neg_abs - abs(Ec))
        for idx in np.argsort(dists):
            if idx not in used:
                used.add(idx)
                jv, Ev = neg[idx]
                pairs.append((jc, jv, Ec, Ev))
                break
    return pairs


def analyze_eigenstates(res_c, res_v, dx, folder):
    """Test sigma_z mapping between +E and -E eigenstates."""
    modes_c = res_c["modes"]
    modes_v = res_v["modes"]
    pairs = pair_pos_neg_modes(res_c["E"], res_v["E"])
    n_pair = min(N_COMPARE_MODES, len(pairs))
    if n_pair == 0:
        print("  Not enough +E/-E modes for eigenstate analysis.")
        return

    # Also filter to modes that have been extracted (index < len(modes))
    pairs = [(jc, jv, Ec, Ev) for jc, jv, Ec, Ev in pairs
             if jc < len(modes_c) and jv < len(modes_v)]
    n_pair = min(N_COMPARE_MODES, len(pairs))
    if n_pair == 0:
        print("  Not enough filtered modes for eigenstate analysis.")
        return

    print("\n" + "=" * 75)
    print("  EIGENSTATE ANALYSIS: sigma_z mapping test")
    print("  Prediction: if chiral sym exact, sigma_z|+E> = |-E>")
    print("    => |<sigma_z phi_c | phi_v>| = 1")
    print("    => density(+E) = density(-E)")
    print("    => P(+E) = P(-E)  (sigma_z does not change |psi_A|^2 or |psi_B|^2)")
    print("=" * 75)

    print(f"\n  {'#':>2s}  {'E_cond':>10s}  {'E_val':>10s}  "
          f"{'|<sz c|v>|':>12s}  {'rho_RMS_diff':>14s}  {'P_RMS_diff':>14s}")
    print("-" * 75)

    ext = [-L / 2, L / 2, -L / 2, L / 2]
    fig, axes = plt.subplots(n_pair, 5, figsize=(22, 4 * n_pair))
    if n_pair == 1:
        axes = axes.reshape(1, -1)

    for k in range(n_pair):
        jc, jv, Ec, Ev = pairs[k]
        phi_c = normalize(modes_c[jc], dx)
        phi_v = normalize(modes_v[jv], dx)

        # sigma_z transform
        sz_c = apply_sigma_z(phi_c)
        ov = overlap(sz_c, phi_v, dx)
        # phase-align
        if abs(ov) > 1e-15:
            sz_c = sz_c * np.exp(-1j * np.angle(ov))
            ov = abs(ov)

        # density comparison
        rho_c = density(phi_c)
        rho_v = density(phi_v)
        rho_norm = np.sqrt(np.sum(rho_c**2) * dx**2 + 1e-30)
        rho_diff_rms = np.sqrt(np.sum((rho_c - rho_v)**2) * dx**2) / rho_norm

        # sublattice polarization comparison
        eps = 1e-15
        P_c = (np.abs(phi_c[0])**2 - np.abs(phi_c[1])**2) / (rho_c + eps)
        P_v = (np.abs(phi_v[0])**2 - np.abs(phi_v[1])**2) / (rho_v + eps)
        P_diff_rms = np.sqrt(np.sum((P_c - P_v)**2 * rho_c) * dx**2) / \
                     np.sqrt(np.sum(rho_c) * dx**2 + 1e-30)

        print(f"  {k+1:2d}  {Ec:+10.4f}  {Ev:+10.4f}  "
              f"{abs(ov):12.6f}  {rho_diff_rms:14.6f}  {P_diff_rms:14.6f}")

        # Plot 5 columns: rho_c | rho_v | rho_c - rho_v | P_c | P_v
        vmax_rho = max(rho_c.max(), rho_v.max())

        axes[k, 0].imshow(rho_c, extent=ext, origin="lower", vmin=0, vmax=vmax_rho)
        axes[k, 0].set_ylabel(f"E=({Ec:+.2f}, {Ev:+.2f})", fontsize=9)

        axes[k, 1].imshow(rho_v, extent=ext, origin="lower", vmin=0, vmax=vmax_rho)

        diff = rho_c - rho_v
        vabs = max(abs(diff.min()), abs(diff.max()), 1e-15)
        axes[k, 2].imshow(diff, extent=ext, origin="lower",
                          cmap="RdBu_r", vmin=-vabs, vmax=vabs)

        axes[k, 3].imshow(P_c, extent=ext, origin="lower",
                          cmap="RdBu_r", vmin=-1, vmax=1)

        axes[k, 4].imshow(P_v, extent=ext, origin="lower",
                          cmap="RdBu_r", vmin=-1, vmax=1)

    # Column titles
    col_titles = [r"$\rho(+E)$  (cond)", r"$\rho(-E)$  (val)",
                  r"$\rho(+E) - \rho(-E)$", "P(+E)", "P(-E)"]
    for col in range(5):
        axes[0, col].set_title(col_titles[col], fontsize=10)

    print("-" * 75)
    print("  |<sz c|v>| = 1  =>  sigma_z maps +E to -E exactly")
    print("  rho_RMS_diff = 0  =>  same density pattern")
    print("  P_RMS_diff = 0  =>  same sublattice polarization")
    print("=" * 75)

    fig.suptitle(r"Eigenstate analysis: $\sigma_z |{+E}\rangle \approx |{-E}\rangle$ ?",
                 fontsize=14)
    fig.tight_layout()
    _save(fig, folder, "eigenstate_analysis.png")


# ╔═══════════════════════════════════════════════════════════════╗
# ║                        MAIN                                  ║
# ╚═══════════════════════════════════════════════════════════════╝

def main():
    xp = get_backend(BACKEND)
    print(f"Backend: {xp.name}")

    folder = make_output_folder()
    print(f"Output -> {folder}/")

    x_np = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

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
    halfV, kinetic = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)

    # -- run both bands --
    print("=" * 60)
    print("  CONDUCTION BAND (helicity +1)")
    print("=" * 60)
    res_cond = run_band("conduction", xp, X, Y, X_np, Y_np, KX, KY, K, dx,
                        V, W, chi_R, chi_cpu, sqrt_chi, halfV, kinetic, folder)

    print()
    print("=" * 60)
    print("  VALENCE BAND (helicity -1)")
    print("=" * 60)
    res_val = run_band("valence", xp, X, Y, X_np, Y_np, KX, KY, K, dx,
                       V, W, chi_R, chi_cpu, sqrt_chi, halfV, kinetic, folder)

    # -- comparison plots --
    print("\nGenerating comparison plots ...")
    plot_spectrum_comparison(res_cond, res_val, folder)
    plot_correlation_comparison(res_cond, res_val, folder)
    plot_mode_comparison(res_cond, res_val, folder)
    plot_sublattice_polarization(res_cond, res_val, folder)

    # -- summary --
    print("\n  Conduction band top modes:")
    for j in range(min(N_COMPARE_MODES, len(res_cond["E"]))):
        print(f"    #{j+1}  E={res_cond['E'][j]:+.6f}  "
              f"Gamma={res_cond['Gamma'][j]:.6f}  |a|={abs(res_cond['a'][j]):.4e}")
    print("\n  Valence band top modes:")
    for j in range(min(N_COMPARE_MODES, len(res_val["E"]))):
        print(f"    #{j+1}  E={res_val['E'][j]:+.6f}  "
              f"Gamma={res_val['Gamma'][j]:.6f}  |a|={abs(res_val['a'][j]):.4e}")

    # -- chiral symmetry test --
    pairs = pair_pos_neg_modes(res_cond["E"], res_val["E"])
    if pairs:
        print("\n  Chiral symmetry test: |E_cond(+)| vs |E_val(-)|")
        print("=" * 82)
        print(f"  {'#':>2s}  {'E_cond(+)':>12s}  {'E_val(-)':>12s}  "
              f"{'|E_c|':>10s}  {'|E_v|':>10s}  "
              f"{'delta':>10s}  {'dGamma':>10s}")
        print("-" * 82)
        n_pair = min(N_COMPARE_MODES, len(pairs))
        deltas = []
        for k in range(n_pair):
            _, _, Ec, Ev = pairs[k]
            Gc = res_cond["Gamma"][pairs[k][0]]
            Gv = res_val["Gamma"][pairs[k][1]]
            delta = abs(Ec) - abs(Ev)
            deltas.append(delta)
            print(f"  {k+1:2d}  {Ec:+12.6f}  {Ev:+12.6f}  "
                  f"{abs(Ec):10.6f}  {abs(Ev):10.6f}  "
                  f"{delta:+10.6f}  {Gc - Gv:+10.6f}")
        print("=" * 82)
        print(f"  Mean |delta| = {np.mean(np.abs(deltas)):.6f}  "
              f"(perfect chiral symmetry -> 0)")

    # -- eigenstate analysis --
    dx_val = L / n
    analyze_eigenstates(res_cond, res_val, dx_val, folder)

    print(f"\nDone.  All outputs in:  {folder}/")


if __name__ == "__main__":
    main()
