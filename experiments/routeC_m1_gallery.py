#!/usr/bin/env python3
"""
Route C mode gallery for m=1 (control case).
Show reconstructed modes with C2v symmetrization, compared to TF reference.
Only narrow modes (Γ < 0.1).
"""
import sys, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (make_grid, gaussian_wavepacket, build_potential,
    radial_CAP, build_analysis_mask, build_split_operator, propagate)
from spectral import extract_resonances, time_filter_mode
from scripts.fixed_pole_recon import (fixed_pole_constrained_dmd,
    fixed_pole_recon_direct, spinor_overlap, c2v_err, fft_flip, _normalize_mask)

OUT = os.path.join(_root, "figures", "fixed_pole")
os.makedirs(OUT, exist_ok=True)

n=256; L=10.0; vF=1.0; dt=0.01; n_steps=15000; save_every=10
pot_params=dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
sigma0=0.6; k_mag=3.0; Gamma_cut=8.0

def sym_c2v(r):
    rx = fft_flip(r, 1); ry = fft_flip(r, 0)
    return 0.25*(r + rx + ry + fft_flip(rx, 0))

def main():
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx2 = (L/n)**2; dt_snap = dt*save_every
    extent = [-L/2, L/2, -L/2, L/2]

    # Propagate m=1
    print("Propagating m=1...")
    V = build_potential("harmonic_ellipse", X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius, cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask("harmonic_ellipse", X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    chi_cpu = xp.to_numpy(chi_R); sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, 1.0, V, W, KX, KY, K, xp)
    psi0_up, psi0_dn = gaussian_wavepacket(X, Y, x0=0, y0=0, kx0=k_mag, ky0=0,
        sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
    result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
        max_record=n_steps//save_every+2, measure_transmission=False)
    snaps = result["snapshots"]; times = result["snap_times"]

    # MP poles
    spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
        Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)
    # Filter: conduction band, narrow
    good = (spec["E"] > 0) & (spec["Gamma"] < 0.1) & (spec["Gamma"] > 0)
    E_mp = spec["E"][good]; G_mp = spec["Gamma"][good]; omega_mp = spec["omega"][good]
    n_mp = min(8, len(E_mp))
    E_mp = E_mp[:n_mp]; G_mp = G_mp[:n_mp]; omega_mp = omega_mp[:n_mp]

    print(f"\n{n_mp} narrow poles (Γ < 0.1):")
    for j in range(n_mp):
        print(f"  E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

    # TF modes
    tf_modes = [time_filter_mode(snaps, times, omega_mp[j], chi_cpu) for j in range(n_mp)]

    # Route C modes
    print("\nRoute C extraction...")
    rc_modes = fixed_pole_constrained_dmd(snaps, times, omega_mp, chi_cpu,
        n_modes=n_mp, rank=20)

    # Route A modes (for comparison)
    ra_modes = fixed_pole_recon_direct(snaps, times, omega_mp, chi_cpu,
        n_modes=n_mp, lambda_tik=1e-3)

    # ── Gallery: 4 rows × n_mp cols ──
    # Row 0: Route C raw
    # Row 1: Route C C2v symmetrized
    # Row 2: Time filter C2v symmetrized
    # Row 3: Route A C2v symmetrized

    fig, axes = plt.subplots(4, n_mp, figsize=(3.8*n_mp, 15))
    if n_mp == 1:
        axes = axes[:, np.newaxis]

    for j in range(n_mp):
        # Route C
        rho_rc = np.abs(rc_modes[j][0])**2 + np.abs(rc_modes[j][1])**2
        rho_rc_sym = sym_c2v(rho_rc)
        xe_rc, ye_rc = c2v_err(rho_rc)
        ov_rc = spinor_overlap(rc_modes[j], tf_modes[j], chi_cpu, dx2)

        # TF
        rho_tf = np.abs(tf_modes[j][0])**2 + np.abs(tf_modes[j][1])**2
        rho_tf_sym = sym_c2v(rho_tf)
        xe_tf, ye_tf = c2v_err(rho_tf)

        # Route A
        rho_ra = np.abs(ra_modes[j][0])**2 + np.abs(ra_modes[j][1])**2
        rho_ra_sym = sym_c2v(rho_ra)
        xe_ra, ye_ra = c2v_err(rho_ra)

        # Row 0: Route C raw
        ax = axes[0, j]
        ax.imshow(rho_rc, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=rho_rc.max())
        ax.set_title(f"E={E_mp[j]:.3f}  Γ={G_mp[j]:.4f}\nC₂v: {xe_rc:.3f}/{ye_rc:.3f}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

        # Row 1: Route C sym
        ax = axes[1, j]
        ax.imshow(rho_rc_sym, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=rho_rc_sym.max())
        ax.set_title(f"ov={ov_rc:.4f}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

        # Row 2: TF sym
        ax = axes[2, j]
        ax.imshow(rho_tf_sym, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=rho_tf_sym.max())
        ax.set_title(f"C₂v: {xe_tf:.3f}/{ye_tf:.3f}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

        # Row 3: Route A sym
        ax = axes[3, j]
        ax.imshow(rho_ra_sym, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=rho_ra_sym.max())
        ax.set_title(f"C₂v: {xe_ra:.3f}/{ye_ra:.3f}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("Route C\n(raw)", fontsize=10)
    axes[1, 0].set_ylabel("Route C\n(C₂v sym)", fontsize=10)
    axes[2, 0].set_ylabel("Time filter\n(C₂v sym)", fontsize=10)
    axes[3, 0].set_ylabel("Route A\n(C₂v sym)", fontsize=10)

    fig.suptitle(f"Mode reconstruction gallery — m=1, {n_mp} narrow modes (Γ < 0.1)\n"
                 f"Route C: avg C₂v = {np.mean([c2v_err(np.abs(m[0])**2+np.abs(m[1])**2)[0] for m in rc_modes]):.3f}  |  "
                 f"TF: avg C₂v = {np.mean([c2v_err(np.abs(m[0])**2+np.abs(m[1])**2)[0] for m in tf_modes]):.3f}",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "gallery_m1_narrow.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure → {OUT}/gallery_m1_narrow.png")

    # Print summary
    print(f"\nSummary:")
    print(f"{'#':>3} {'E':>8} {'Γ':>8} {'RC_C2v_x':>10} {'TF_C2v_x':>10} {'RA_C2v_x':>10} {'overlap':>8}")
    for j in range(n_mp):
        xe_rc, _ = c2v_err(np.abs(rc_modes[j][0])**2 + np.abs(rc_modes[j][1])**2)
        xe_tf, _ = c2v_err(np.abs(tf_modes[j][0])**2 + np.abs(tf_modes[j][1])**2)
        xe_ra, _ = c2v_err(np.abs(ra_modes[j][0])**2 + np.abs(ra_modes[j][1])**2)
        ov = spinor_overlap(rc_modes[j], tf_modes[j], chi_cpu, dx2)
        print(f"{j+1:3d} {E_mp[j]:8.4f} {G_mp[j]:8.4f} {xe_rc:10.4f} {xe_tf:10.4f} {xe_ra:10.4f} {ov:8.4f}")

if __name__ == "__main__":
    main()
