#!/usr/bin/env python3
"""Generate DMD mode gallery at rank=10,20,30,40 for m=0, with C2v symmetrization."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (make_grid, gaussian_wavepacket, build_potential,
    radial_CAP, build_analysis_mask, build_split_operator, propagate)
from spectral import extract_resonances, time_filter_mode
from scripts.dmd_extract import exact_dmd, spinor_overlap, c2v_err

n=256; L=10.0; vF=1.0; dt=0.01; n_steps=15000; save_every=10
pot_params=dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
sigma0=0.6; k_mag=3.0; Gamma_cut=8.0
dx2=(L/n)**2; dt_snap=dt*save_every; OUT=os.path.join(_root, "figures", "dmd")

def fft_flip(a, ax): return np.roll(np.flip(a, axis=ax), 1, axis=ax)
def sym_c2v(r): return 0.25*(r+fft_flip(r,1)+fft_flip(r,0)+fft_flip(fft_flip(r,1),0))

def main():
    xp = get_backend("auto")
    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    print("Propagating m=0 ...")
    kx0 = k_mag/np.sqrt(2); ky0 = kx0
    V = build_potential("harmonic_ellipse", X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius, cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask("harmonic_ellipse", X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    chi_cpu = xp.to_numpy(chi_R); sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, 0.0, V, W, KX, KY, K, xp)
    p0u, p0d = gaussian_wavepacket(X, Y, x0=0, y0=0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
    result = propagate(p0u, p0d, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
        max_record=n_steps//save_every+2, measure_transmission=False)
    snaps = result["snapshots"]; times = result["snap_times"]

    spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
        Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)
    good = (spec["E"]>2) & (spec["E"]<6) & (spec["Gamma"]<2)
    E_mp = spec["E"][good][:8]; G_mp = spec["Gamma"][good][:8]; omega_mp = spec["omega"][good][:8]
    n_mp = len(E_mp)
    tf_modes = [time_filter_mode(snaps, times, omega_mp[j], chi_cpu) for j in range(n_mp)]

    extent = [-L/2, L/2, -L/2, L/2]

    for rank_test in [10, 20, 30, 40]:
        dmd = exact_dmd(snaps, times, chi_cpu, rank=rank_test, n_eff=150)
        if len(dmd["modes"]) == 0:
            continue
        n_show = min(6, len(dmd["modes"]))

        fig, axes = plt.subplots(3, n_show, figsize=(4.5*n_show, 13))
        if n_show == 1:
            axes = axes[:, np.newaxis]

        for j in range(n_show):
            phi = dmd["modes"][j]
            rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
            rho_sym = sym_c2v(rho)
            xe, ye = c2v_err(rho)

            ax = axes[0, j]
            ax.imshow(rho, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=rho.max())
            ax.set_title(f"DMD raw\nE={dmd['E'][j]:.3f} Γ={dmd['Gamma'][j]:.3f}\nx={xe:.3f} y={ye:.3f}", fontsize=8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

            ax = axes[1, j]
            ax.imshow(rho_sym, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=rho_sym.max())
            ax.set_title("DMD C₂v sym", fontsize=8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

            dists = np.abs(E_mp - dmd["E"][j])
            if len(dists) > 0:
                i_tf = np.argmin(dists)
                rho_tf = np.abs(tf_modes[i_tf][0])**2 + np.abs(tf_modes[i_tf][1])**2
                ov = spinor_overlap(phi, tf_modes[i_tf], chi_cpu, dx2)
                ax = axes[2, j]
                ax.imshow(sym_c2v(rho_tf), extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=rho_tf.max())
                ax.set_title(f"TF ref E={E_mp[i_tf]:.3f}\noverlap={ov:.4f}", fontsize=8)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

        axes[0,0].set_ylabel("DMD raw", fontsize=10)
        axes[1,0].set_ylabel("DMD sym", fontsize=10)
        axes[2,0].set_ylabel("Time filter", fontsize=10)
        fig.suptitle(f"DMD modes (m=0, rank={rank_test}, n_eff=150)", fontsize=13)
        plt.tight_layout()
        out = os.path.join(OUT, f"modes_rank{rank_test}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  rank={rank_test}: {out}")

    print("Done")

if __name__ == "__main__":
    main()
