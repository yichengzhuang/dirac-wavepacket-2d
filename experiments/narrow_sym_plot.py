#!/usr/bin/env python3
"""
Re-extract the top-8 narrowest (by Γ/ΔE) modes from m=0 propagations,
plot raw density vs C₂v-symmetrized side by side.
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

n = 256; L = 10.0; vF = 1.0
dt = 0.01; n_steps = 15000; save_every = 10
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5; cap_strength = 10.0; cap_power = 2
sigma0 = 0.6; k_mag = 3.0
Gamma_threshold = 0.25


def fft_flip(arr, axis):
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)

def sym_c2v(rho):
    rx  = fft_flip(rho, axis=1)
    ry  = fft_flip(rho, axis=0)
    rxy = fft_flip(rx,  axis=0)
    return 0.25 * (rho + rx + ry + rxy)

def c2v_err(rho):
    s = np.sum(rho) + 1e-30
    return (np.sum(np.abs(fft_flip(rho, 1) - rho)) / s,
            np.sum(np.abs(fft_flip(rho, 0) - rho)) / s)


def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    chi_R, _ = build_analysis_mask("harmonic_ellipse", X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)

    # 4 propagation directions
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    all_narrow = []   # (E, Gamma, omega, prop_idx)
    prop_cache = {}   # prop_idx -> {snapshots, snap_times}

    for ia, angle in enumerate(angles):
        kx0 = k_mag * np.cos(angle)
        ky0 = k_mag * np.sin(angle)
        print(f"Propagation {ia+1}/4: k=({kx0:+.2f},{ky0:+.2f})")

        V = build_potential("harmonic_ellipse", X_np, Y_np, pot_params, xp)
        W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                       cap_strength=cap_strength, cap_power=cap_power, xp=xp)
        sqrt_chi = xp.sqrt(chi_R)
        halfV, kin = build_split_operator(vF, dt, 0.0, V, W, KX, KY, K, xp)
        psi0_up, psi0_dn = gaussian_wavepacket(
            X, Y, x0=0.0, y0=0.0, kx0=kx0, ky0=ky0,
            sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
        result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
            n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
            max_record=n_steps//save_every+2, measure_transmission=False)

        spec = extract_resonances(result["C_R"], result["corr_times"],
            dt*save_every, Gamma_cut=8.0, pencil_fraction=4, L_pencil_max=400)

        prop_cache[ia] = dict(snapshots=result["snapshots"],
                              snap_times=result["snap_times"])

        for j in range(len(spec["E"])):
            if spec["E"][j] > 0 and spec["Gamma"][j] < Gamma_threshold:
                all_narrow.append((spec["E"][j], spec["Gamma"][j],
                                   spec["omega"][j], ia))

        n_nar = sum(1 for e, g, _, _ in all_narrow if _ == ia)
        print(f"  {len(spec['E'])} poles, {n_nar} narrow")

        # free heavy arrays early
        del result, psi0_up, psi0_dn

    # deduplicate by energy (within ΔE < 0.1)
    all_narrow.sort(key=lambda x: x[0])
    deduped = []
    used = set()
    for i in range(len(all_narrow)):
        if i in used:
            continue
        cluster = [all_narrow[i]]
        used.add(i)
        for j in range(i+1, len(all_narrow)):
            if j in used:
                continue
            if abs(all_narrow[j][0] - all_narrow[i][0]) < 0.1:
                cluster.append(all_narrow[j])
                used.add(j)
            else:
                break
        deduped.append(min(cluster, key=lambda x: x[1]))  # smallest Γ

    print(f"\n{len(all_narrow)} raw → {len(deduped)} unique narrow modes")

    # filter to E < 8 (the physically interesting range)
    deduped = [d for d in deduped if d[0] < 8.0]
    print(f"  After E < 8 filter: {len(deduped)} modes")

    # compute Γ/ΔE for each
    for i in range(len(deduped)):
        dEs = [abs(deduped[i][0] - deduped[j][0])
               for j in range(len(deduped)) if j != i]
        dE_min = min(dEs) if dEs else 1.0
        deduped[i] = (*deduped[i], deduped[i][1] / dE_min if dE_min > 0 else 99)

    # sort by Γ/ΔE, pick top 8
    deduped.sort(key=lambda x: x[4])
    selected = deduped[:8]

    print("Selected 8 modes (sorted by Γ/ΔE):")
    print(f"  {'E':>8} {'Γ':>8} {'Γ/ΔE':>8} {'prop':>5}")
    for E, G, omega, ia, ratio in selected:
        print(f"  {E:8.4f} {G:8.4f} {ratio:8.3f} {ia+1:5d}")

    # time-filter and plot
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    all_rho_raw = []
    all_rho_sym = []

    for k, (E, G, omega, ia, ratio) in enumerate(selected):
        pd = prop_cache[ia]
        phi = time_filter_mode(pd["snapshots"], pd["snap_times"], omega, chi_cpu)
        rho_raw = np.abs(phi[0])**2 + np.abs(phi[1])**2
        rho_sym = sym_c2v(rho_raw)
        all_rho_raw.append(rho_raw)
        all_rho_sym.append(rho_sym)

        row = k // 4
        col = k % 4

        # raw (top half)
        ax = axes[row * 2, col]
        im = ax.imshow(rho_raw, extent=extent, origin="lower",
                       cmap="inferno", vmin=0, vmax=rho_raw.max())
        xe, ye = c2v_err(rho_raw)
        ax.set_title(f"E={E:.3f}  $\\Gamma$={G:.3f}\n"
                     f"$\\Gamma/\\Delta E$={ratio:.2f}\n"
                     f"x={xe:.3f} y={ye:.3f}",
                     fontsize=8)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.75)

        # symmetrized (bottom half)
        ax = axes[row * 2 + 1, col]
        im = ax.imshow(rho_sym, extent=extent, origin="lower",
                       cmap="inferno", vmin=0, vmax=rho_sym.max())
        ax.set_title("$C_{2v}$ sym", fontsize=8)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle("Narrow modes ($\\Gamma<0.25$, $E<8$, sorted by $\\Gamma/\\Delta E$):\n"
                 "odd rows = raw, even rows = $C_{2v}$ symmetrized",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "task2_narrow_modes_sym.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")

    # save densities for future use (no re-propagation needed)
    np.savez(os.path.join(OUT_DIR, "narrow_modes_density.npz"),
             rho_raw=all_rho_raw, rho_sym=all_rho_sym,
             E=np.array([s[0] for s in selected]),
             Gamma=np.array([s[1] for s in selected]),
             ratio=np.array([s[4] for s in selected]))
    print(f"Saved densities → {OUT_DIR}/narrow_modes_density.npz")
    print(f"Total: {time.perf_counter()-t0:.1f} s")


if __name__ == "__main__":
    main()
