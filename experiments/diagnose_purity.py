#!/usr/bin/env python3
"""
Diagnose mode purity as a function of billiard parameters.
Tests whether time-filtered modes have C₂v density symmetry
(expected for exact eigenstates, violated by multi-doublet mixing).
"""
import sys, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import extract_resonances, time_filter_mode

# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

# (name, a, b, L, n, cap_radius, k_harm, kx0, sigma0, n_steps, dt)
configs = [
    ("small",  2.0, 1.2,  10.0, 512,  3.5,  3.0, -1.0, 0.6, 10000, 0.01),
    ("medium", 5.0, 3.0,  20.0, 512,  8.0,  3.0, -1.0, 1.5, 20000, 0.01),
    ("large", 10.0, 6.0,  40.0, 512, 16.0,  3.0, -1.0, 3.0, 30000, 0.01),
]

# Fixed parameters
vF = 1.0
m  = 0.0
cap_strength = 10.0
cap_power    = 2
decim        = 10
save_every   = 10
max_record   = 2500

mask_safety_factor = 0.95
mask_margin        = 1.30

Gamma_cut       = 8.0
pencil_fraction = 4
L_pencil_max    = 400
n_modes         = 3

POTENTIAL_OPTION = "harmonic_ellipse"


# ═══════════════════════════════════════════════
# Symmetry measurements
# ═══════════════════════════════════════════════

def fft_flip(arr, axis):
    """FFT-consistent reflection: index j -> (n-j) % n along given axis."""
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)


def measure_purity(phi, dx):
    """
    Measure C₂v symmetry of density.

    Returns (x_err, y_err, eo_diff):
        x_err   - relative L1 error under x-reflection of density
        y_err   - relative L1 error under y-reflection of density
        eo_diff - relative L1 difference between Mx-even and Mx-odd densities
    """
    rho = np.abs(phi[0])**2 + np.abs(phi[1])**2

    # x-symmetry: rho(-x, y) = rho(x, y)
    rho_xflip = fft_flip(rho, axis=1)
    x_err = np.sum(np.abs(rho_xflip - rho)) / np.sum(rho)

    # y-symmetry: rho(x, -y) = rho(x, y)
    rho_yflip = fft_flip(rho, axis=0)
    y_err = np.sum(np.abs(rho_yflip - rho)) / np.sum(rho)

    # Mx even/odd density difference
    def apply_Mx(p):
        return np.stack([-1j * fft_flip(p[1], 1),
                          1j * fft_flip(p[0], 1)], axis=0)

    Mx_phi = apply_Mx(phi)
    phi_even = phi + Mx_phi
    phi_odd  = phi - Mx_phi
    ne = np.sqrt(np.sum((np.abs(phi_even[0])**2 + np.abs(phi_even[1])**2)) * dx**2)
    no = np.sqrt(np.sum((np.abs(phi_odd[0])**2  + np.abs(phi_odd[1])**2))  * dx**2)

    if ne > 1e-10 and no > 1e-10:
        phi_even = phi_even / ne
        phi_odd  = phi_odd  / no
        rho_e = np.abs(phi_even[0])**2 + np.abs(phi_even[1])**2
        rho_o = np.abs(phi_odd[0])**2  + np.abs(phi_odd[1])**2
        eo_diff = np.sum(np.abs(rho_e - rho_o)) / np.sum(rho_e)
    else:
        eo_diff = 0.0  # already an Mx eigenstate

    return x_err, y_err, eo_diff


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Collect results across configs
    all_results = []

    for ic, (name, a, b, L, n, cap_radius, k_harm, kx0, sigma0, n_steps, dt) in enumerate(configs):
        t0_config = time.perf_counter()
        print(f"\n{'='*70}")
        print(f"  Config {ic+1}/{len(configs)}: {name}  (a={a}, b={b}, L={L}, n={n})")
        print(f"{'='*70}")

        pot_params = dict(k_harm=k_harm, a=a, b=b)

        # ── grid & operators ──
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

        # ── wavepacket ──
        psi0_up, psi0_dn = gaussian_wavepacket(
            X, Y, x0=0.0, y0=0.0, kx0=kx0, ky0=0.0,
            sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp,
        )

        # ── propagate ──
        print(f"Propagating {n_steps} steps (T={n_steps*dt:.0f}) ...")
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
        P_inside   = result.get("P_inside", np.array([]))
        print(f"Recorded {len(C_R)} corr samples, {len(snapshots)} snapshots")

        # ── C_R decay ratio ──
        CR_ratio = abs(C_R[-1]) / abs(C_R[0]) if len(C_R) > 1 and abs(C_R[0]) > 0 else float("nan")

        # ── extract resonances ──
        print("Harmonic inversion ...")
        spec = extract_resonances(C_R, corr_times, dt * decim,
                                  Gamma_cut=Gamma_cut,
                                  pencil_fraction=pencil_fraction,
                                  L_pencil_max=L_pencil_max)
        E_arr, Gamma_arr, omega_arr, a_arr = spec["E"], spec["Gamma"], spec["omega"], spec["a"]
        nm = min(n_modes, len(omega_arr))

        if nm == 0:
            print(f"  WARNING: no modes extracted for {name}!")
            all_results.append({
                "name": name, "a": a, "b": b, "nm": 0,
                "CR_ratio": CR_ratio, "modes": [],
            })
            continue

        for j in range(nm):
            print(f"  mode {j+1}:  E={E_arr[j]:.6f}  Gamma={Gamma_arr[j]:.6f}")

        # ── filter modes ──
        print(f"Filtering {nm} modes ...")
        modes = [time_filter_mode(snapshots, snap_times, omega_arr[j], chi_cpu)
                 for j in range(nm)]

        # ── measure purity ──
        mode_results = []
        for j in range(nm):
            x_err, y_err, eo_diff = measure_purity(modes[j], dx_val)
            mode_results.append({
                "E": float(E_arr[j]),
                "Gamma": float(Gamma_arr[j]),
                "x_err": x_err,
                "y_err": y_err,
                "eo_diff": eo_diff,
                "phi": modes[j],
            })

        elapsed = time.perf_counter() - t0_config
        print(f"Config {name} done in {elapsed:.1f} s")

        all_results.append({
            "name": name, "a": a, "b": b, "nm": nm,
            "L": L, "dx_val": dx_val,
            "CR_ratio": CR_ratio, "modes": mode_results,
        })

    # ═══════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  PURITY DIAGNOSTIC SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<8} {'a':>5} {'b':>5} {'|C_R end/start|':>16} "
          f"{'Mode':>5} {'E':>10} {'Gamma':>10} {'x_err':>8} {'y_err':>8} {'eo_diff':>8}")
    print("-" * 90)
    for res in all_results:
        if res["nm"] == 0:
            print(f"{res['name']:<8} {res['a']:5.1f} {res['b']:5.1f} "
                  f"{res['CR_ratio']:16.6f}   (no modes extracted)")
            continue
        for j, mr in enumerate(res["modes"]):
            tag = res["name"] if j == 0 else ""
            cr  = f"{res['CR_ratio']:.6f}" if j == 0 else ""
            a_s = f"{res['a']:.1f}" if j == 0 else ""
            b_s = f"{res['b']:.1f}" if j == 0 else ""
            print(f"{tag:<8} {a_s:>5} {b_s:>5} {cr:>16} "
                  f"{j+1:>5} {mr['E']:10.4f} {mr['Gamma']:10.4f} "
                  f"{mr['x_err']:8.4f} {mr['y_err']:8.4f} {mr['eo_diff']:8.4f}")
    print(f"{'='*90}")

    # ═══════════════════════════════════════════════
    # Figure
    # ═══════════════════════════════════════════════
    n_configs = len(all_results)
    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 8))
    if n_configs == 1:
        axes = axes[:, np.newaxis]

    # Top row: density of mode 1 for each billiard size
    for ic, res in enumerate(all_results):
        ax = axes[0, ic]
        if res["nm"] == 0:
            ax.text(0.5, 0.5, "no modes", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"{res['name']} (a={res['a']}, b={res['b']})")
            continue

        phi = res["modes"][0]["phi"]
        rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
        half_L = res["L"] / 2
        extent = [-half_L, half_L, -half_L, half_L]
        im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                       vmin=0, vmax=rho.max())
        ax.set_title(f"{res['name']}: mode 1\n"
                     f"E={res['modes'][0]['E']:.3f}, "
                     f"$\\Gamma$={res['modes'][0]['Gamma']:.3f}",
                     fontsize=10)
        ax.set_aspect("equal")

    # Bottom row: bar chart of x_err, y_err, eo_diff across sizes
    for ic, res in enumerate(all_results):
        ax = axes[1, ic]
        if res["nm"] == 0:
            ax.text(0.5, 0.5, "no modes", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"{res['name']}")
            continue

        nm = res["nm"]
        x_labels = [f"mode {j+1}" for j in range(nm)]
        x_pos = np.arange(nm)
        width = 0.25

        x_errs  = [mr["x_err"]  for mr in res["modes"]]
        y_errs  = [mr["y_err"]  for mr in res["modes"]]
        eo_diffs = [mr["eo_diff"] for mr in res["modes"]]

        ax.bar(x_pos - width, x_errs,  width, label="x_err",  color="C0")
        ax.bar(x_pos,         y_errs,  width, label="y_err",  color="C1")
        ax.bar(x_pos + width, eo_diffs, width, label="eo_diff", color="C2")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("relative error")
        ax.set_title(f"{res['name']} (a={res['a']}, b={res['b']})", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Mode purity: C$_{2v}$ density symmetry vs billiard size", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(fig_dir, "fig_purity_scan.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()
