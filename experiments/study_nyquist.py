#!/usr/bin/env python3
"""
Study 1.6: Nyquist / save_every study.

Theory: E_Nyq = π/(dt·save_every). LDOS for E < E_Nyq should be
independent of save_every. Above E_Nyq, aliasing artifacts appear.

Method: M=16, save_every=10 (finest). Post-process by subsampling
snapshots to emulate save_every=20, 50, 100.

Output: figures/study_nyquist/
"""

import sys, os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "scripts"))
sys.path.insert(0, _root)

from ldos_utils import (
    integrated_spectrum, measure_c2v_error, ensure_dir,
)
from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import compute_ldos

SAVE_EVERY_VALUES = [10, 20, 50, 100]

n  = 256
L  = 10.0
vF = 1.0
m  = 0.0
dt = 0.01
n_steps = 10000
save_every_finest = 10  # finest cadence

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius = 3.5
cap_strength = 10.0
cap_power = 2
sigma0 = 0.6
k_mag = 3.0

# M=16
N_ang = 4
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)
positions = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (-0.3, 0.0)]

# Energy grid covering above and below Nyquist for save_every=100
E_values = np.linspace(0.5, 8.0, 400)

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_nyquist"))


def main():
    M = N_ang * len(positions)
    print(f"Study 1.6: Nyquist, M={M}")

    xp = get_backend("auto")
    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius, cap_strength, cap_power, xp)
    chi_R, _ = build_analysis_mask(POTENTIAL_OPTION, X_np, Y_np, pot_params,
                                    cap_radius, 0.95, 1.30, xp)
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)
    sqrt_chi = xp.sqrt(chi_R)
    chi_cpu = xp.to_numpy(chi_R).real.astype(np.float64)

    n_E = len(E_values)
    ldos_sums = {se: np.zeros((n_E, n, n)) for se in SAVE_EVERY_VALUES}

    t0 = time.time()
    j = 0
    for x0, y0 in positions:
        for angle in angles:
            kx0 = k_mag * np.cos(angle)
            ky0 = k_mag * np.sin(angle)
            psi0_up, psi0_dn = gaussian_wavepacket(
                X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
                sigma=sigma0, theta=None, band="conduction",
                dx=dx, xp=xp,
            )

            # Propagate with finest save_every
            result = propagate(
                psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
                n_steps=n_steps, dt=dt,
                save_every=save_every_finest, corr_every=save_every_finest,
                max_record=n_steps // save_every_finest + 2,
                measure_transmission=False,
            )
            snaps_np = result["snapshots"]
            times_np = result["snap_times"]
            j += 1

            # For each save_every, subsample
            for se in SAVE_EVERY_VALUES:
                factor = se // save_every_finest
                snaps_sub = snaps_np[::factor]
                times_sub = times_np[::factor]
                ldos_one = compute_ldos(snaps_sub, times_sub, E_values, chi_cpu)
                ldos_sums[se] += ldos_one

            elapsed = time.time() - t0
            print(f"  [{j}/{M}] {elapsed:.0f}s", flush=True)

    total_time = time.time() - t0
    print(f"Total: {total_time:.0f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════════

    spectra = {}
    results = {}
    for se in SAVE_EVERY_VALUES:
        ldos_avg = ldos_sums[se] / M
        spec = integrated_spectrum(ldos_avg, chi_cpu)
        spectra[se] = spec
        E_nyq = np.pi / (dt * se)
        results[se] = dict(E_nyq=float(E_nyq))

    # Compute differences from finest (save_every=10)
    ref_spec = spectra[10]
    for se in SAVE_EVERY_VALUES:
        E_nyq = results[se]["E_nyq"]
        # Below Nyquist: should match
        mask_below = E_values < E_nyq * 0.9
        if mask_below.any():
            diff_below = np.sqrt(np.mean(
                (spectra[se][mask_below] - ref_spec[mask_below])**2
            )) / ref_spec[mask_below].max()
        else:
            diff_below = float("nan")
        results[se]["diff_below_nyquist"] = float(diff_below)
        print(f"  save_every={se}: E_Nyq={E_nyq:.1f}, "
              f"diff_below_Nyq={diff_below:.6f}")

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top: spectra
    ax = axes[0]
    for se in SAVE_EVERY_VALUES:
        spec = spectra[se]
        ax.plot(E_values, spec / spec.max(), label=f"save_every={se}", alpha=0.8)
        E_nyq = results[se]["E_nyq"]
        ax.axvline(E_nyq, color="gray", ls=":", alpha=0.5)
    ax.set_ylabel("LDOS (normalized)")
    ax.set_title("Nyquist study: LDOS spectrum vs snapshot cadence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: difference from finest
    ax = axes[1]
    for se in SAVE_EVERY_VALUES[1:]:
        diff = np.abs(spectra[se] - ref_spec) / max(ref_spec.max(), 1e-30)
        ax.semilogy(E_values, diff, label=f"save_every={se}", alpha=0.8)
        E_nyq = results[se]["E_nyq"]
        ax.axvline(E_nyq, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("E")
    ax.set_ylabel("|Δ LDOS| / max")
    ax.set_title("Difference from save_every=10 baseline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "nyquist_study.png"), dpi=150)
    plt.close(fig)

    # Metrics
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\n{'='*60}")
    print("Study 1.6 Summary:")
    for se in SAVE_EVERY_VALUES:
        r = results[se]
        print(f"  save_every={se}: E_Nyq={r['E_nyq']:.1f}, "
              f"diff_below={r['diff_below_nyquist']:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
