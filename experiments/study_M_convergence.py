#!/usr/bin/env python3
"""
Study 1.1: M-convergence of multi-propagation LDOS averaging.

Theory prediction (central limit theorem):
    ε_C2v ~ C / √M

Method:
    Run M=128 propagations (16 angles × 8 positions).
    At checkpoints M = 1,2,4,8,16,24,32,48,64,96,128, measure:
      - C₂v error (x_err, y_err) at the 3 strongest peaks
      - Spectral correlation with M=128 reference
      - Spatial L2 distance from M=128 reference

Output: figures/study_M/
    m_convergence.png       — log-log ε_C2v vs M with M^{-1/2} line
    spectrum_convergence.png — spectra at M=4,16,48,128
    spatial_convergence.png  — 2D maps at strongest peak for M=4,16,48,128
    single_propagation_hist.png — histogram of single-propagation C₂v errors
    metrics.json             — all quantitative results
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
    run_ldos_sweep, fft_flip, symmetrize_c2v, measure_c2v_error,
    integrated_spectrum, find_peaks_simple, ensure_dir,
)

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

n  = 256
L  = 10.0
vF = 1.0
m  = 0.0

dt         = 0.01
n_steps    = 10000   # T_max = 100
save_every = 50

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

sigma0 = 0.6
k_mag  = 3.0

# 16 angles × 8 positions = 128 propagations
N_ang  = 16
angles = np.linspace(0, 2 * np.pi, N_ang, endpoint=False)

# 8 positions: center + ring(0.3, 3pts) + ring(0.6, 4pts)
positions = [(0.0, 0.0)]
for r, npts in [(0.3, 3), (0.6, 4)]:
    for a in np.linspace(0, 2*np.pi, npts, endpoint=False):
        positions.append((r * np.cos(a), r * np.sin(a)))
# total: 1 + 3 + 4 = 8 positions

E_values = np.linspace(0.5, 5.5, 300)

# Checkpoints
CHECKPOINTS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_M"))

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    M_total = N_ang * len(positions)
    assert M_total == 128, f"Expected M=128, got {M_total}"
    print(f"Study 1.1: M-convergence, {M_total} propagations")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Output: {OUT_DIR}")
    print()

    n_E = len(E_values)

    # Storage for per-propagation LDOS (to compute running averages)
    # We store cumulative sum and per-propagation C₂v errors
    ldos_sum = np.zeros((n_E, n, n))
    single_c2v_errors = []  # list of (x_err, y_err) for each propagation

    # Checkpoint results
    checkpoint_data = {}  # M -> {x_err, y_err, spectrum, ldos_avg_at_peak}

    # We need the grid/chi for later, get it from the first propagation
    from accel import get_backend
    from accel.dynamics_accel import (
        make_grid, gaussian_wavepacket, build_potential, radial_CAP,
        build_analysis_mask, build_split_operator, propagate,
    )
    from spectral import compute_ldos

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

            result = propagate(
                psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
                n_steps=n_steps, dt=dt,
                save_every=save_every, corr_every=save_every,
                max_record=n_steps // save_every + 2,
                measure_transmission=False,
            )
            snaps_np = result["snapshots"]
            times_np = result["snap_times"]

            ldos_one = compute_ldos(snaps_np, times_np, E_values, chi_cpu)
            ldos_sum += ldos_one
            j += 1

            # Single-propagation C₂v error at the overall strongest peak
            # (we'll identify the peak from the cumulative spectrum)
            spec_one = integrated_spectrum(ldos_one, chi_cpu)
            peak_idx_one = np.argmax(spec_one)
            xe, ye = measure_c2v_error(ldos_one[peak_idx_one])
            single_c2v_errors.append((xe, ye))

            elapsed = time.time() - t0
            print(f"  [{j}/{M_total}] angle={np.degrees(angle):.0f}° "
                  f"pos=({x0:.2f},{y0:.2f})  "
                  f"single_c2v=({xe:.4f},{ye:.4f})  "
                  f"{elapsed:.0f}s", flush=True)

            # Check if this is a checkpoint
            if j in CHECKPOINTS:
                ldos_avg_j = ldos_sum / j
                spec_j = integrated_spectrum(ldos_avg_j, chi_cpu)

                # Find top-3 peaks in the cumulative spectrum
                peak_idx = np.argsort(spec_j)[::-1][:3]
                c2v_at_peaks = []
                for pi in peak_idx:
                    xe, ye = measure_c2v_error(ldos_avg_j[pi])
                    c2v_at_peaks.append((float(xe), float(ye)))

                # Mean C₂v error across top-3 peaks
                mean_x = np.mean([c[0] for c in c2v_at_peaks])
                mean_y = np.mean([c[1] for c in c2v_at_peaks])

                checkpoint_data[j] = dict(
                    c2v_at_peaks=c2v_at_peaks,
                    mean_x_err=float(mean_x),
                    mean_y_err=float(mean_y),
                    peak_energies=[float(E_values[pi]) for pi in peak_idx],
                    spectrum=spec_j.tolist(),
                )

                print(f"    ★ Checkpoint M={j}: mean C₂v = "
                      f"({mean_x:.5f}, {mean_y:.5f})")

                # Save spatial map at strongest peak for visualization
                np.save(os.path.join(OUT_DIR, f"ldos_avg_M{j}.npy"),
                        ldos_avg_j[peak_idx[0]])

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── Final reference (M=128) ──
    ldos_ref = ldos_sum / M_total
    spec_ref = integrated_spectrum(ldos_ref, chi_cpu)
    np.save(os.path.join(OUT_DIR, "ldos_ref_M128.npy"), ldos_ref)
    np.save(os.path.join(OUT_DIR, "E_values.npy"), E_values)

    # ══════════════════════════════════════════════════════════════════════
    # Analysis and Figures
    # ══════════════════════════════════════════════════════════════════════

    # 1. M-convergence: log-log plot
    Ms = sorted(checkpoint_data.keys())
    mean_errs = [checkpoint_data[m_]["mean_x_err"] + checkpoint_data[m_]["mean_y_err"]
                 for m_ in Ms]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(Ms, mean_errs, "ko-", lw=2, ms=8, label="Measured ε_C₂v")

    # M^{-1/2} reference line
    M_ref = np.array(Ms, dtype=float)
    C_fit = mean_errs[-1] * np.sqrt(Ms[-1])  # anchor at last point
    ax.loglog(M_ref, C_fit / np.sqrt(M_ref), "r--", lw=1.5,
              label=r"$C / \sqrt{M}$ reference")

    # Fit power law
    log_M = np.log(np.array(Ms[2:], dtype=float))  # skip M=1,2
    log_e = np.log(np.array(mean_errs[2:]))
    slope, intercept = np.polyfit(log_M, log_e, 1)
    ax.set_title(f"M-convergence of C₂v error\n"
                 f"Fitted slope: {slope:.3f} (theory: -0.500)")
    ax.set_xlabel("Number of propagations M")
    ax.set_ylabel("Mean C₂v error (x + y)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "m_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"  → m_convergence.png (slope={slope:.3f})")

    # 2. Spectrum convergence
    fig, ax = plt.subplots(figsize=(8, 4))
    for m_ in [4, 16, 48, 128]:
        if m_ in checkpoint_data:
            spec = np.array(checkpoint_data[m_]["spectrum"])
            spec_norm = spec / spec.max()
            ax.plot(E_values, spec_norm, label=f"M={m_}", alpha=0.8)
    ax.set_xlabel("E")
    ax.set_ylabel("Integrated LDOS (normalized)")
    ax.set_title("Spectral convergence with M")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectrum_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"  → spectrum_convergence.png")

    # 3. Spatial convergence
    display_Ms = [4, 16, 48, 128]
    fig, axes = plt.subplots(1, len(display_Ms), figsize=(4*len(display_Ms), 4))
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]
    for i, m_ in enumerate(display_Ms):
        fpath = os.path.join(OUT_DIR, f"ldos_avg_M{m_}.npy")
        if os.path.exists(fpath):
            rho = np.load(fpath)
        else:
            rho = ldos_ref[np.argmax(spec_ref)]
        ax = axes[i]
        im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno")
        xe, ye = measure_c2v_error(rho)
        ax.set_title(f"M={m_}\nε=({xe:.4f},{ye:.4f})", fontsize=10)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Spatial LDOS at strongest peak vs M", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spatial_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"  → spatial_convergence.png")

    # 4. Single-propagation C₂v error histogram
    single_errs = np.array(single_c2v_errors)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(single_errs[:, 0], bins=20, alpha=0.7, label="x_err")
    axes[0].hist(single_errs[:, 1], bins=20, alpha=0.7, label="y_err")
    axes[0].set_xlabel("C₂v error (single propagation)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].set_title("Distribution of single-propagation asymmetries")

    # Verify: σ_single / √M should predict M-averaged error
    sigma_x = np.std(single_errs[:, 0])
    sigma_y = np.std(single_errs[:, 1])
    predicted = (sigma_x + sigma_y) / np.sqrt(M_ref)
    axes[1].loglog(Ms, mean_errs, "ko-", lw=2, ms=8, label="Measured")
    axes[1].loglog(M_ref, predicted, "b--", lw=1.5,
                   label=f"σ/√M (σ_x={sigma_x:.3f}, σ_y={sigma_y:.3f})")
    axes[1].set_xlabel("M")
    axes[1].set_ylabel("Mean C₂v error")
    axes[1].legend()
    axes[1].set_title("CLT prediction vs measurement")
    axes[1].grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "single_propagation_hist.png"), dpi=150)
    plt.close(fig)
    print(f"  → single_propagation_hist.png")

    # 5. Save metrics
    metrics = dict(
        M_total=M_total,
        total_time_s=total_time,
        fitted_slope=float(slope),
        theory_slope=-0.5,
        sigma_x_single=float(sigma_x),
        sigma_y_single=float(sigma_y),
        checkpoints={str(k): v for k, v in checkpoint_data.items()},
    )
    # Remove full spectrum from JSON (too large)
    for k in metrics["checkpoints"]:
        del metrics["checkpoints"][k]["spectrum"]

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → metrics.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"Study 1.1 Summary:")
    print(f"  Fitted slope: {slope:.3f} (theory: -0.500)")
    print(f"  σ_x(single): {sigma_x:.4f}, σ_y(single): {sigma_y:.4f}")
    print(f"  M=48  C₂v: {checkpoint_data[48]['mean_x_err']:.5f}, "
          f"{checkpoint_data[48]['mean_y_err']:.5f}")
    print(f"  M=128 C₂v: {checkpoint_data[128]['mean_x_err']:.5f}, "
          f"{checkpoint_data[128]['mean_y_err']:.5f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
