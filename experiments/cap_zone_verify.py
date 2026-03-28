#!/usr/bin/env python3
"""
CAP Step B: Verify wavefunction density is negligible at the CAP boundary.

For the actual harmonic ellipse system, check whether the physical
wavefunction extends into the CAP region. If it does, the CAP is
artificially damping the quasi-bound state — contaminating Γ.

Method:
  - Standard harmonic ellipse propagation
  - At each snapshot, measure max density on the circle r = cap_radius
  - Also measure density along x-axis (worst case: closest to CAP)
  - Compare ρ(r_cap) / ρ_max — should be < 1% for clean separation

Output: figures/cap_validate/
"""

import sys, os, time, json
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

OUT_DIR = os.path.join(_root, "figures", "cap_validate")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

n  = 256
L  = 10.0
vF = 1.0
m  = 0.0
dt = 0.01
n_steps = 10000  # T_max = 100
save_every = 100  # save 101 snapshots

POTENTIAL_OPTION = "harmonic_ellipse"
pot_params = dict(k_harm=3.0, a=2.0, b=1.2)

cap_radius   = 3.5
cap_strength = 10.0
cap_power    = 2

sigma0 = 0.6

# Test different energies via k_mag
KMAG_VALUES = [2.0, 3.0, 4.0, 5.0]


def main():
    print("=" * 60)
    print("CAP Step B: Zone density verification")
    print(f"Harmonic ellipse: a={pot_params['a']}, b={pot_params['b']}, "
          f"k_harm={pot_params['k_harm']}")
    print(f"CAP: radius={cap_radius}, strength={cap_strength}, power={cap_power}")
    print(f"k_mag values: {KMAG_VALUES}")
    print("=" * 60)

    xp = get_backend("auto")
    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx_val = float(L / n)

    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius, cap_strength, cap_power, xp)
    chi_R, _ = build_analysis_mask(POTENTIAL_OPTION, X_np, Y_np, pot_params,
                                    cap_radius, 0.95, 1.30, xp)
    halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)
    sqrt_chi = xp.sqrt(chi_R)

    R_np = np.sqrt(X_np**2 + Y_np**2)

    # Annular mask at r = cap_radius ± dr
    dr = dx_val * 2
    cap_ring = ((R_np > cap_radius - dr) & (R_np < cap_radius + dr)).astype(float)

    # Also prepare angular slices
    # x-axis slice: y ~ 0, x > 0
    mid_y = n // 2
    x_profile_idx = mid_y  # y=0 row

    results = {}

    for k_mag in KMAG_VALUES:
        E_peak = vF * k_mag
        # Turning radius along x
        x_turn = pot_params['a'] * np.sqrt(2 * E_peak / pot_params['k_harm'])
        gap = cap_radius - x_turn

        print(f"\n{'─'*50}")
        print(f"k_mag={k_mag}, E_peak={E_peak:.1f}, x_turn={x_turn:.2f}, "
              f"gap to CAP={gap:.2f}")

        # Launch along x (worst case for x-axis proximity)
        psi0_up, psi0_dn = gaussian_wavepacket(
            X, Y, x0=0.0, y0=0.0, kx0=k_mag, ky0=0.0,
            sigma=sigma0, theta=None, band="conduction",
            dx=dx_val, xp=xp,
        )

        result = propagate(
            psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx_val, xp,
            n_steps=n_steps, dt=dt,
            save_every=save_every, corr_every=save_every,
            max_record=n_steps // save_every + 2,
            measure_transmission=False,
        )

        snaps = result["snapshots"]
        snap_times = result["snap_times"]

        # For each snapshot, measure density at CAP boundary
        rho_at_cap = []
        rho_max_inside = []
        x_profiles = []

        for i, snap in enumerate(snaps):
            rho = np.abs(snap[0])**2 + np.abs(snap[1])**2

            # Max density on the CAP ring
            rho_cap = np.max(rho * cap_ring)

            # Max density inside (r < cap_radius - 0.5)
            inside = R_np < (cap_radius - 0.5)
            rho_in = np.max(rho[inside]) if inside.any() else 0

            rho_at_cap.append(float(rho_cap))
            rho_max_inside.append(float(rho_in))

            # x-axis density profile at y=0
            x_profiles.append(rho[mid_y, :])

        rho_at_cap = np.array(rho_at_cap)
        rho_max_inside = np.array(rho_max_inside)
        x_profiles = np.array(x_profiles)

        # Ratio: density at CAP / max density inside
        ratio = np.zeros_like(rho_at_cap)
        nonzero = rho_max_inside > 0
        ratio[nonzero] = rho_at_cap[nonzero] / rho_max_inside[nonzero]

        # Time-averaged ratio (skip t=0 when packet hasn't spread yet)
        skip = max(1, len(ratio) // 10)
        avg_ratio = float(np.mean(ratio[skip:]))
        max_ratio = float(np.max(ratio[skip:]))

        results[k_mag] = dict(
            E_peak=float(E_peak),
            x_turn=float(x_turn),
            gap=float(gap),
            avg_ratio=avg_ratio,
            max_ratio=max_ratio,
            snap_times=snap_times.tolist() if hasattr(snap_times, 'tolist') else list(snap_times),
            ratio_vs_time=ratio.tolist(),
        )

        status = "✓" if max_ratio < 0.01 else ("⚠" if max_ratio < 0.05 else "✗")
        print(f"  ρ(cap)/ρ(inside): avg={avg_ratio:.4f}, max={max_ratio:.4f} {status}")

        # Save x-axis profile at a late time
        np.save(os.path.join(OUT_DIR, f"x_profile_k{k_mag}.npy"), x_profiles)

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    # 1. Ratio vs time for all k_mag
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for k_mag in KMAG_VALUES:
        r = results[k_mag]
        t = r["snap_times"]
        ratio = r["ratio_vs_time"]
        ax.semilogy(t, ratio, label=f"k={k_mag} (E={r['E_peak']:.0f}, "
                    f"gap={r['gap']:.2f})")
    ax.axhline(0.01, color="r", ls="--", lw=1, label="1% threshold")
    ax.set_xlabel("Time")
    ax.set_ylabel("ρ(r=cap_radius) / ρ_max(inside)")
    ax.set_title("Density leakage into CAP region")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. x-axis density profile at late time
    ax = axes[1]
    x_axis = x_np
    for k_mag in KMAG_VALUES:
        profiles = np.load(os.path.join(OUT_DIR, f"x_profile_k{k_mag}.npy"))
        # Use profile at t ~ T/2 (quasi-steady state)
        idx = len(profiles) // 2
        profile = profiles[idx]
        profile_norm = profile / profile.max() if profile.max() > 0 else profile
        ax.semilogy(x_axis, profile_norm + 1e-15,
                    label=f"k={k_mag}", alpha=0.8)

    ax.axvline(cap_radius, color="r", ls="--", lw=1.5, label=f"CAP start (r={cap_radius})")
    # Add turning points
    for k_mag in KMAG_VALUES:
        x_t = results[k_mag]["x_turn"]
        ax.axvline(x_t, color="gray", ls=":", alpha=0.5)
        ax.text(x_t, 0.5, f"E={results[k_mag]['E_peak']:.0f}",
                fontsize=7, rotation=90, va="center")
    ax.set_xlabel("x (along major axis, y=0)")
    ax.set_ylabel("ρ(x,0) / ρ_max (log scale)")
    ax.set_title("Density profile along x-axis\n(gray: turning points, red: CAP)")
    ax.set_xlim(0, L/2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "zone_density.png"), dpi=150)
    plt.close(fig)
    print(f"\n  → zone_density.png")

    # Save metrics
    metrics = {str(k): {kk: vv for kk, vv in v.items()
                        if kk != "ratio_vs_time" and kk != "snap_times"}
               for k, v in results.items()}
    with open(os.path.join(OUT_DIR, "zone_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → zone_metrics.json")

    # Summary
    print(f"\n{'='*60}")
    print("Summary: density at CAP boundary / max density inside")
    print(f"{'k_mag':>6} {'E_peak':>7} {'x_turn':>7} {'gap':>6} "
          f"{'avg_ratio':>10} {'max_ratio':>10} {'status':>8}")
    for k_mag in KMAG_VALUES:
        r = results[k_mag]
        status = "SAFE" if r["max_ratio"] < 0.01 else \
                 ("WARN" if r["max_ratio"] < 0.05 else "DANGER")
        print(f"{k_mag:6.1f} {r['E_peak']:7.1f} {r['x_turn']:7.2f} "
              f"{r['gap']:6.2f} {r['avg_ratio']:10.4f} "
              f"{r['max_ratio']:10.4f} {status:>8}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
