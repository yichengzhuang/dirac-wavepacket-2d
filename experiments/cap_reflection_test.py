#!/usr/bin/env python3
"""
CAP Step A: Reflection coefficient measurement.

Send a free Dirac wave packet toward the CAP boundary.
Measure what fraction of probability reflects back vs gets absorbed.

If R > 1%, the CAP is not absorbing well enough.

Method:
  - V=0 (free particle), large grid L=20, n=512
  - Launch from inside, aimed outward
  - Track P_inside(t) = ∫_{r < r_detect} |ψ|² dr²
  - After wave packet passes through CAP, residual P_inside = R (reflection)

Scan:
  - Energies: E = 1, 2, 3, 4, 5 (via k_mag)
  - CAP configs: several (W0, p, r_cap) combinations
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
    make_grid, gaussian_wavepacket, radial_CAP,
    build_split_operator, propagate,
)

OUT_DIR = os.path.join(_root, "figures", "cap_validate")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

L  = 20.0      # large grid
n  = 512
vF = 1.0
m  = 0.0
dt = 0.01

sigma0 = 0.5   # wave packet width
x_launch = 3.0 # launch position (inside, aimed outward along +x)

# Energies to test
ENERGIES = [1.0, 2.0, 3.0, 4.0, 5.0]

# CAP configurations to test: (cap_radius, cap_strength, cap_power)
CAP_CONFIGS = [
    (3.5,  10.0, 2, "default"),
    (3.5,  5.0,  2, "W0=5"),
    (3.5,  20.0, 2, "W0=20"),
    (3.5,  10.0, 3, "p=3"),
    (3.5,  10.0, 1, "p=1"),
    (5.0,  10.0, 2, "R=5"),
    (6.0,  10.0, 2, "R=6"),
]

# Detection radius: measure P inside this sphere after wave packet has passed
r_detect = 2.0  # well inside the launch point


def compute_density(psi_up, psi_dn, xp):
    """Total probability density |ψ_A|² + |ψ_B|²."""
    up = xp.to_numpy(psi_up)
    dn = xp.to_numpy(psi_dn)
    return np.abs(up)**2 + np.abs(dn)**2


def main():
    print("=" * 60)
    print("CAP Step A: Reflection coefficient measurement")
    print(f"Grid: L={L}, n={n}, dx={L/n:.4f}")
    print(f"Launch: x={x_launch}, σ={sigma0}")
    print(f"Detection radius: {r_detect}")
    print(f"Energies: {ENERGIES}")
    print(f"CAP configs: {len(CAP_CONFIGS)}")
    print("=" * 60)

    xp = get_backend("auto")

    # Grid
    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx_val = float(L / n)

    # Detection mask (r < r_detect)
    R_np = np.sqrt(X_np**2 + Y_np**2)
    detect_mask = (R_np < r_detect).astype(np.float64)

    # V = 0 everywhere (free particle)
    V = xp.array(np.zeros_like(X_np))

    results = {}

    for cap_radius, cap_strength, cap_power, label in CAP_CONFIGS:
        print(f"\n{'─'*50}")
        print(f"CAP config: {label} (R={cap_radius}, W0={cap_strength}, p={cap_power})")

        W = radial_CAP(X_np, Y_np, cap_radius, cap_strength, cap_power, xp)
        halfV, kin = build_split_operator(vF, dt, m, V, W, KX, KY, K, xp)

        # Dummy chi for propagate (not used for LDOS here)
        chi_R = xp.array(np.ones_like(X_np))
        sqrt_chi = xp.sqrt(chi_R)

        config_results = {}

        for E in ENERGIES:
            k_mag = E / vF
            # Propagation time: wave packet needs to travel from x_launch
            # to CAP (at cap_radius), get absorbed, and any reflection travel back.
            # Distance ~ 2*(cap_radius - x_launch + 3*sigma) / vF
            travel_dist = 2 * (cap_radius - x_launch + 5*sigma0)
            T_needed = travel_dist / vF + 10  # extra time for slow components
            n_steps = int(T_needed / dt)
            n_steps = min(n_steps, 10000)  # cap at 10000

            # Launch along +x from (x_launch, 0)
            psi0_up, psi0_dn = gaussian_wavepacket(
                X, Y, x0=x_launch, y0=0.0, kx0=k_mag, ky0=0.0,
                sigma=sigma0, theta=None, band="conduction",
                dx=dx_val, xp=xp,
            )

            # Initial probability inside detection region
            rho0 = compute_density(psi0_up, psi0_dn, xp)
            P0_inside = np.sum(rho0 * detect_mask) * dx_val**2
            P0_total  = np.sum(rho0) * dx_val**2

            # Propagate
            t0 = time.time()
            result = propagate(
                psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx_val, xp,
                n_steps=n_steps, dt=dt,
                save_every=n_steps,  # only save final state
                corr_every=n_steps,
                max_record=5,
                measure_transmission=False,
            )
            elapsed = time.time() - t0

            # Final state
            snaps = result["snapshots"]
            if len(snaps) > 0:
                final_psi = snaps[-1]  # (2, n, n)
                rho_final = np.abs(final_psi[0])**2 + np.abs(final_psi[1])**2
            else:
                rho_final = np.zeros_like(rho0)

            P_final_inside = np.sum(rho_final * detect_mask) * dx_val**2
            P_final_total  = np.sum(rho_final) * dx_val**2

            # Reflection coefficient
            # R = probability that came back to detection region / initial
            # Note: P0_inside includes the initial packet. After packet leaves,
            # P_inside should drop to ~0 if no reflection.
            R_coeff = P_final_inside / P0_total

            # Also measure total survival (how much was absorbed by CAP)
            absorption = 1.0 - P_final_total / P0_total

            config_results[E] = dict(
                R=float(R_coeff),
                absorption=float(absorption),
                P0_total=float(P0_total),
                P_final_total=float(P_final_total),
                P_final_inside=float(P_final_inside),
                n_steps=n_steps,
                T_prop=float(n_steps * dt),
            )

            status = "✓" if R_coeff < 0.01 else "✗"
            print(f"  E={E:.0f}: R={R_coeff:.4f} {status}, "
                  f"absorbed={absorption:.3f}, "
                  f"T={n_steps*dt:.0f}, {elapsed:.0f}s")

        results[label] = config_results

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    # 1. Reflection coefficient vs Energy for all CAP configs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for cap_r, cap_s, cap_p, label in CAP_CONFIGS:
        if label in results:
            Es = sorted(results[label].keys())
            Rs = [results[label][E]["R"] for E in Es]
            ax.semilogy(Es, Rs, "o-", label=label, ms=6)
    ax.axhline(0.01, color="r", ls="--", lw=1, label="1% threshold")
    ax.set_xlabel("Energy E")
    ax.set_ylabel("Reflection coefficient R")
    ax.set_title("CAP reflection test\n(free Dirac particle → CAP)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for cap_r, cap_s, cap_p, label in CAP_CONFIGS:
        if label in results:
            Es = sorted(results[label].keys())
            As = [results[label][E]["absorption"] for E in Es]
            ax.plot(Es, As, "o-", label=label, ms=6)
    ax.set_xlabel("Energy E")
    ax.set_ylabel("Absorption fraction")
    ax.set_title("Total absorption by CAP")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "reflection_test.png"), dpi=150)
    plt.close(fig)
    print(f"\n  → reflection_test.png")

    # Save metrics
    with open(os.path.join(OUT_DIR, "reflection_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → reflection_metrics.json")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for label in results:
        Rs = [results[label][E]["R"] for E in sorted(results[label].keys())]
        max_R = max(Rs)
        status = "PASS" if max_R < 0.01 else "FAIL"
        print(f"  {label:15s}: max R = {max_R:.4f}  [{status}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
