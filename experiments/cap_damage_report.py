#!/usr/bin/env python3
"""
CAP Step E: Damage assessment.

Compare LDOS from C1 (L=10, cap_R=3.5) vs C5 (L=20, cap_R=8.0)
at M=48 — energy-resolved comparison to determine which energy
ranges are reliable.

Reads the .npy outputs from cap_convergence_M48.py.

Output: figures/cap_validate/
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "scripts"))
sys.path.insert(0, _root)

from ldos_utils import measure_c2v_error, ensure_dir

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "cap_validate"))
E_values = np.linspace(0.5, 5.5, 300)


def build_common_mask(n, L, aR=2.6, bR=1.56):
    x = np.linspace(-L/2, L/2, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")
    rho2 = (X / aR)**2 + (Y / bR)**2
    return (rho2 <= 1.0).astype(np.float64)


def main():
    print("CAP Step E: Damage assessment")

    # Load M=48 LDOS from cap_convergence_M48.py
    c1_path = os.path.join(OUT_DIR, "ldos_M48_C1_L10_R3.5.npy")
    c5_path = os.path.join(OUT_DIR, "ldos_M48_C5_L20_R8.0.npy")

    if not os.path.exists(c1_path) or not os.path.exists(c5_path):
        print(f"ERROR: Need both {c1_path} and {c5_path}")
        print("Run cap_convergence_M48.py first.")
        return

    ldos_c1 = np.load(c1_path)  # (300, 256, 256)
    ldos_c5 = np.load(c5_path)  # (300, 512, 512)

    # Build common masks
    mask_c1 = build_common_mask(256, 10.0)
    mask_c5 = build_common_mask(512, 20.0)

    # Integrated spectra inside mask
    spec_c1 = (ldos_c1 * mask_c1[None]).sum(axis=(1, 2))
    spec_c5 = (ldos_c5 * mask_c5[None]).sum(axis=(1, 2))

    # Normalize
    s1 = spec_c1 / spec_c1.max()
    s5 = spec_c5 / spec_c5.max()

    # ── Energy-resolved error ──
    abs_diff = np.abs(s1 - s5)
    rel_diff = abs_diff / np.maximum(s5, 0.01)

    # Find peaks in reference (C5)
    idx5, _ = find_peaks(s5, prominence=0.03)
    peak_E5 = E_values[idx5]

    # For each peak, find closest peak in C1 and compare
    idx1, _ = find_peaks(s1, prominence=0.03)
    peak_E1 = E_values[idx1]

    print(f"\nC1 peaks: {len(idx1)}, C5 peaks: {len(idx5)}")
    print(f"\nPeak-by-peak comparison:")
    print(f"{'E_C5':>7} {'E_C1':>7} {'ΔE':>7} {'h_C1':>6} {'h_C5':>6} {'Δh/h':>7}")

    peak_data = []
    for i5 in idx5:
        E5 = E_values[i5]
        h5 = s5[i5]
        # Find closest C1 peak
        if len(idx1) > 0:
            closest = idx1[np.argmin(np.abs(E_values[idx1] - E5))]
            E1 = E_values[closest]
            h1 = s1[closest]
            dE = abs(E1 - E5)
            dh = abs(h1 - h5) / h5 if h5 > 0 else 0
        else:
            E1, h1, dE, dh = np.nan, np.nan, np.nan, np.nan

        peak_data.append(dict(E_C5=E5, E_C1=E1, dE=dE, h_C5=h5, h_C1=h1, dh=dh))
        print(f"{E5:7.3f} {E1:7.3f} {dE:7.4f} {h1:6.3f} {h5:6.3f} {dh:7.3f}")

    # ── Determine reliability boundary ──
    # Sliding window: average rel_diff in windows of width 0.5
    window = 30  # indices, ~0.5 in E
    rolling_err = np.convolve(rel_diff, np.ones(window)/window, mode="same")

    # Find E where rolling error first exceeds 10%
    threshold = 0.10
    above = np.where(rolling_err > threshold)[0]
    if len(above) > 0:
        E_boundary = E_values[above[0]]
    else:
        E_boundary = E_values[-1]

    # Global metrics
    l2_full = np.sqrt(np.mean((s1 - s5)**2))
    l2_low = np.sqrt(np.mean((s1[:150] - s5[:150])**2))   # E < 3
    l2_high = np.sqrt(np.mean((s1[150:] - s5[150:])**2))  # E > 3

    print(f"\n{'='*60}")
    print(f"Global L2 error:     {l2_full:.4f}")
    print(f"L2 (E < 3):          {l2_low:.4f}")
    print(f"L2 (E > 3):          {l2_high:.4f}")
    print(f"Reliability boundary: E < {E_boundary:.2f}")
    print(f"{'='*60}")

    # ── Figures ──

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. Spectra overlay
    ax = axes[0]
    ax.plot(E_values, s1, "b-", lw=1.5, label="C1: L=10, R=3.5 (current)")
    ax.plot(E_values, s5, "r--", lw=1.5, label="C5: L=20, R=8.0 (reference)")
    ax.set_ylabel("LDOS (normalized)")
    ax.set_title("CAP damage assessment (M=48)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Relative difference
    ax = axes[1]
    ax.plot(E_values, rel_diff, "k-", lw=0.8, alpha=0.5)
    ax.plot(E_values, rolling_err, "r-", lw=2, label="Rolling avg")
    ax.axhline(threshold, color="orange", ls="--", label=f"{threshold*100:.0f}% threshold")
    ax.axvline(E_boundary, color="g", ls="-", lw=2,
               label=f"Boundary: E={E_boundary:.2f}")
    ax.set_ylabel("Relative difference")
    ax.set_ylim(0, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Spatial comparison at strongest peak
    ax = axes[2]
    ax.plot(E_values, abs_diff, "k-", lw=1)
    ax.axhline(0.05, color="r", ls="--", label="5%")
    ax.axhline(0.01, color="g", ls="--", label="1%")
    ax.set_xlabel("E")
    ax.set_ylabel("|ΔLDOS|")
    ax.set_title("Absolute difference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cap_damage_report.png"), dpi=150)
    plt.close(fig)
    print(f"  → cap_damage_report.png")

    # Save
    metrics = dict(
        l2_full=float(l2_full),
        l2_low_E=float(l2_low),
        l2_high_E=float(l2_high),
        reliability_boundary=float(E_boundary),
        n_peaks_C1=len(idx1),
        n_peaks_C5=len(idx5),
        peak_comparison=peak_data,
    )
    with open(os.path.join(OUT_DIR, "cap_damage_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"  → cap_damage_metrics.json")


if __name__ == "__main__":
    main()
