#!/usr/bin/env python3
"""
Study 2.2: Level spacing statistics — integrable vs chaotic.

Theory predictions:
    Ellipse (integrable) → Poisson: P(s) = exp(-s)
    Stadium (chaotic)    → Wigner:  P(s) = (π/2)s exp(-πs²/4)

Method: Post-process existing LDOS data from ldos_shapes.py.
    Detect peaks, compute nearest-neighbor spacing, normalize, compare.

Output: figures/study_shapes/
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

from ldos_utils import integrated_spectrum, ensure_dir

OUT_DIR = ensure_dir(os.path.join(_root, "figures", "study_shapes"))

# Paths to existing LDOS data
SHAPES_DIR = os.path.join(_root, "figures", "ldos_shapes")
SOFTWALL_DIR = os.path.join(_root, "figures", "ldos_softwall")
LDOS_MAP_DIR = os.path.join(_root, "figures", "ldos_map")


def load_ldos_and_E(directory, filename):
    """Load LDOS array and E_values from a directory."""
    ldos = np.load(os.path.join(directory, filename))
    E_path = os.path.join(directory, "E_values.npy")
    E_values = np.load(E_path)
    return ldos, E_values


def extract_peak_spacings(spectrum, E_values, prominence=0.005, distance=2):
    """Extract nearest-neighbor spacings from spectrum peaks.

    Note: For massless Dirac, Γ/ΔE ~ O(1) means very few peaks are
    resolvable. Getting only 4-7 peaks in [0.5, 5.5] is EXPECTED,
    not a bug in peak detection.
    """
    idx, props = find_peaks(spectrum / spectrum.max(),
                            prominence=prominence, distance=distance)
    if len(idx) < 3:
        return np.array([]), np.array([])
    peak_E = E_values[idx]
    spacings = np.diff(np.sort(peak_E))
    return spacings, peak_E


def main():
    print("Study 2.2: Level spacing statistics")
    print(f"Output: {OUT_DIR}")

    # Load available data
    datasets = {}

    # Try loading shapes data
    for name, fname in [("harmonic_ellipse", "ldos_harmonic_ellipse.npy"),
                        ("harmonic_circle", "ldos_harmonic_circle.npy"),
                        ("soft_stadium", "ldos_soft_stadium.npy")]:
        fpath = os.path.join(SHAPES_DIR, fname)
        if os.path.exists(fpath):
            ldos, E_values = load_ldos_and_E(SHAPES_DIR, fname)
            # We need the mask — approximate by using the non-zero region
            chi = np.ones_like(ldos[0])
            spec = integrated_spectrum(ldos, chi)
            datasets[name] = dict(spectrum=spec, E_values=E_values)
            print(f"  Loaded {name}: {ldos.shape}")
        else:
            print(f"  Missing {fpath}")

    # Also try soft-wall data
    for name, fname in [("soft_ellipse", "ldos_soft_ellipse.npy"),
                        ("soft_stadium_sw", "ldos_soft_stadium.npy")]:
        fpath = os.path.join(SOFTWALL_DIR, fname)
        if os.path.exists(fpath):
            ldos, E_values = load_ldos_and_E(SOFTWALL_DIR, fname)
            chi = np.ones_like(ldos[0])
            spec = integrated_spectrum(ldos, chi)
            datasets[name] = dict(spectrum=spec, E_values=E_values)
            print(f"  Loaded {name}: {ldos.shape}")

    # Also load main LDOS map if available
    fpath = os.path.join(LDOS_MAP_DIR, "ldos_avg.npy")
    if os.path.exists(fpath):
        ldos, E_values = load_ldos_and_E(LDOS_MAP_DIR, "ldos_avg.npy")
        chi = np.ones_like(ldos[0])
        spec = integrated_spectrum(ldos, chi)
        datasets["harmonic_ellipse_hi"] = dict(spectrum=spec, E_values=E_values)
        print(f"  Loaded harmonic_ellipse_hi: {ldos.shape}")

    if not datasets:
        print("ERROR: No LDOS data found! Run ldos_shapes.py first.")
        return

    # ── Peak extraction and spacing statistics ──
    results = {}
    for name, data in datasets.items():
        spec = data["spectrum"]
        E_values = data["E_values"]

        # Use only E > 1.5 to avoid edge effects
        mask = E_values > 1.5
        spacings, peak_E = extract_peak_spacings(spec[mask], E_values[mask])

        if len(spacings) < 2:
            print(f"  {name}: only {len(spacings)} spacings, skipping")
            continue

        # Unfold: normalize spacings by local mean
        # (simple moving average of 5 spacings)
        if len(spacings) >= 5:
            kernel = np.ones(5) / 5
            local_mean = np.convolve(spacings, kernel, mode="same")
            local_mean[local_mean == 0] = spacings.mean()
            s_norm = spacings / local_mean
        else:
            s_norm = spacings / spacings.mean()

        results[name] = dict(
            spacings=spacings,
            s_norm=s_norm,
            n_peaks=len(peak_E),
            mean_spacing=float(spacings.mean()),
            std_spacing=float(spacings.std()),
            peak_energies=peak_E.tolist(),
        )
        print(f"  {name}: {len(peak_E)} peaks, "
              f"<s>={spacings.mean():.4f} ± {spacings.std():.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════════════════

    # 1. P(s) distributions
    s_theory = np.linspace(0, 4, 200)
    P_poisson = np.exp(-s_theory)
    P_wigner = (np.pi / 2) * s_theory * np.exp(-np.pi * s_theory**2 / 4)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s_theory, P_poisson, "k--", lw=2, label="Poisson (integrable)")
    ax.plot(s_theory, P_wigner, "k:", lw=2, label="Wigner (chaotic)")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (name, r) in enumerate(results.items()):
        s = r["s_norm"]
        ax.hist(s, bins=15, density=True, alpha=0.4, color=colors[i],
                label=f"{name} ({r['n_peaks']} peaks)")

    ax.set_xlabel("s (normalized spacing)")
    ax.set_ylabel("P(s)")
    ax.set_title("Nearest-neighbor level spacing distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "level_spacing_P_s.png"), dpi=150)
    plt.close(fig)
    print(f"  → level_spacing_P_s.png")

    # 2. Spectrum comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, data) in enumerate(datasets.items()):
        if name in results:
            spec = data["spectrum"]
            E = data["E_values"]
            ax.plot(E, spec / spec.max() + i * 0.3, label=name, alpha=0.8)
    ax.set_xlabel("E")
    ax.set_ylabel("LDOS (normalized, offset)")
    ax.set_title("Spectrum comparison across shapes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectrum_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  → spectrum_comparison.png")

    # Metrics
    metrics = {}
    for name, r in results.items():
        metrics[name] = dict(
            n_peaks=r["n_peaks"],
            mean_spacing=r["mean_spacing"],
            std_spacing=r["std_spacing"],
        )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Study 2.2 Summary:")
    for name, r in results.items():
        print(f"  {name}: {r['n_peaks']} peaks, "
              f"<s>={r['mean_spacing']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
