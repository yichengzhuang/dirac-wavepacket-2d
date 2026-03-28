#!/usr/bin/env python3
"""
System size scan: verify Γ/ΔE ~ O(1) independent of billiard size.

Scan a = 1.5, 2.0, 2.5, 3.0 (keeping aspect ratio b/a = 0.6 fixed).
For each size: propagate, extract MP poles, compute Γ/ΔE.

Theory predicts: Γ/ΔE ~ EL/(2πvF) ~ O(1) at lowest modes,
independent of L. This is the key signature of Klein tunnelling.

For comparison, also run Schrödinger (massive Dirac, m=5) where
Γ/ΔE should be exponentially small.
"""
import sys, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (make_grid, gaussian_wavepacket, build_potential,
    radial_CAP, build_analysis_mask, build_split_operator, propagate)
from spectral import extract_resonances

OUT = os.path.join(_root, "figures", "size_scan")
os.makedirs(OUT, exist_ok=True)

vF = 1.0; dt = 0.01; save_every = 10
k_harm = 3.0; aspect = 0.6  # b/a fixed
sigma0 = 0.6; Gamma_cut = 8.0

# Size configurations: a values and matching parameters
SIZES = [
    dict(a=1.5, b=0.9,  L=8.0,   n=256, cap_radius=2.8, n_steps=15000, k_mag=3.0),
    dict(a=2.0, b=1.2,  L=10.0,  n=256, cap_radius=3.5, n_steps=15000, k_mag=3.0),
    dict(a=2.5, b=1.5,  L=12.0,  n=256, cap_radius=4.5, n_steps=15000, k_mag=3.0),
    dict(a=3.0, b=1.8,  L=14.0,  n=256, cap_radius=5.5, n_steps=20000, k_mag=3.0),
]


def run_and_extract(xp, mass, size_cfg):
    """Propagate + MP extract for one size."""
    a = size_cfg["a"]; b = size_cfg["b"]; L = size_cfg["L"]
    nn = size_cfg["n"]; cr = size_cfg["cap_radius"]
    n_steps = size_cfg["n_steps"]; k_mag = size_cfg["k_mag"]

    x_np = np.linspace(-L/2, L/2, nn, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(nn, L, xp)

    pot_params = dict(k_harm=k_harm, a=a, b=b)
    V = build_potential("harmonic_ellipse", X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cr, cap_strength=10.0, cap_power=2, xp=xp)
    chi_R, _ = build_analysis_mask("harmonic_ellipse", X_np, Y_np, pot_params,
        cap_radius=cr, safety_factor=0.95, mask_margin=1.30, xp=xp)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, mass, V, W, KX, KY, K, xp)

    kx0 = k_mag / np.sqrt(2); ky0 = kx0
    psi0_up, psi0_dn = gaussian_wavepacket(
        X, Y, x0=0, y0=0, kx0=kx0, ky0=ky0,
        sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)

    result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
        max_record=n_steps//save_every+2, measure_transmission=False)

    dt_snap = dt * save_every
    spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
        Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)

    # Filter: conduction band, reasonable Γ
    E_lo = mass + 1.5
    good = (spec["E"] > E_lo) & (spec["E"] < E_lo + 4) & (spec["Gamma"] > 0) & (spec["Gamma"] < 3)
    E = spec["E"][good]; G = spec["Gamma"][good]

    # Compute ΔE (nearest neighbour spacing)
    if len(E) >= 2:
        E_sorted = np.sort(E)
        spacings = np.diff(E_sorted)
        avg_dE = np.mean(spacings)
    else:
        avg_dE = 1.0

    # Compute avg Γ/ΔE for top modes
    n_use = min(5, len(E))
    if n_use > 0:
        avg_G = np.mean(G[:n_use])
        ratios = G[:n_use] / avg_dE if avg_dE > 0 else G[:n_use]
        avg_ratio = np.mean(ratios)
    else:
        avg_G = 0; avg_ratio = 0; ratios = []

    return dict(E=E, G=G, avg_dE=avg_dE, avg_G=avg_G, avg_ratio=avg_ratio,
                n_poles=len(E), ratios=ratios[:n_use])


def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    # ── Massless (m=0) scan ──
    print("\n" + "="*60)
    print("  MASSLESS (m=0): Γ/ΔE vs system size")
    print("="*60)

    m0_results = []
    for cfg in SIZES:
        a = cfg["a"]
        print(f"\n  a={a}, b={cfg['b']}, L={cfg['L']}, n={cfg['n']} ...")
        r = run_and_extract(xp, 0.0, cfg)
        area = np.pi * a * cfg["b"]
        print(f"    {r['n_poles']} poles, avg ΔE={r['avg_dE']:.4f}, "
              f"avg Γ={r['avg_G']:.4f}, avg Γ/ΔE={r['avg_ratio']:.3f}")
        m0_results.append((a, area, r))

    # ── Massive (m=5) scan for comparison ──
    print("\n" + "="*60)
    print("  MASSIVE (m=5): Γ/ΔE vs system size (Schrödinger-like)")
    print("="*60)

    m5_results = []
    for cfg in SIZES:
        a = cfg["a"]
        cfg_m = dict(cfg)
        cfg_m["k_mag"] = max(6.0, cfg["k_mag"])  # need higher k to be above gap
        print(f"\n  a={a}, b={cfg['b']}, L={cfg['L']} ...")
        r = run_and_extract(xp, 5.0, cfg_m)
        area = np.pi * a * cfg["b"]
        print(f"    {r['n_poles']} poles, avg ΔE={r['avg_dE']:.4f}, "
              f"avg Γ={r['avg_G']:.4f}, avg Γ/ΔE={r['avg_ratio']:.3f}")
        m5_results.append((a, area, r))

    # ── Summary table ──
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n{'a':>5} {'Area':>6} {'m=0 Γ/ΔE':>10} {'m=5 Γ/ΔE':>10} {'m=0 Γ':>8} {'m=5 Γ':>8}")
    print("-" * 55)
    for i in range(len(SIZES)):
        a0, area0, r0 = m0_results[i]
        a5, area5, r5 = m5_results[i]
        print(f"{a0:5.1f} {area0:6.1f} {r0['avg_ratio']:10.3f} {r5['avg_ratio']:10.3f} "
              f"{r0['avg_G']:8.4f} {r5['avg_G']:8.4f}")

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    a_vals = [m[0] for m in m0_results]
    area_vals = [m[1] for m in m0_results]

    # Panel 1: Γ/ΔE vs area
    ax = axes[0]
    ax.plot(area_vals, [m[2]["avg_ratio"] for m in m0_results],
            'o-', ms=8, lw=2, color='C0', label='massless ($m=0$)')
    ax.plot(area_vals, [m[2]["avg_ratio"] for m in m5_results],
            's-', ms=8, lw=2, color='C1', label='massive ($m=5$)')
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel("Billiard area $A = \\pi ab$")
    ax.set_ylabel("$\\langle\\Gamma/\\Delta E\\rangle$")
    ax.set_title("$\\Gamma/\\Delta E$ vs system size")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 2: Γ vs area
    ax = axes[1]
    ax.plot(area_vals, [m[2]["avg_G"] for m in m0_results],
            'o-', ms=8, lw=2, color='C0', label='massless')
    ax.plot(area_vals, [m[2]["avg_G"] for m in m5_results],
            's-', ms=8, lw=2, color='C1', label='massive')
    ax.set_xlabel("Billiard area $A$")
    ax.set_ylabel("$\\langle\\Gamma\\rangle$")
    ax.set_title("Resonance width vs size")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 3: ΔE vs area
    ax = axes[2]
    ax.plot(area_vals, [m[2]["avg_dE"] for m in m0_results],
            'o-', ms=8, lw=2, color='C0', label='massless')
    ax.plot(area_vals, [m[2]["avg_dE"] for m in m5_results],
            's-', ms=8, lw=2, color='C1', label='massive')
    # Weyl prediction: ΔE = 2πvF²/(EA) ∝ 1/A
    A_th = np.linspace(min(area_vals)*0.9, max(area_vals)*1.1, 50)
    E_typ = 3.5
    ax.plot(A_th, 2*np.pi*vF**2/(E_typ*A_th), 'k--', alpha=0.5, label='Weyl $\\propto 1/A$')
    ax.set_xlabel("Billiard area $A$")
    ax.set_ylabel("$\\langle\\Delta E\\rangle$")
    ax.set_title("Level spacing vs size")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle("System size dependence: massless vs massive Dirac", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "size_scan.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure → {OUT}/size_scan.png")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
