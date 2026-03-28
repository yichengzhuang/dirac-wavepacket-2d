#!/usr/bin/env python3
"""
LDOS for soft-wall potentials: ellipse and Bunimovich stadium.
V(r) = V₀ / (1 + λ exp[μ(1 - f(r)²)])  — bathtub shape.
M=48 propagations, n=256, n_E=300.

Parameters follow the reference: V₀=70, λ=25, μ=5.
"""
import sys, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (
    make_grid, gaussian_wavepacket, build_potential, radial_CAP,
    build_analysis_mask, build_split_operator, propagate,
)
from spectral import compute_ldos

OUT = os.path.join(_root, "figures", "ldos_softwall")
os.makedirs(OUT, exist_ok=True)

# ── Common ──
n=256; L=10.0; vF=1.0; m_mass=0.0
dt=0.01; n_steps=10000; save_every=50
sigma0=0.6; k_mag=3.0
cap_strength=10.0; cap_power=2

E_min, E_max, n_E = 0.5, 5.5, 300
E_values = np.linspace(E_min, E_max, n_E)

N_ang = 8
angles = np.linspace(0, 2*np.pi, N_ang, endpoint=False)
r_ring = 0.4
ring_angs = np.linspace(0, 2*np.pi, 5, endpoint=False)
positions = [(0.0, 0.0)] + [(r_ring*np.cos(a), r_ring*np.sin(a)) for a in ring_angs]
M = N_ang * len(positions)

# ── Shapes ──
SHAPES = [
    {
        "name": "soft_ellipse",
        "potential": "soft_ellipse",
        "pot_params": dict(a=2.0, b=1.2, V0=70.0, lam=25.0, mu=5.0),
        "cap_radius": 3.5,
    },
    {
        "name": "soft_stadium",
        "potential": "soft_stadium",
        "pot_params": dict(a_half=1.0, b=1.0, V0=70.0, lam=25.0, mu=5.0),
        "cap_radius": 3.0,
    },
]


def fft_flip(arr, axis):
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)

def sym_c2v(rho):
    rx = fft_flip(rho, axis=1)
    ry = fft_flip(rho, axis=0)
    return 0.25*(rho + rx + ry + fft_flip(rx, 0))

def c2v_err(rho):
    s = np.sum(rho) + 1e-30
    return (np.sum(np.abs(fft_flip(rho,1)-rho))/s,
            np.sum(np.abs(fft_flip(rho,0)-rho))/s)


def run_ldos(shape, xp, X, Y, KX, KY, K, dx, X_np, Y_np):
    name = shape["name"]
    pot = shape["potential"]
    params = shape["pot_params"]
    cr = shape["cap_radius"]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    V = build_potential(pot, X_np, Y_np, params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cr,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    chi_R, _ = build_analysis_mask(pot, X_np, Y_np, params,
        cap_radius=cr, safety_factor=0.95, mask_margin=1.30, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, m_mass, V, W, KX, KY, K, xp)

    ldos_sum = np.zeros((n_E, n, n))
    t0 = time.perf_counter()
    n_done = 0

    for x0, y0 in positions:
        for angle in angles:
            kx0 = k_mag * np.cos(angle)
            ky0 = k_mag * np.sin(angle)
            n_done += 1
            psi0_up, psi0_dn = gaussian_wavepacket(
                X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
                sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
            result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
                n_steps=n_steps, dt=dt, save_every=save_every,
                corr_every=save_every, max_record=n_steps//save_every+2,
                measure_transmission=False)
            ldos_one = compute_ldos(result["snapshots"], result["snap_times"],
                                     E_values, chi_R=chi_cpu)
            ldos_sum += ldos_one
            del result, ldos_one, psi0_up, psi0_dn

            if n_done % 8 == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / n_done * (M - n_done)
                print(f"  [{n_done}/{M}]  {elapsed:.0f}s  ETA {eta/60:.1f} min", flush=True)

    ldos_avg = ldos_sum / M
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    spec = (ldos_avg * chi_cpu[np.newaxis]).sum(axis=(1,2))
    top3 = np.argsort(spec)[::-1][:3]
    print(f"  C₂v errors (raw):")
    for idx in top3:
        xe, ye = c2v_err(ldos_avg[idx])
        print(f"    E={E_values[idx]:.3f}  x={xe:.4f}  y={ye:.4f}")

    np.save(os.path.join(OUT, f"ldos_{name}.npy"), ldos_avg)
    return ldos_avg, chi_cpu


def make_plots(results):
    half_L = L / 2
    extent = [-half_L, half_L, -half_L, half_L]
    x = np.linspace(-half_L, half_L, n, endpoint=False)
    ns = len(results)

    # Spectrum
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name, ldos, chi in results:
        spec = (ldos * chi[np.newaxis]).sum(axis=(1,2))
        ax.plot(E_values, spec/spec.max(), lw=2, label=name)
    ax.set_xlabel("E"); ax.set_ylabel("LDOS (norm.)")
    ax.set_title("Soft-wall LDOS spectrum (M=48, massless)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "spectrum.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Peak maps
    fig, axes = plt.subplots(ns, 4, figsize=(16, 4*ns))
    for i, (name, ldos, chi) in enumerate(results):
        spec = (ldos * chi[np.newaxis]).sum(axis=(1,2))
        peaks = np.sort(np.argsort(spec)[::-1][:4])
        for j, idx in enumerate(peaks):
            ax = axes[i, j] if ns > 1 else axes[j]
            rho = sym_c2v(ldos[idx])
            ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                      vmin=0, vmax=rho.max())
            ax.set_title(f"E={E_values[idx]:.2f}", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        (axes[i, 0] if ns > 1 else axes[0]).set_ylabel(name, fontsize=10)
    fig.suptitle("Soft-wall LDOS peaks (C₂v sym)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "peaks.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 1D line cuts
    fig, axes = plt.subplots(ns, 2, figsize=(14, 5*ns))
    for i, (name, ldos, chi) in enumerate(results):
        ldos_s = np.array([sym_c2v(ldos[j]) for j in range(n_E)])
        ax = axes[i, 0] if ns > 1 else axes[0]
        ax.pcolormesh(x, E_values, ldos_s[:, n//2, :], cmap="inferno", shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("E"); ax.set_title(f"{name}: LDOS(E,x)")
        ax.set_xlim(-3, 3)
        ax = axes[i, 1] if ns > 1 else axes[1]
        ax.pcolormesh(x, E_values, ldos_s[:, :, n//2], cmap="inferno", shading="auto")
        ax.set_xlabel("y"); ax.set_ylabel("E"); ax.set_title(f"{name}: LDOS(E,y)")
        ax.set_xlim(-2, 2)
    fig.suptitle("Soft-wall 1D LDOS (n_E=300, C₂v sym)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "linecuts.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figures → {OUT}/")


def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}, M={M}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)

    results = []
    for shape in SHAPES:
        ldos, chi = run_ldos(shape, xp, X, Y, KX, KY, K, dx, X_np, Y_np)
        results.append((shape["name"], ldos, chi))

    np.save(os.path.join(OUT, "E_values.npy"), E_values)
    make_plots(results)

    print(f"\nTotal: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
