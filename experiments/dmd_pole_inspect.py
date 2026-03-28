#!/usr/bin/env python3
"""Quick inspection: print ALL DMD poles for m=0 at rank=30,40.
Check if poles come in Kramers pairs and if multiple modes cluster near same E."""
import sys, os, numpy as np
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from accel import get_backend
from accel.dynamics_accel import (make_grid, gaussian_wavepacket, build_potential,
    radial_CAP, build_analysis_mask, build_split_operator, propagate)
from spectral import extract_resonances
from scripts.dmd_extract import exact_dmd, spinor_overlap, c2v_err

n=256; L=10.0; vF=1.0; dt=0.01; n_steps=15000; save_every=10
pot_params=dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
sigma0=0.6; k_mag=3.0; Gamma_cut=8.0

xp = get_backend("auto")
x_np = np.linspace(-L/2, L/2, n, endpoint=False)
X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
X, Y, KX, KY, K, dx = make_grid(n, L, xp)
dx2 = (L/n)**2

print("Propagating m=0 (45°)...")
kx0 = k_mag/np.sqrt(2); ky0 = kx0
V = build_potential("harmonic_ellipse", X_np, Y_np, pot_params, xp)
W = radial_CAP(X_np, Y_np, cap_radius=cap_radius, cap_strength=cap_strength, cap_power=cap_power, xp=xp)
chi_R, _ = build_analysis_mask("harmonic_ellipse", X_np, Y_np, pot_params,
    cap_radius=cap_radius, safety_factor=0.95, mask_margin=1.30, xp=xp)
chi_cpu = xp.to_numpy(chi_R); sqrt_chi = xp.sqrt(chi_R)
halfV, kin = build_split_operator(vF, dt, 0.0, V, W, KX, KY, K, xp)
p0u, p0d = gaussian_wavepacket(X, Y, x0=0, y0=0, kx0=kx0, ky0=ky0,
    sigma=sigma0, theta=None, band="conduction", dx=dx, xp=xp)
result = propagate(p0u, p0d, halfV, kin, sqrt_chi, chi_R, dx, xp,
    n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
    max_record=n_steps//save_every+2, measure_transmission=False)
snaps = result["snapshots"]; times = result["snap_times"]

# MP reference
dt_snap = dt * save_every
spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
    Gamma_cut=Gamma_cut, pencil_fraction=4, L_pencil_max=400)
good = (spec["E"]>2)&(spec["E"]<6)&(spec["Gamma"]<2)
E_mp = spec["E"][good]; G_mp = spec["Gamma"][good]
print(f"\nMP poles in [2,6] ({len(E_mp)}):")
for j in range(min(10, len(E_mp))):
    print(f"  E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

for rank in [20, 30, 40]:
    dmd = exact_dmd(snaps, times, chi_cpu, rank=rank, n_eff=150)
    E = dmd["E"]; G = dmd["Gamma"]

    print(f"\n{'='*60}")
    print(f"DMD rank={rank}: {len(E)} physical poles (sorted by Γ)")
    print(f"{'='*60}")
    print(f"{'#':>3} {'E':>8} {'Γ':>8} {'Γ/ΔE':>8} {'x_err':>8} {'y_err':>8}  {'pair?':>6}")

    for j in range(len(E)):
        # Check for Kramers pair: another pole within ΔE < 0.05
        pair = ""
        for k in range(len(E)):
            if k != j and abs(E[k] - E[j]) < 0.05 and abs(G[k] - G[j]) < 0.05:
                pair = f"↔{k+1}"
                break

        # C2v error
        if j < len(dmd["modes"]):
            rho = np.abs(dmd["modes"][j][0])**2 + np.abs(dmd["modes"][j][1])**2
            xe, ye = c2v_err(rho)
        else:
            xe = ye = -1

        # Nearest neighbor ΔE
        dEs = [abs(E[j] - E[k]) for k in range(len(E)) if k != j]
        dE_min = min(dEs) if dEs else 1.0
        ratio = G[j] / dE_min if dE_min > 0 else 99

        print(f"{j+1:3d} {E[j]:8.4f} {G[j]:8.4f} {ratio:8.3f} {xe:8.4f} {ye:8.4f}  {pair:>6}")

    # Check pairwise overlap between modes
    print(f"\nPairwise overlap (top 6 modes):")
    n_check = min(6, len(dmd["modes"]))
    print(f"     ", end="")
    for j in range(n_check):
        print(f"  m{j+1:d}   ", end="")
    print()
    for i in range(n_check):
        print(f"  m{i+1:d}", end="")
        for j in range(n_check):
            ov = spinor_overlap(dmd["modes"][i], dmd["modes"][j], chi_cpu, dx2)
            print(f"  {ov:.3f}", end="")
        print()
