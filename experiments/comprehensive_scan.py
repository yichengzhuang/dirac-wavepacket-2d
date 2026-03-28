#!/usr/bin/env python3
"""
Comprehensive scan: Tasks B + C + D combined.

Task B: Mass scan (m=0, 0.3, 0.5, 0.7, 1.0) + FDM comparison
Task C: FDM angle count scan (M=1,2,4,8) at best mass
Task D: Wavepacket parameters (anisotropic σ, off-center launch)

All share one propagation infrastructure, output to figures/fdm_debug/.
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
from spectral import extract_resonances, time_filter_mode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdm_debug import (
    build_probes_fixed, build_BA, solve_fdm, reconstruct_modes,
)

OUT = os.path.join(_root, "figures", "fdm_debug")
os.makedirs(OUT, exist_ok=True)

n=256; L=10.0; vF=1.0; dt=0.01; n_steps=15000; save_every=10
POTENTIAL_OPTION="harmonic_ellipse"
pot_params=dict(k_harm=3.0, a=2.0, b=1.2)
cap_radius=3.5; cap_strength=10.0; cap_power=2
mask_safety_factor=0.95; mask_margin=1.30
sigma0=0.6; k_mag=3.0
Gamma_cut=8.0; pencil_fraction=4; L_pencil_max=400


def spinor_overlap(a, b, chi, dx2):
    ab = (np.vdot(a[0]*chi, b[0]) + np.vdot(a[1]*chi, b[1])) * dx2
    aa = (np.vdot(a[0]*chi, a[0]) + np.vdot(a[1]*chi, a[1])) * dx2
    bb = (np.vdot(b[0]*chi, b[0]) + np.vdot(b[1]*chi, b[1])) * dx2
    d = np.real(aa)*np.real(bb)
    return float(np.abs(ab)**2/d) if d > 1e-30 else 0.0

def fft_flip(a, axis):
    return np.roll(np.flip(a, axis=axis), 1, axis=axis)

def c2v_err(rho):
    s = np.sum(rho)+1e-30
    return np.sum(np.abs(fft_flip(rho,1)-rho))/s, np.sum(np.abs(fft_flip(rho,0)-rho))/s


def do_propagation(xp, mass, kx0, ky0, X, Y, KX, KY, K, dx,
                   X_np, Y_np, chi_R, chi_cpu,
                   sig_x=None, sig_y=None, x0=0.0, y0=0.0):
    """Run one propagation. Supports anisotropic sigma via manual envelope."""
    V = build_potential(POTENTIAL_OPTION, X_np, Y_np, pot_params, xp)
    W = radial_CAP(X_np, Y_np, cap_radius=cap_radius,
                   cap_strength=cap_strength, cap_power=cap_power, xp=xp)
    sqrt_chi = xp.sqrt(chi_R)
    halfV, kin = build_split_operator(vF, dt, mass, V, W, KX, KY, K, xp)

    if sig_x is not None and sig_y is not None and sig_x != sig_y:
        # Anisotropic wavepacket: build manually
        env_np = np.exp(-((X_np-x0)**2/(2*sig_x**2) + (Y_np-y0)**2/(2*sig_y**2)))
        k_abs = np.sqrt(kx0**2 + ky0**2)
        if k_abs > 0:
            theta_k = np.arctan2(ky0, kx0)
        else:
            theta_k = 0.0
        phase = np.exp(1j*(kx0*X_np + ky0*Y_np))
        # Chiral spinor (conduction band)
        psi_up_np = env_np * phase * np.cos(theta_k/2)
        psi_dn_np = env_np * phase * np.sin(theta_k/2) * np.exp(1j*theta_k)
        # normalize
        dx_np = L/n
        rho = np.abs(psi_up_np)**2 + np.abs(psi_dn_np)**2
        norm = np.sqrt(np.sum(rho) * dx_np**2)
        psi_up_np /= norm; psi_dn_np /= norm
        psi0_up = xp.array(psi_up_np.astype(np.complex128))
        psi0_dn = xp.array(psi_dn_np.astype(np.complex128))
    else:
        sigma = sig_x if sig_x else sigma0
        psi0_up, psi0_dn = gaussian_wavepacket(
            X, Y, x0=x0, y0=y0, kx0=kx0, ky0=ky0,
            sigma=sigma, theta=None, band="conduction", dx=dx, xp=xp)

    result = propagate(psi0_up, psi0_dn, halfV, kin, sqrt_chi, chi_R, dx, xp,
        n_steps=n_steps, dt=dt, save_every=save_every, corr_every=save_every,
        max_record=n_steps//save_every+2, measure_transmission=False)

    psi0_cpu = np.stack([xp.to_numpy(psi0_up), xp.to_numpy(psi0_dn)])
    return dict(snapshots=result["snapshots"], snap_times=result["snap_times"],
                C_R=result["C_R"], corr_times=result["corr_times"],
                psi0_cpu=psi0_cpu)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Task B: Mass scan                                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

def task_B(xp, X, Y, KX, KY, K, dx, X_np, Y_np, chi_R, chi_cpu, dx2):
    print("\n" + "="*70)
    print("  TASK B: Mass scan + FDM comparison")
    print("="*70)

    dt_snap = dt * save_every
    masses = [0.0, 0.3, 0.5, 0.7, 1.0]
    kx0 = k_mag/np.sqrt(2); ky0 = k_mag/np.sqrt(2)

    results = []
    for mass_val in masses:
        print(f"\n  m = {mass_val} ...")
        data = do_propagation(xp, mass_val, kx0, ky0, X, Y, KX, KY, K, dx,
                              X_np, Y_np, chi_R, chi_cpu)

        spec = extract_resonances(data["C_R"], data["corr_times"], dt_snap,
            Gamma_cut=Gamma_cut, pencil_fraction=pencil_fraction,
            L_pencil_max=L_pencil_max)
        E_mp, G_mp, omega_mp = spec["E"], spec["Gamma"], spec["omega"]

        # Physical filter: shift energy window by m to compare equivalent modes
        # Dirac dispersion E² = k² + m²: same k (same spatial mode) → E shifts by ~m
        E_lo = mass_val + 2.0
        E_hi = mass_val + 5.0
        good = ((E_mp > E_lo) & (E_mp < E_hi) & (G_mp > 0) & (G_mp < 2.0))
        if good.sum() < 2:
            good = (E_mp > max(mass_val, 0.1)) & (E_mp < 8.0) & (G_mp > 0) & (G_mp < 2.0)
        E_mp, G_mp, omega_mp = E_mp[good], G_mp[good], omega_mp[good]
        # sort by amplitude (strongest poles first, not narrowest)
        order_mp = np.arange(len(E_mp))  # already sorted by |a| from extract_resonances
        n_mp = min(5, len(E_mp))

        print(f"    Conduction-band poles (E∈[{E_lo:.1f},{E_hi:.1f}], Γ<2.0): "
              f"{len(E_mp)} found, using top {n_mp}")
        for j in range(n_mp):
            print(f"      E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

        # Γ/ΔE for narrow modes only
        ratios = []
        for j in range(n_mp):
            dEs = [abs(E_mp[j]-E_mp[k]) for k in range(n_mp) if k != j]
            dE_min = min(dEs) if dEs else 1.0
            ratios.append(G_mp[j] / dE_min if dE_min > 0 else 99)
        avg_ratio = np.mean(ratios) if ratios else 99

        # time-filter top-3 narrow modes
        tf_modes = [time_filter_mode(data["snapshots"], data["snap_times"],
                    omega_mp[j], chi_cpu) for j in range(min(3, n_mp))]

        # FDM (MP-guided, using narrow mode energies)
        E_probes = E_mp[:n_mp].copy()
        if len(E_probes) < 5:
            E_probes = np.sort(np.concatenate([
                E_probes, np.linspace(max(mass_val+0.5, 1.0), 5.0,
                                       5-len(E_probes))]))
        chi_flat = chi_cpu.ravel()
        probes, shifted = build_probes_fixed(data["snapshots"], data["snap_times"],
                                              E_probes, chi_flat)
        B, A = build_BA(probes, shifted, dx2)
        fdm = solve_fdm(B, A, dt_snap, eps_svd=1e-2)

        # FDM overlap with TF
        fdm_modes = []
        overlaps = []
        if fdm["vecs"].shape[1] > 0:
            n_fdm = min(5, fdm["vecs"].shape[1])
            order = np.argsort(fdm["Gamma"])[:n_fdm]
            fdm["E"] = fdm["E"][order]
            fdm["Gamma"] = fdm["Gamma"][order]
            fdm["vecs"] = fdm["vecs"][:, order]
            fdm_modes = reconstruct_modes(fdm["vecs"], probes, n, chi_cpu, dx2)

            for j in range(min(len(fdm_modes), len(tf_modes))):
                dists = np.abs(fdm["E"][j] - E_mp[:n_mp])
                i_best = np.argmin(dists)
                if i_best < len(tf_modes):
                    overlaps.append(spinor_overlap(fdm_modes[j], tf_modes[i_best],
                                                   chi_cpu, dx2))

        # C₂v of TF modes
        tf_xerr = [c2v_err(np.abs(m[0])**2+np.abs(m[1])**2)[0] for m in tf_modes]
        avg_xerr = np.mean(tf_xerr) if tf_xerr else 1.0
        avg_overlap = np.mean(overlaps) if overlaps else 0.0

        print(f"    MP poles: {n_mp}, Γ/ΔE avg: {avg_ratio:.3f}, "
              f"TF x_err avg: {avg_xerr:.4f}, FDM overlap avg: {avg_overlap:.4f}")

        results.append(dict(m=mass_val, n_mp=n_mp, avg_ratio=avg_ratio,
                            avg_xerr=avg_xerr, avg_overlap=avg_overlap,
                            E_mp=E_mp[:n_mp], G_mp=G_mp[:n_mp]))

        del data

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ms = [r["m"] for r in results]

    ax = axes[0]
    ax.plot(ms, [r["avg_ratio"] for r in results], 'o-', lw=2, color='C0')
    ax.axhline(1.0, ls='--', color='r', alpha=0.5, label='Γ/ΔE = 1')
    ax.set_xlabel("mass m"); ax.set_ylabel("avg Γ/ΔE")
    ax.set_title("Mode separability"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ms, [r["avg_xerr"] for r in results], 's-', lw=2, color='C1')
    ax.set_xlabel("mass m"); ax.set_ylabel("avg x_err (TF)")
    ax.set_title("Time filter quality"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ms, [r["avg_overlap"] for r in results], '^-', lw=2, color='C2')
    ax.axhline(0.9, ls='--', color='r', alpha=0.5, label='0.9 threshold')
    ax.set_xlabel("mass m"); ax.set_ylabel("avg FDM-TF overlap")
    ax.set_title("FDM spatial accuracy"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Task B: Mass scan — mode extraction quality vs mass", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT, "taskB_mass_scan.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Figure → {out}")

    return results


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Task C: Angle count scan                                       ║
# ╚═══════════════════════════════════════════════════════════════════╝

def task_C(xp, X, Y, KX, KY, K, dx, X_np, Y_np, chi_R, chi_cpu, dx2,
           best_mass=0.5):
    print("\n" + "="*70)
    print(f"  TASK C: FDM angle count scan (m={best_mass})")
    print("="*70)

    dt_snap = dt * save_every
    M_values = [1, 2, 4, 8]
    chi_flat = chi_cpu.ravel()

    # Pre-run 8 propagations (reuse subsets for M=1,2,4,8)
    angles_all = np.linspace(0, 2*np.pi, 8, endpoint=False)
    all_data = []
    for ia, angle in enumerate(angles_all):
        kx0 = k_mag*np.cos(angle); ky0 = k_mag*np.sin(angle)
        print(f"  Propagation {ia+1}/8: k=({kx0:+.2f},{ky0:+.2f})")
        data = do_propagation(xp, best_mass, kx0, ky0, X, Y, KX, KY, K, dx,
                              X_np, Y_np, chi_R, chi_cpu)
        all_data.append(data)

    # MP reference from first propagation
    spec = extract_resonances(all_data[0]["C_R"], all_data[0]["corr_times"],
        dt_snap, Gamma_cut=Gamma_cut, pencil_fraction=pencil_fraction,
        L_pencil_max=L_pencil_max)
    E_mp = spec["E"]; G_mp = spec["Gamma"]
    good = (E_mp > 0) & (G_mp < 2.0)
    E_mp, G_mp = E_mp[good], G_mp[good]
    n_mp = min(5, len(E_mp))

    results = []
    for M_ang in M_values:
        print(f"\n  M = {M_ang} directions ...")
        # Use first M_ang propagations
        E_probes = E_mp[:n_mp].copy()
        if len(E_probes) < 5:
            E_probes = np.sort(np.concatenate([
                E_probes, np.linspace(1.0, 5.0, 5-len(E_probes))]))

        all_probes = []; all_shifted = []
        for ia in range(M_ang):
            p, s = build_probes_fixed(all_data[ia]["snapshots"],
                                       all_data[ia]["snap_times"],
                                       E_probes, chi_flat)
            all_probes.append(p); all_shifted.append(s)

        probes_cat = np.concatenate(all_probes)
        shifted_cat = np.concatenate(all_shifted)
        B, A = build_BA(probes_cat, shifted_cat, dx2)
        fdm = solve_fdm(B, A, dt_snap, eps_svd=1e-2)

        from fdm_debug import metric_spectral_match
        sm = metric_spectral_match(fdm["E"], fdm["Gamma"], E_mp[:n_mp], G_mp[:n_mp])
        print(f"    FDM poles: {len(fdm['E'])}, spectral_match: {sm:.4f}")
        results.append(dict(M=M_ang, n_poles=len(fdm["E"]), sm=sm))

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    Ms = [r["M"] for r in results]
    ax1.plot(Ms, [r["sm"] for r in results], 'o-', lw=2)
    ax1.set_xlabel("# propagation directions M"); ax1.set_ylabel("spectral match")
    ax1.set_title(f"FDM quality vs angle count (m={best_mass})")
    ax1.grid(alpha=0.3)

    ax2.plot(Ms, [r["n_poles"] for r in results], 's-', lw=2, color='C1')
    ax2.set_xlabel("# directions M"); ax2.set_ylabel("# physical poles")
    ax2.set_title("Number of extracted poles"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT, "taskC_angle_scan.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Figure → {out}")

    del all_data
    return results


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Task D: Wavepacket parameters                                  ║
# ╚═══════════════════════════════════════════════════════════════════╝

def task_D(xp, X, Y, KX, KY, K, dx, X_np, Y_np, chi_R, chi_cpu, dx2):
    print("\n" + "="*70)
    print("  TASK D: Wavepacket parameter exploration (m=0)")
    print("="*70)

    dt_snap = dt * save_every
    kx0 = k_mag/np.sqrt(2); ky0 = k_mag/np.sqrt(2)

    configs = [
        ("isotropic σ=0.6",        dict(sig_x=0.6, sig_y=0.6, x0=0.0, y0=0.0)),
        ("narrow-x σx=0.3 σy=0.9", dict(sig_x=0.3, sig_y=0.9, x0=0.0, y0=0.0)),
        ("narrow-y σx=0.9 σy=0.3", dict(sig_x=0.9, sig_y=0.3, x0=0.0, y0=0.0)),
        ("off-center x0=1.0",      dict(sig_x=0.6, sig_y=0.6, x0=1.0, y0=0.0)),
        ("wide σ=1.2",             dict(sig_x=1.2, sig_y=1.2, x0=0.0, y0=0.0)),
    ]

    results = []
    for name, params in configs:
        print(f"\n  {name} ...")
        data = do_propagation(xp, 0.0, kx0, ky0, X, Y, KX, KY, K, dx,
                              X_np, Y_np, chi_R, chi_cpu, **params)

        spec = extract_resonances(data["C_R"], data["corr_times"], dt_snap,
            Gamma_cut=Gamma_cut, pencil_fraction=pencil_fraction,
            L_pencil_max=L_pencil_max)
        E_mp, G_mp = spec["E"], spec["Gamma"]
        good = (E_mp > 0) & (G_mp < 2.0)
        E_mp, G_mp = E_mp[good], G_mp[good]
        n_mp = min(5, len(E_mp))

        ratios = []
        for j in range(n_mp):
            dEs = [abs(E_mp[j]-E_mp[k]) for k in range(n_mp) if k != j]
            dE_min = min(dEs) if dEs else 1.0
            ratios.append(G_mp[j]/dE_min if dE_min > 0 else 99)
        avg_ratio = np.mean(ratios) if ratios else 99

        # TF C₂v
        omega_mp = spec["omega"][good]
        tf = [time_filter_mode(data["snapshots"], data["snap_times"],
              omega_mp[j], chi_cpu) for j in range(min(3, n_mp))]
        xerrs = [c2v_err(np.abs(m[0])**2+np.abs(m[1])**2)[0] for m in tf]
        avg_xerr = np.mean(xerrs) if xerrs else 1.0

        print(f"    n_mp={n_mp}, avg Γ/ΔE={avg_ratio:.3f}, avg x_err={avg_xerr:.4f}")
        results.append(dict(name=name, n_mp=n_mp, avg_ratio=avg_ratio,
                            avg_xerr=avg_xerr))
        del data

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    names = [r["name"] for r in results]
    x = np.arange(len(names))

    ax1.bar(x, [r["avg_ratio"] for r in results], color='C0')
    ax1.axhline(1.0, ls='--', color='r')
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=7, rotation=20)
    ax1.set_ylabel("avg Γ/ΔE"); ax1.set_title("Mode separability")

    ax2.bar(x, [r["avg_xerr"] for r in results], color='C1')
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=7, rotation=20)
    ax2.set_ylabel("avg x_err (TF)"); ax2.set_title("Time filter quality")

    fig.suptitle("Task D: Wavepacket parameter exploration (m=0)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT, "taskD_wavepacket.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Figure → {out}")
    return results


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  Main                                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝

def main():
    t0 = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx2 = (L/n)**2

    chi_R, _ = build_analysis_mask(POTENTIAL_OPTION, X_np, Y_np, pot_params,
        cap_radius=cap_radius, safety_factor=mask_safety_factor,
        mask_margin=mask_margin, xp=xp)
    chi_cpu = xp.to_numpy(chi_R)

    rB = task_B(xp, X, Y, KX, KY, K, dx, X_np, Y_np, chi_R, chi_cpu, dx2)

    # Find best mass for Task C:
    # Want smallest Γ/ΔE (best separability) among masses where TF is imperfect
    # (if TF already perfect, FDM can't do better)
    valid = [r for r in rB if r["avg_ratio"] < 2.0 and r["n_mp"] >= 2]
    if valid:
        best_m = min(valid, key=lambda r: r["avg_ratio"])["m"]
    else:
        best_m = 0.5
    print(f"\nBest mass for angle scan: m={best_m}")

    rC = task_C(xp, X, Y, KX, KY, K, dx, X_np, Y_np, chi_R, chi_cpu, dx2,
                best_mass=best_m)

    rD = task_D(xp, X, Y, KX, KY, K, dx, X_np, Y_np, chi_R, chi_cpu, dx2)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"  TOTAL: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
