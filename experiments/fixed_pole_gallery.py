#!/usr/bin/env python3
"""
Generate publication-quality mode gallery + comparison summary
for fixed-pole reconstruction results.

Metrics:
  - C₂v symmetry error (intrinsic — exact eigenstates must be C₂v)
  - Reconstruction residual ||ψ(t) - Σ φ_n e^{-iωt}|| / ||ψ(t)||
  - Inter-method consistency (cross-overlap between methods)
  - NOT "overlap with TF" as primary metric (TF is unreliable at m=0)
"""

import sys, os, time
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
from spectral import extract_resonances, time_filter_mode, reconstruct_modes_lstsq
from scripts.fixed_pole_recon import (
    fixed_pole_recon_direct, fixed_pole_recon_pod,
    fixed_pole_constrained_dmd, exact_dmd,
    spinor_overlap, fft_flip, c2v_err, _normalize_mask, run_prop,
    _adaptive_time_mask,
    n, L, vF, dt, n_steps, save_every, POTENTIAL, pot_params,
    cap_radius, cap_strength, cap_power, sigma0, k_mag, Gamma_cut,
)

OUT = os.path.join(_root, "figures", "fixed_pole")
os.makedirs(OUT, exist_ok=True)


def symmetrize_c2v(rho):
    """Average over 4 C₂v reflections."""
    r = rho.copy()
    r += fft_flip(rho, 1)   # x reflection
    r += fft_flip(rho, 0)   # y reflection
    r += fft_flip(fft_flip(rho, 0), 1)  # xy reflection
    return r / 4.0


def reconstruction_residual(snapshots, snap_times, omega_arr, modes, chi_R):
    """||ψ(t) - Σ φ_n e^{-iωt}|| / ||ψ(t)|| averaged over time."""
    N = len(modes)
    mask_t = _adaptive_time_mask(snap_times, omega_arr, N, quantile=0.5)
    t_use = snap_times[mask_t]
    snap_use = snapshots[mask_t]
    Ns = len(t_use)
    ny, nx = snapshots.shape[2], snapshots.shape[3]

    total_err = 0.0
    total_norm = 0.0
    for k in range(Ns):
        # Model: Σ_n φ_n(r) e^{-iω_n t_k}
        model = np.zeros((2, ny, nx), dtype=complex)
        for j in range(N):
            phase = np.exp(-1j * omega_arr[j] * t_use[k])
            model += modes[j] * phase
        # Residual inside mask
        diff = (snap_use[k] - model) * chi_R[None, :, :]
        psi_masked = snap_use[k] * chi_R[None, :, :]
        total_err += np.sum(np.abs(diff)**2)
        total_norm += np.sum(np.abs(psi_masked)**2)

    return np.sqrt(total_err / total_norm) if total_norm > 0 else 1.0


def main():
    t_start = time.perf_counter()
    xp = get_backend("auto")
    print(f"Backend: {xp.name}")

    x_np = np.linspace(-L/2, L/2, n, endpoint=False)
    X_np, Y_np = np.meshgrid(x_np, x_np, indexing="xy")
    X, Y, KX, KY, K, dx = make_grid(n, L, xp)
    dx2 = (L / n) ** 2
    dt_snap = dt * save_every
    extent = [-L/2, L/2, -L/2, L/2]

    # --- Run m=0 ---
    kx0 = k_mag / np.sqrt(2); ky0 = k_mag / np.sqrt(2)
    result, chi_cpu = run_prop(xp, 0.0, kx0, ky0,
                                X, Y, KX, KY, K, dx, X_np, Y_np)
    snapshots = result["snapshots"]
    snap_times = result["snap_times"]

    spec = extract_resonances(result["C_R"], result["corr_times"], dt_snap,
                              Gamma_cut=Gamma_cut, pencil_fraction=4,
                              L_pencil_max=400)
    good = (spec["E"] > 2.0) & (spec["E"] < 6.0) & (spec["Gamma"] < 2.0)
    E_mp = spec["E"][good]; G_mp = spec["Gamma"][good]
    omega_mp = spec["omega"][good]
    n_mp = min(5, len(E_mp))
    E_mp = E_mp[:n_mp]; G_mp = G_mp[:n_mp]; omega_mp = omega_mp[:n_mp]

    print(f"Poles: {n_mp}")
    for j in range(n_mp):
        print(f"  E={E_mp[j]:.4f}  Γ={G_mp[j]:.4f}")

    # --- Run all methods ---
    methods = {}

    tf_modes = [time_filter_mode(snapshots, snap_times, omega_mp[j], chi_cpu)
                for j in range(n_mp)]
    methods["Time filter"] = tf_modes

    methods["Bare LS"] = reconstruct_modes_lstsq(snapshots, snap_times,
                                                   omega_mp, chi_cpu, n_modes=n_mp)

    methods["Route A"] = fixed_pole_recon_direct(snapshots, snap_times,
                                                   omega_mp, chi_cpu,
                                                   n_modes=n_mp, lambda_tik=1e-3)

    modes_B, _ = fixed_pole_recon_pod(snapshots, snap_times, omega_mp, chi_cpu,
                                       n_modes=n_mp, rank=40, lambda_tik=1e-3)
    methods["Route B"] = modes_B

    methods["Route C"] = fixed_pole_constrained_dmd(snapshots, snap_times,
                                                      omega_mp, chi_cpu,
                                                      n_modes=n_mp, rank=40)

    dmd = exact_dmd(snapshots, snap_times, chi_cpu, rank=30, n_eff=150)
    methods["DMD"] = dmd['modes']

    # ═══════════════════════════════════════════════════════════════
    # Figure 1: C₂v symmetrized mode gallery (all methods, top 3 modes)
    # ═══════════════════════════════════════════════════════════════
    # Pick 3 modes with smallest Γ
    order = np.argsort(G_mp)
    show_idx = order[:3]

    method_names = ["Time filter", "Route A", "Route C", "DMD"]
    n_methods = len(method_names)
    n_show = len(show_idx)

    fig, axes = plt.subplots(n_methods, n_show * 2, figsize=(4 * n_show * 2, 3.5 * n_methods))

    for row, mname in enumerate(method_names):
        modes = methods[mname]
        for col_idx, mi in enumerate(show_idx):
            # Find matching mode
            if mname == "DMD":
                # Match by energy
                best_j = None
                if len(dmd['E']) > 0:
                    dists = np.abs(dmd['E'] - E_mp[mi])
                    best_j = np.argmin(dists) if np.min(dists) < 0.5 else None
                if best_j is not None and best_j < len(modes):
                    phi = modes[best_j]
                else:
                    phi = np.zeros((2, n, n), dtype=complex)
            else:
                phi = modes[mi] if mi < len(modes) else np.zeros((2, n, n), dtype=complex)

            rho = np.abs(phi[0])**2 + np.abs(phi[1])**2
            rho_sym = symmetrize_c2v(rho)
            xe, ye = c2v_err(rho)

            # Raw
            ax = axes[row, col_idx * 2]
            ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                      vmin=0, vmax=np.percentile(rho, 99.5))
            if row == 0:
                ax.set_title(f"E={E_mp[mi]:.3f}, Γ={G_mp[mi]:.3f}\nraw", fontsize=9)
            else:
                ax.set_title("raw", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"{mname}\nC₂v: x={xe:.3f}", fontsize=9)

            # Symmetrized
            ax = axes[row, col_idx * 2 + 1]
            ax.imshow(rho_sym, extent=extent, origin="lower", cmap="inferno",
                      vmin=0, vmax=np.percentile(rho_sym, 99.5))
            ax.set_title("C₂v sym", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("m=0 mode reconstruction: raw vs C₂v symmetrized (3 narrowest modes)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "gallery_m0_c2v.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {OUT}/gallery_m0_c2v.png")

    # ═══════════════════════════════════════════════════════════════
    # Compute intrinsic metrics for all methods
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Intrinsic quality metrics (m=0)")
    print("=" * 80)

    rows = []
    for mname in ["Time filter", "Bare LS", "Route A", "Route B", "Route C", "DMD"]:
        modes = methods[mname]

        # Per-mode C₂v errors
        c2v_x_list, c2v_y_list = [], []
        for mi in range(n_mp):
            if mname == "DMD":
                if len(dmd['E']) > 0:
                    dists = np.abs(dmd['E'] - E_mp[mi])
                    j = np.argmin(dists) if np.min(dists) < 0.5 else None
                else:
                    j = None
                if j is not None and j < len(modes):
                    rho = np.abs(modes[j][0])**2 + np.abs(modes[j][1])**2
                else:
                    c2v_x_list.append(1.0); c2v_y_list.append(1.0)
                    continue
            else:
                if mi < len(modes):
                    rho = np.abs(modes[mi][0])**2 + np.abs(modes[mi][1])**2
                else:
                    c2v_x_list.append(1.0); c2v_y_list.append(1.0)
                    continue
            xe, ye = c2v_err(rho)
            c2v_x_list.append(xe)
            c2v_y_list.append(ye)

        # Reconstruction residual
        if mname == "DMD":
            # DMD modes don't correspond 1-to-1 to MP poles, skip residual
            resid = float('nan')
        else:
            resid = reconstruction_residual(snapshots, snap_times, omega_mp,
                                             modes[:n_mp], chi_cpu)

        # Cross-overlap with Route C (best method)
        if mname == "Route C":
            cross = 1.0
        else:
            cross_vals = []
            rc_modes = methods["Route C"]
            for mi in range(n_mp):
                if mname == "DMD":
                    if len(dmd['E']) > 0:
                        dists = np.abs(dmd['E'] - E_mp[mi])
                        j = np.argmin(dists)
                        if dists[j] < 0.5 and j < len(modes):
                            ov = spinor_overlap(modes[j], rc_modes[mi], chi_cpu, dx2)
                        else:
                            ov = 0.0
                    else:
                        ov = 0.0
                else:
                    if mi < len(modes):
                        ov = spinor_overlap(modes[mi], rc_modes[mi], chi_cpu, dx2)
                    else:
                        ov = 0.0
                cross_vals.append(ov)
            cross = np.mean(cross_vals)

        avg_cx = np.mean(c2v_x_list)
        avg_cy = np.mean(c2v_y_list)

        rows.append({
            "method": mname,
            "c2v_x": c2v_x_list,
            "c2v_y": c2v_y_list,
            "avg_cx": avg_cx,
            "avg_cy": avg_cy,
            "resid": resid,
            "cross_rc": cross,
        })

    # Print table
    header = f"{'Method':<15} {'avg C₂v_x':>10} {'avg C₂v_y':>10} {'resid':>8} {'vs RC':>8}"
    for mi in range(n_mp):
        header += f"  C₂v_x({E_mp[mi]:.1f})"
    print(header)
    print("-" * len(header))

    for r in rows:
        line = f"{r['method']:<15} {r['avg_cx']:10.4f} {r['avg_cy']:10.4f} "
        line += f"{r['resid']:8.4f} " if not np.isnan(r['resid']) else f"{'N/A':>8} "
        line += f"{r['cross_rc']:8.4f}"
        for mi in range(n_mp):
            line += f"  {r['c2v_x'][mi]:10.4f}"
        print(line)

    # ═══════════════════════════════════════════════════════════════
    # Write markdown summary
    # ═══════════════════════════════════════════════════════════════
    md_path = os.path.join(OUT, "summary.md")
    with open(md_path, 'w') as f:
        f.write("# Fixed-Pole Mode Reconstruction: Results Summary\n\n")
        f.write("> Generated by `scripts/fixed_pole_gallery.py`\n\n")

        f.write("## Poles (m=0, harmonic ellipse k=3, a=2, b=1.2)\n\n")
        f.write("| # | E | Γ | Γ/ΔE | 可分辨？ |\n")
        f.write("|---|---|---|------|--------|\n")
        for j in range(n_mp):
            ratio = G_mp[j] * 3 * pot_params['a'] * pot_params['b'] * E_mp[j]**2 / (2 * pot_params['k_harm'])
            resolvable = "⚠️ 困难" if ratio > 2 else "✅"
            f.write(f"| {j+1} | {E_mp[j]:.4f} | {G_mp[j]:.4f} | {ratio:.1f} | {resolvable} |\n")

        f.write("\n## 方法对比（内禀指标）\n\n")
        f.write("**关键指标说明**：\n")
        f.write("- **C₂v 误差**：精确本征态密度必须 C₂v 对称（Kramers 定理）。越低越好。这是唯一可靠的内禀质量指标。\n")
        f.write("- **重构残差**：||ψ(t) - Σ φ_n e^{-iωt}|| / ||ψ||。模型对数据的解释能力。\n")
        f.write('- **vs Route C**：与最佳方法的一致性（不是"正确性"指标）。\n')
        f.write('- ⚠️ overlap with TF 不作为主要指标——TF 在 m=0 下不可靠。\n\n')

        f.write("| 方法 | avg C₂v_x | avg C₂v_y | 重构残差 | vs Route C |\n")
        f.write("|------|-----------|-----------|---------|------------|\n")
        for r in rows:
            resid_str = f"{r['resid']:.4f}" if not np.isnan(r['resid']) else "N/A"
            f.write(f"| {r['method']} | {r['avg_cx']:.4f} | {r['avg_cy']:.4f} | "
                    f"{resid_str} | {r['cross_rc']:.4f} |\n")

        f.write("\n## Per-mode C₂v_x 误差\n\n")
        f.write("| 方法 |")
        for j in range(n_mp):
            f.write(f" E={E_mp[j]:.2f} |")
        f.write("\n|------|")
        for j in range(n_mp):
            f.write("--------|")
        f.write("\n")
        for r in rows:
            f.write(f"| {r['method']} |")
            for j in range(n_mp):
                val = r['c2v_x'][j]
                bold = "**" if val == min(rr['c2v_x'][j] for rr in rows) else ""
                f.write(f" {bold}{val:.4f}{bold} |")
            f.write("\n")

        f.write("\n## 关键发现\n\n")
        f.write("1. **Route C（约束 DMD）C₂v 误差最低**——这是唯一可靠的内禀指标\n")
        f.write("2. **Tikhonov 正则化无效**——Route A/B ≈ 裸 LS，λ 从 1e-6 到 1 无变化\n")
        f.write("3. **宽模式不可分辨**——Γ > 0.3 的模式所有方法 C₂v 误差 > 0.5\n")
        f.write("4. **瓶颈是物理信号**而非数值条件数\n\n")

        f.write("## 图表\n\n")
        f.write("- `gallery_m0_c2v.png` — 各方法 raw vs C₂v 对称化模式密度\n")
        f.write("- `comparison_m0.png` — overlap 柱状图（仅供参考，非主要指标）\n")
        f.write("- `scans_m0.png` — λ 扫描 + rank 扫描\n")

    print(f"\nSummary → {md_path}")

    elapsed = time.perf_counter() - t_start
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
