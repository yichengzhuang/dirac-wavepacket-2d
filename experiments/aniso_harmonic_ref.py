#!/usr/bin/env python3
"""
参考对照：二维各向异性谐振子的前 10 个本征态。

势能 V(x,y) = ½(ωx² x² + ωy² y²)，取 ωx=1, ωy=5/3，
等势线为轴比 2:1.2 的椭圆（长轴沿 x）。

解析解：
  ψ_{nx,ny}(x,y) = φ_{nx}(x; ωx) × φ_{ny}(y; ωy)
  E_{nx,ny}      = ωx(nx+½) + ωy(ny+½)

其中 φ_n 为一维谐振子本征函数（Hermite-Gauss）。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial, sqrt, pi

# ---------- 参数 ----------
omega_x = 1.0
omega_y = 5.0 / 3.0  # ωy/ωx = 5/3  →  等势椭圆长轴/短轴 = 2:1.2
N_states = 10        # 按能量排序取前 10 个

# 空间网格
Lx, Ly = 8.0, 6.0   # 长轴方向取更大范围
Nx, Ny = 201, 201
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing="xy")

# ---------- 一维谐振子波函数 ----------
def psi_1d(n, xi, omega):
    """一维谐振子第 n 个本征函数 φ_n(xi; ω)"""
    alpha = sqrt(omega)          # sqrt(mω/ℏ), ℏ=m=1
    Hn = hermite(n)
    norm = (alpha / sqrt(2**n * factorial(n) * sqrt(pi)))
    return norm * Hn(alpha * xi) * np.exp(-0.5 * alpha**2 * xi**2)

# ---------- 枚举量子数并按能量排序 ----------
# 能量上界估计：取足够多的 (nx, ny)
max_n = 12
states = []
for nx in range(max_n):
    for ny in range(max_n):
        E = omega_x * (nx + 0.5) + omega_y * (ny + 0.5)
        states.append((E, nx, ny))
states.sort()
states = states[:N_states]

# ---------- 计算波函数并绘图 ----------
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.ravel()

# 势能等高线参考
V = 0.5 * (omega_x**2 * X**2 + omega_y**2 * Y**2)

for idx, (E, nx, ny) in enumerate(states):
    psi = psi_1d(nx, X, omega_x) * psi_1d(ny, Y, omega_y)
    density = psi**2

    ax = axes[idx]
    # 画密度
    vmax = density.max()
    im = ax.pcolormesh(X, Y, density, cmap="inferno", shading="auto",
                       vmin=0, vmax=vmax, rasterized=True)
    # 叠加势能等高线（画到该态能量处，即经典转向线）
    ax.contour(X, Y, V, levels=[E], colors="cyan", linewidths=0.8,
               linestyles="--")
    ax.set_aspect("equal")
    ax.set_xlim(-Lx, Lx)
    ax.set_ylim(-Ly, Ly)
    ax.set_title(rf"$({nx},{ny})$  $E={E:.1f}$", fontsize=11)
    ax.tick_params(labelsize=7)

fig.suptitle(
    rf"2D anisotropic HO   $\omega_x={omega_x},\;\omega_y={omega_y:.4g}$"
    r"   (equipotential axis ratio 2:1.2)   —   lowest 10 eigenstates $|\psi|^2$",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("figures/sch_ref/aniso_harmonic_ref.png", dpi=150)
print("已保存 figures/sch_ref/aniso_harmonic_ref.png")
print("\n能级列表：")
for i, (E, nx, ny) in enumerate(states):
    deg = sum(1 for e, _, _ in states if abs(e - E) < 1e-10)
    label = f"  (简并 ×{deg})" if deg > 1 else ""
    print(f"  #{i}: (nx={nx}, ny={ny})  E = {E:.2f}{label}")
