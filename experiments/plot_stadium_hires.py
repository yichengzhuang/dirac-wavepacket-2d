#!/usr/bin/env python3
"""Post-process: generate figures from saved ldos_hires.npy (1.2GB)."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import find_peaks

OUT = "figures/ldos_stadium_scar"
ldos = np.load(f"{OUT}/ldos_hires.npy")
E = np.load(f"{OUT}/E_values_hires.npy")
n = ldos.shape[1]; L = 10.0; half_L = L/2
extent = [-half_L, half_L, -half_L, half_L]
x = np.linspace(-half_L, half_L, n, endpoint=False)
print(f"Loaded: {ldos.shape}, E=[{E[0]:.1f}, {E[-1]:.1f}], n={n}")

def fft_flip(a, ax):
    return np.roll(np.flip(a, axis=ax), 1, axis=ax)

def sym_c2v(r):
    rx = fft_flip(r, 1); ry = fft_flip(r, 0)
    return 0.25*(r + rx + ry + fft_flip(rx, 0))

# 1. Gallery 10×10
print("Gallery...")
n_show = 100
idx = np.linspace(0, len(E)-1, n_show, dtype=int)
fig, axes = plt.subplots(10, 10, figsize=(35, 35))
for k, i in enumerate(idx):
    ax = axes[k//10, k%10]
    rho = sym_c2v(ldos[i])
    ax.imshow(rho, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=rho.max())
    ax.set_title(f"E={E[i]:.1f}", fontsize=6)
    ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Stadium LDOS E∈[0.5,30] — n=512, k_mag×8, M=384", fontsize=14)
plt.tight_layout()
fig.savefig(f"{OUT}/gallery_hires.png", dpi=120, bbox_inches="tight")
plt.close(); print("  gallery_hires.png")

# 2. Spectrum
print("Spectrum...")
spec = ldos.sum(axis=(1,2))
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(E, spec/spec.max(), "b-", lw=1)
ax.set_xlabel("E"); ax.set_ylabel("LDOS (norm.)")
ax.set_title("Stadium LDOS spectrum (n=512, high-res)")
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/spectrum_hires.png", dpi=150, bbox_inches="tight")
plt.close(); print("  spectrum_hires.png")

# 3. 1D line cuts
print("Linecuts...")
ldos_sym = np.array([sym_c2v(ldos[j]) for j in range(len(E))])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.pcolormesh(x, E, ldos_sym[:, n//2, :], cmap="inferno", shading="auto")
ax1.set_xlabel("x"); ax1.set_ylabel("E"); ax1.set_title("LDOS(E,x) at y=0")
ax1.set_xlim(-2.5, 2.5)
ax2.pcolormesh(x, E, ldos_sym[:, :, n//2], cmap="inferno", shading="auto")
ax2.set_xlabel("y"); ax2.set_ylabel("E"); ax2.set_title("LDOS(E,y) at x=0")
ax2.set_xlim(-2, 2)
fig.suptitle("Stadium 1D LDOS (n=512, high-res)", fontsize=12)
plt.tight_layout()
fig.savefig(f"{OUT}/linecuts_hires.png", dpi=150, bbox_inches="tight")
plt.close(); print("  linecuts_hires.png")

# 4. Scar candidates: 16 strongest peaks at E>5
print("Scars...")
spec_high = spec.copy(); spec_high[E < 5] = 0
peaks_idx, _ = find_peaks(spec_high, distance=3)
if len(peaks_idx) > 0:
    peaks_sorted = peaks_idx[np.argsort(spec[peaks_idx])[::-1]]
    n_scar = min(16, len(peaks_sorted))
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    axes = axes.ravel()
    for j in range(n_scar):
        ax = axes[j]
        rho = sym_c2v(ldos[peaks_sorted[j]])
        im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                      vmin=0, vmax=rho.max())
        ax.set_title(f"E={E[peaks_sorted[j]]:.2f}", fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    for j in range(n_scar, 16):
        axes[j].axis("off")
    fig.suptitle("Scar candidates (E>5, 16 strongest, n=512)", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{OUT}/scar_hires.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  scar_hires.png")

# 5. Selected energies
print("Selected energies...")
E_select = [3, 5, 8, 10, 13, 15, 18, 20, 23, 25]
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.ravel()
for j, E_t in enumerate(E_select):
    idx_e = np.argmin(np.abs(E - E_t))
    ax = axes[j]
    rho = sym_c2v(ldos[idx_e])
    im = ax.imshow(rho, extent=extent, origin="lower", cmap="inferno",
                  vmin=0, vmax=rho.max())
    ax.set_title(f"E={E[idx_e]:.1f}", fontsize=11)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Stadium LDOS at selected energies (n=512)", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT}/selected_energies_hires.png", dpi=150, bbox_inches="tight")
plt.close(); print("  selected_energies_hires.png")

print("Done!")
