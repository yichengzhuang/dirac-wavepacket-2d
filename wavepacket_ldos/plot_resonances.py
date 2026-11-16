"""Resonances of the Form-P hill from two wavepacket launches.

Reads formp_launches.npz.  Writes formp_resonances.npz (complex energies z in meV, local
weights rho in 1 / nm^2) and formp_resonances.png.  The reference energies come from a
direct diagonalization of the same discretized Hamiltonian (same grid, box and absorber).
"""

import numpy as np
import matplotlib.pyplot as plt

from dirac import form_p
from resonances import band, poles, weights, density

E_lo, E_hi = -110.0, 110.0      # energy window (meV)
edge = 60.0                     # filter transition (meV)

d = np.load("formp_launches.npz")
ref = np.load("reference/diag_formp_resonances.npz")
t, x, y = d["t"], d["x"], d["y"]
psi0, psi = d["psi0"], d["psi"]
dx = x[1] - x[0]

# every launch, sublattice and pixel is one channel
f = np.moveaxis(psi, 1, 0).reshape(len(t), -1)
tb, fb, gain = band(t, f, E_lo, E_hi, edge)

# energies within `edge` of the window pass the filter in part, so they join the fit
z = poles(tb, fb, E_lo - edge, E_hi + edge)
F = weights(tb, fb, z, gain).reshape((len(z),) + psi0.shape)

inside = (z.real > E_lo) & (z.real < E_hi)
z = z[inside]
rho = np.array([density(Fn, psi0, dx) for Fn in F[inside]])
np.savez("formp_resonances.npz", z=z, x=x, y=y, rho=rho)

fig = plt.figure(figsize=(17, 8.5), constrained_layout=True)
grid = fig.add_gridspec(5, 8)
extent = [x[0], x[-1], y[0], y[-1]]

ax = fig.add_subplot(grid[:3, :3])
ax.plot(ref["z"].real, -2 * ref["z"].imag, "o", mfc="none", mec="k", ms=11,
        label="direct diagonalization")
ax.plot(z.real, -2 * z.imag, "x", color="C0", mew=1.6, label="two launches")
ax.set_xlim(E_lo, E_hi)
ax.set_ylim(30, 0)
ax.set_xlabel("E (meV)")
ax.set_ylabel("Γ (meV)")
ax.set_title("complex energies E − iΓ/2")
ax.legend(frameon=False, loc="lower left")

# the launches on the potential hill: |psi0|^2 and the direction of their momentum
ax = fig.add_subplot(grid[3:, :3])
V = form_p(*np.meshgrid(x, y))
im = ax.imshow(V, origin="lower", cmap="Greys", extent=extent)
fig.colorbar(im, ax=ax, label="V (meV)")
density0 = (abs(psi0)**2).sum((0, 1))
ax.contour(x, y, density0, levels=density0.max() * np.array([0.1, 0.5]),
           colors="C0", linewidths=0.8)
x0, y0, _, angle = d["launches"].T
phi = np.deg2rad(angle)
ax.quiver(x0, y0, np.cos(phi), np.sin(phi), color="C0", scale=12)
ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_title("launches on the hill V(r)")

# the first 25 resonances of the window
for k, (zn, m) in enumerate(zip(z[:25], rho.real)):
    ax = fig.add_subplot(grid[k // 5, 3 + k % 5])
    lim = abs(m).max()
    ax.imshow(m, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"E = {zn.real:.1f}, Γ = {-2 * zn.imag:.1f} meV", fontsize=8)

fig.suptitle("Form-P hill: resonances from two launches; maps show Re ρ_n(r), "
             f"|x| < {x[-1]:.0f} nm, |y| < {y[-1]:.0f} nm")
fig.savefig("formp_resonances.png", dpi=130)
