"""Wavepacket-frame LDOS against the tight-binding (pybinding KPM) line scans of the same hill.

The reference is the LDOS per A site per eV of a graphene lattice (a = 0.4919 nm, t = 1.85 eV,
hbar v_F = 788.1 meV nm), broadened by a Jackson kernel of about 2 meV.  One unit cell holds
one A and one B site, and both valleys contribute, so the single-valley Dirac LDOS (summed
over sublattices) times the cell area is the same quantity.
"""

import numpy as np
import matplotlib.pyplot as plt

from dirac import form_p
from ldos import ldos

A_CELL = np.sqrt(3) / 2 * 0.4919**2     # nm^2
TO_REFERENCE = A_CELL * 1000            # 1/(nm^2 meV)  ->  1/(site eV)

E = np.arange(-100.0, 100.001, 0.25)    # meV
sigma_E = 2.0                           # Gaussian broadening (meV)

sig = np.load("formp_signal.npz")
ref = np.load("reference/pybinding_formp.npz")

fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)

for row, a in enumerate(("x", "y")):
    pos = sig[a]
    sim = ldos(sig["t"], sig["C_" + a], E, sigma_E, kernel="gauss")
    sim = sim * TO_REFERENCE

    # reference on the same energies and positions
    r = ref["ldos_" + a]
    r = np.array([np.interp(E, ref["E"], col) for col in r.T]).T
    r = np.array([np.interp(pos, ref[a], line) for line in r])

    # classical turning line E = V along this axis
    V = form_p(pos, 0 * pos) if a == "x" else form_p(0 * pos, pos)

    extent = [pos[0], pos[-1], E[0], E[-1]]
    vmax = np.percentile(r, 99.5)
    panels = [(r, "tight binding (pybinding)"), (sim, "Dirac wavepacket frame")]
    for col, (img, title) in enumerate(panels):
        ax = axes[row, col]
        ax.imshow(img, origin="lower", aspect="auto", extent=extent,
                  cmap="inferno", vmin=0, vmax=vmax)
        ax.plot(pos, V, "w--", lw=0.8)
        ax.set_ylim(E[0], E[-1])
        ax.set_title(f"{title}, {a} axis")
        ax.set_xlabel(f"{a} (nm)")
        ax.set_ylabel("E (meV)")

    c = np.argmin(abs(pos))
    ax = axes[row, 2]
    ax.plot(E, r[:, c], "k", lw=1.2, label="tight binding")
    ax.plot(E, sim[:, c], "C1", lw=1.0, label="Dirac wavepackets")
    ax.set_title(f"centre of the hill, {a} axis")
    ax.set_xlabel("E (meV)")
    ax.set_ylabel("LDOS (1 / site eV)")
    ax.legend(frameon=False)

fig.suptitle("Form-P potential hill: 1D LDOS along the two axes")
fig.savefig("formp_ldos.png", dpi=150)
