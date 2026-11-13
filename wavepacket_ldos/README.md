# LDOS of 2D Dirac electrons from wavepacket dynamics

Massless Dirac electrons (one valley of graphene) in a smooth electrostatic potential,

$$H = \hbar v\,\boldsymbol\sigma\cdot\mathbf p + V(\mathbf r) - iW(\mathbf r),$$

with an absorbing potential $W$ at the edge of a periodic box.

Gaussian wavepackets $|\mathbf r_0,\mathbf k_0,s\rangle$, taken with both band spinors $s=\pm1$
and spread uniformly over a region of phase space, resolve the identity:
$\sum_j w\,|j\rangle\langle j| = 1$, with $w$ the phase-space area per point divided by $(2\pi)^2$.
Propagating every member of this frame gives the local return amplitude

$$C(\mathbf r,t) = w\sum_j \sum_{\alpha=A,B}\psi_{j\alpha}(\mathbf r,t)\,\psi_{j\alpha}(\mathbf r,0)^*
= \mathrm{tr}\,\langle\mathbf r|e^{-iHt/\hbar}|\mathbf r\rangle ,$$

and the local density of states follows by a windowed Fourier transform,

$$\rho(\mathbf r,E) = \frac{1}{\pi\hbar}\,\mathrm{Re}\int_0^T dt\; e^{iEt/\hbar} f(t)\, C(\mathbf r,t).$$

$f(t)=e^{-\eta t/\hbar}$ gives $-\mathrm{Im}\,G(\mathbf r,\mathbf r;E+i\eta)/\pi$ (Lorentzian broadening);
$f(t)=e^{-(\sigma t/\hbar)^2/2}$ gives a Gaussian broadening $\sigma$.

## Files

- `dirac.py` — grid, potential, absorber, split-operator propagator (PyTorch, GPU if available)
- `ldos.py` — wavepacket frame, return signal, LDOS
- `run_formp.py` — return signal along the two axes of the Form-P hill
  $V = 146 - 0.0288\,x^2 - 0.0677\,y^2$ meV ($x, y$ in nm)
- `plot_formp.py` — LDOS line scans compared with tight binding
- `reference/pybinding_formp.npz` — tight-binding LDOS of the same hill (pybinding, KPM):
  per A site per eV, Jackson kernel of about 2 meV, energies `E` (meV), positions `x`, `y` (nm)

## Usage

```bash
python run_formp.py      # writes formp_signal.npz
python plot_formp.py     # writes formp_ldos.png
```

Units are nm, meV and fs throughout.  Requires numpy, scipy, matplotlib and PyTorch.
