# LDOS and resonances of 2D Dirac electrons from wavepacket dynamics

Massless Dirac electrons (one valley of graphene) in a smooth electrostatic potential,

$$H = \hbar v\,\boldsymbol\sigma\cdot\mathbf p + V(\mathbf r) - iW(\mathbf r),$$

with an absorbing potential $W$ at the edge of a periodic box.

## LDOS from a wavepacket frame

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

## Resonances from two launches

Once the direct, non-resonant part has left, the wavefunction of any launch $\psi_s$ is a sum of
decaying resonances with complex energies $z_n = E_n - i\Gamma_n/2$,

$$\psi_s(\mathbf r,t) = \sum_n P_n\psi_s(\mathbf r)\, e^{-iz_n t/\hbar},$$

where $P_n$ projects onto resonance $n$.  Since $H^T = \sigma_y H \sigma_y$, every resonance is a
doublet, and its left eigenvectors are $\sigma_y$ times its right ones (transposed, not conjugated).
Two launches with independent parts in each doublet therefore fix $P_n$, and with it the local weight
$\rho_n(\mathbf r) = \mathrm{tr}\,\langle\mathbf r|P_n|\mathbf r\rangle$, the share of resonance $n$
in the LDOS, $-\mathrm{Im}[\rho_n(\mathbf r)/(E-z_n)]/\pi$.  `resonances.py` works in four steps:

1. **band** — a band-pass filter keeps one energy window, and the signal is resampled coarsely;
   each exponential keeps its $z_n$ and its weight is multiplied by a known gain;
2. **poles** — the $z_n$ are the eigenvalues of one time step of the filtered wavefunctions
   (a matrix pencil over all pixels, sublattices and launches at once);
3. **weights** — $P_n\psi_s(\mathbf r)$ by least squares with the poles held fixed;
4. **density** — $\rho_n(\mathbf r)$ from the weights of the two launches.

## Files

- `dirac.py` — grid, potential, absorber, split-operator propagator (PyTorch, GPU if available)
- `ldos.py` — wavepacket frame, return signal, LDOS
- `resonances.py` — band filter, poles, weights, local weights of the resonances
- `run_formp.py` — return signal of the frame along the two axes of the Form-P hill
  $V = 146 - 0.0288\,x^2 - 0.0677\,y^2$ meV ($x, y$ in nm)
- `plot_formp.py` — LDOS line scans compared with tight binding
- `run_resonances.py` — wavefunctions of two launches on the hill
- `plot_resonances.py` — resonance energies and their maps
- `reference/pybinding_formp.npz` — tight-binding LDOS of the same hill (pybinding, KPM):
  per A site per eV, Jackson kernel of about 2 meV, energies `E` (meV), positions `x`, `y` (nm)
- `reference/diag_formp_resonances.npz` — complex energies `z` (meV) of the resonances with
  $|E| < 110$ meV, from a direct diagonalization of the same discretized Hamiltonian

## Usage

```bash
python run_formp.py          # writes formp_signal.npz
python plot_formp.py         # writes formp_ldos.png
python run_resonances.py     # writes formp_launches.npz
python plot_resonances.py    # writes formp_resonances.npz, formp_resonances.png
```

Units are nm, meV and fs throughout.  Requires numpy, scipy, matplotlib and PyTorch.
