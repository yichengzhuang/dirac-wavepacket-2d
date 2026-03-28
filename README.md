# 2D Dirac Billiard: LDOS and Mode Extraction

Open-boundary simulation of 2D Dirac fermions in confining potentials.
Extracts the local density of states (LDOS) and quasi-bound resonance
modes from wavepacket propagation data.

## Installation

```bash
git clone https://github.com/yichengzhuang/dirac-wavepacket-2d.git
cd dirac-wavepacket-2d
pip install -r requirements.txt
```

For GPU acceleration, install [PyTorch](https://pytorch.org/) with CUDA support.

## Quick start

```bash
python run.py
```

This runs the default configuration (massless Dirac, harmonic ellipse,
256x256 grid) and writes results to `output/`.

See [`examples/`](examples/) for more configurations.

## Pipeline

```
                     ┌─ snapshots ψ(r,t) ──→ Fourier transform ──→ LDOS
Propagation ──→     │
                     └─ C_R(t) ──→ Matrix Pencil ──→ poles ω_n ──→ modes φ_n(r)
```

1. **Propagation**: Split-operator method for the 2D Dirac equation
   with complex absorbing potential (CAP) at the boundary.
2. **LDOS**: Real-frequency Fourier transform of snapshots, averaged
   over multiple propagation directions.
3. **Poles**: Matrix pencil (harmonic inversion) extracts complex
   resonance frequencies from the scalar autocorrelation.
4. **Modes**: Two methods reconstruct spatial quasi-eigenstates from
   snapshots + known poles:
   - **Time filtering (TF)**: Per-pixel Fourier filter at each pole frequency.
     Fast (< 1s), treats each pixel independently.
   - **Fixed-pole SVD**: Builds a reduced dynamics operator via SVD of the
     snapshot matrix, then searches its null space at each known pole.
     Couples all spatial points for noise suppression (~40s, dominated by SVD).

## Output

| File | Description |
|------|-------------|
| `ldos.npy` | LDOS(E, r), shape (n_E, ny, nx) |
| `E_values.npy` | Energy grid |
| `C_R.npy` | Autocorrelation C_R(t), complex |
| `corr_times.npy` | Time values for C_R |
| `poles.npy` | Resonance poles (E_n, Gamma_n) |
| `mode_tf_j.npy` | Mode from time filtering, shape (2, ny, nx) |
| `mode_svd_j.npy` | Mode from fixed-pole SVD, shape (2, ny, nx) |
| `autocorrelation.png` | C_R(t) magnitude and Re/Im |
| `ldos_spectrum.png` | Integrated LDOS vs energy |
| `ldos_and_modes.png` | Spatial maps: LDOS / TF mode / SVD mode |

## Code structure

```
├── run.py                  pipeline driver
├── spectral.py             matrix pencil, LDOS, time filtering
├── potentials_2d.py        confining potentials and analysis masks
├── accel/
│   ├── __init__.py          backend auto-detection (CUDA > MPS > CPU)
│   ├── backend.py           abstract backend interface
│   ├── backend_numpy.py     CPU backend (NumPy/SciPy)
│   ├── backend_torch.py     Apple GPU backend (MPS)
│   ├── backend_torch_cuda.py  NVIDIA GPU backend (CUDA)
│   └── dynamics_accel.py    grid, wavepacket, split-operator propagation
├── examples/                example run scripts
├── requirements.txt
└── LICENSE
```

## Parameters

Run `python run.py --help` for a full list. Key options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backend` | auto | `torch_cuda`, `torch_mps`, `numpy` |
| `--n` | 256 | Spatial grid size |
| `--mass` | 0.0 | Dirac mass (0 = massless, Klein tunnelling) |
| `--potential` | harmonic_ellipse | Potential geometry |
| `--n_prop` | 8 | Propagation directions for LDOS averaging |
| `--n_steps` | 15000 | Time steps (T = n_steps * dt) |

Available potentials: `harmonic_ellipse`, `harmonic_circle`,
`soft_ellipse`, `soft_circle`, `soft_stadium`,
`hard_ellipse`, `hard_circle`.

## Units

Natural units: hbar = v_F = 1. All energies and momenta are dimensionless.

## License

[MIT](LICENSE)
