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

## LDOS from a wavepacket frame

[`wavepacket_ldos/`](wavepacket_ldos/) computes the local density of states directly.
A frame of Gaussian wavepackets, with both band spinors and spread uniformly over a region
of phase space, resolves the identity.  Propagating every member gives the local return
amplitude $\mathrm{tr}\,\langle\mathbf r|e^{-iHt/\hbar}|\mathbf r\rangle$, and its windowed
Fourier transform is the LDOS.  The example there is an anisotropic potential hill in
graphene, compared with a tight-binding (pybinding) calculation of the same hill.

```bash
cd wavepacket_ldos
python run_formp.py
python plot_formp.py
```

## Pipeline

```
                     ┌─ snapshots ψ(r,t) ──→ Fourier transform ──→ spectral map
Propagation ──→     │
                     └─ C_R(t) ──→ Matrix Pencil ──→ poles ω_n ──→ modes φ_n(r)
```

1. **Propagation**: Split-operator method for the 2D Dirac equation
   with complex absorbing potential (CAP) at the boundary.
2. **Spectral map**: Real-frequency Fourier transform of snapshots, averaged
   over multiple propagation directions.  The LDOS itself is computed in
   [`wavepacket_ldos/`](wavepacket_ldos/).
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
| `ldos.npy` | Spectral map \|ψ(r, E)\|², shape (n_E, ny, nx) |
| `E_values.npy` | Energy grid |
| `C_R.npy` | Autocorrelation C_R(t), complex |
| `corr_times.npy` | Time values for C_R |
| `poles.npy` | Resonance poles (E_n, Gamma_n) |
| `mode_tf_j.npy` | Mode from time filtering, shape (2, ny, nx) |
| `mode_svd_j.npy` | Mode from fixed-pole SVD, shape (2, ny, nx) |
| `autocorrelation.png` | C_R(t) magnitude and Re/Im |
| `ldos_spectrum.png` | Integrated LDOS vs energy |
| `ldos_and_modes.png` | Spatial maps: LDOS / TF mode / SVD mode |

## Performance

### GPU acceleration

The propagation kernel (split-operator + FFT) runs on GPU via PyTorch
when available.  The backend is selected automatically at startup:

| Backend | Hardware | Typical speedup | Selection |
|---------|----------|----------------|-----------|
| `torch_cuda` | NVIDIA GPU (A100, V100, ...) | **20×** | Auto if CUDA available |
| `torch_mps` | Apple Silicon (M1/M2/M3) | ~5× | Auto if MPS available |
| `numpy` | CPU (OpenBLAS/MKL) | 1× (baseline) | Fallback |

Override with `--backend torch_cuda` or `--backend numpy`.

### Vectorised spectral transform

The snapshot Fourier transform uses matrix multiplication instead of
Python loops:

```
phases @ snap_A   →   (n_E, Ns) @ (Ns, N_sp)   →   (n_E, N_sp)
```

This replaces `n_E × Ns` Python iterations with a single BLAS
`gemm` call.  Verified to match the loop version to machine
precision (relative error < 3×10⁻¹⁵).  Speedup: ~30× on CPU.

### Typical timings (n=256, T=150)

| Operation | CPU (8 cores) | GPU (A100) |
|-----------|:------------:|:----------:|
| Single propagation (15k steps) | 120 s | 6 s |
| Spectral map (300 energies) | 0.6 s | — |
| Matrix pencil pole extraction | 0.1 s | — |
| Fixed-pole SVD mode extraction | 40 s | — |
| **Full pipeline (8 directions)** | **~17 min** | **~1 min** |

## Code structure

```
├── run.py                  pipeline driver (quick start entry point)
├── spectral.py             matrix pencil, LDOS, time filtering
├── potentials_2d.py        confining potentials and analysis masks
├── wavepacket_ldos/        LDOS from a frame of Gaussian wavepackets
├── accel/                  GPU/CPU backend abstraction
│   ├── __init__.py          auto-detection (CUDA > MPS > CPU)
│   ├── backend.py           abstract interface
│   ├── backend_numpy.py     CPU backend (NumPy/SciPy FFT)
│   ├── backend_torch.py     Apple GPU backend (PyTorch MPS)
│   ├── backend_torch_cuda.py  NVIDIA GPU backend (PyTorch CUDA)
│   └── dynamics_accel.py    grid, wavepacket, CAP, split-operator propagation
│
├── core/                   full pipeline with visualization and units
│   ├── main.py              main driver (parameter configuration)
│   ├── dynamics.py          CPU propagation module
│   ├── spectral.py          spectral analysis
│   ├── visualization.py     plotting and animation
│   ├── units.py             natural ↔ physical unit conversion
│   └── accel/               GPU-accelerated propagation
│
├── experiments/            all exploratory and validation scripts
│   ├── dmd_*.py             Dynamic Mode Decomposition attempts
│   ├── fdm_*.py             Filter Diagonalization Method
│   ├── fixed_pole_*.py      fixed-pole SVD mode reconstruction
│   ├── ldos_*.py            LDOS studies (shapes, averaging, scars)
│   ├── study_*.py           parameter convergence studies
│   ├── cap_*.py             CAP convergence and validation
│   ├── kramers_test.py      Kramers density identity verification
│   ├── exact_diag_test.py   exact diagonalization benchmark
│   ├── method_comparison.py systematic comparison of all methods
│   └── submit_*.sh          SLURM submission scripts
│
├── docs/                   method notes and research summary
│   ├── pipeline.pdf         algorithm description (2 pages)
│   └── summary.pdf          physics background and storyline
├── examples/               example run scripts
├── requirements.txt
└── LICENSE
```

## Parameters

Run `python run.py --help` for a full list. Key options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backend` | auto | `torch_cuda`, `torch_mps`, `numpy` |
| `--n` | 256 | Spatial grid size (power of 2 for FFT) |
| `--mass` | 0.0 | Dirac mass (0 = massless, Klein tunnelling) |
| `--potential` | harmonic_ellipse | Potential geometry |
| `--n_prop` | 8 | Propagation directions for LDOS averaging |
| `--n_steps` | 15000 | Time steps (T = n_steps × dt) |
| `--save_every` | 10 | Snapshot cadence (Nyquist: E_max < π/(save_every × dt)) |
| `--n_E` | 300 | Number of energy points for LDOS |
| `--n_modes` | 5 | Number of modes to extract |

Available potentials: `harmonic_ellipse`, `harmonic_circle`,
`soft_ellipse`, `soft_circle`, `soft_stadium`,
`hard_ellipse`, `hard_circle`.

## Units

Natural units: ℏ = v_F = 1. All energies and momenta are dimensionless.
For graphene quantum dots: the default parameters (a=2, k_harm=3)
correspond to a ~50 nm dot with E ~ 63 meV, Γ ~ 16 meV.
`wavepacket_ldos/` works in nm, meV and fs.

## License

[MIT](LICENSE)
