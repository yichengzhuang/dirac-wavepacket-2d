"""
PyTorch MPS (Apple GPU) backend.

All arrays live on the MPS device in complex64 (float32 components).
GPU→CPU transfers happen only when explicitly requested via to_numpy().
"""

import numpy as np
import torch

from .backend import ArrayBackend


class TorchMPSBackend(ArrayBackend):

    name = "torch_mps"

    def __init__(self, device="mps", dtype=torch.complex64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.real_dtype = torch.float32

    # ── array creation / conversion ──────────────────────────

    def array(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        if isinstance(data, (int, float)):
            return torch.tensor(data, dtype=self.real_dtype, device=self.device)
        if isinstance(data, complex):
            return torch.tensor(data, dtype=self.dtype, device=self.device)
        arr = np.asarray(data)
        if np.issubdtype(arr.dtype, np.complexfloating):
            return torch.from_numpy(arr.astype(np.complex64)).to(self.device)
        if np.issubdtype(arr.dtype, np.floating):
            return torch.from_numpy(arr.astype(np.float32)).to(self.device)
        # Boolean / integer → float32
        return torch.from_numpy(arr.astype(np.float32)).to(self.device)

    def to_numpy(self, arr):
        if isinstance(arr, np.ndarray):
            return arr
        return arr.cpu().numpy()

    def zeros(self, shape, dtype=None):
        dt = self.dtype if dtype is None else dtype
        return torch.zeros(shape, dtype=dt, device=self.device)

    def zeros_like(self, x):
        return torch.zeros_like(x)

    def empty_like(self, x):
        return torch.empty_like(x)

    def copy(self, x):
        return x.clone()

    def stack(self, arrays, axis=0):
        return torch.stack(list(arrays), dim=axis)

    # ── math (element-wise) ──────────────────────────────────

    def exp(self, x):
        if isinstance(x, (int, float, complex)):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        return torch.exp(x)

    def cos(self, x):
        return torch.cos(x)

    def sin(self, x):
        return torch.sin(x)

    def sqrt(self, x):
        return torch.sqrt(x)

    def abs(self, x):
        if x.is_complex():
            # Workaround: torch.abs on MPS complex tensors can return
            # shuffled values in PyTorch ≤ 2.3.  Compute manually.
            return torch.sqrt(x.real**2 + x.imag**2)
        return torch.abs(x)

    def real(self, x):
        return x.real

    def conj(self, x):
        return torch.conj(x)

    def where(self, cond, a, b):
        if isinstance(b, (int, float)):
            b = torch.tensor(b, dtype=a.dtype, device=a.device)
        if isinstance(a, (int, float)):
            a = torch.tensor(a, dtype=b.dtype, device=b.device)
        return torch.where(cond, a, b)

    def clip(self, x, lo, hi):
        return torch.clamp(x, min=lo, max=hi)

    # ── reductions ───────────────────────────────────────────

    def sum(self, x):
        return torch.sum(x)

    def vdot(self, a, b):
        """Conjugate inner product: sum(conj(a) * b)."""
        return torch.sum(torch.conj(a) * b)

    def max(self, x):
        return torch.max(x)

    # ── FFT ──────────────────────────────────────────────────

    def fftn(self, x):
        return torch.fft.fftn(x)

    def ifftn(self, x):
        return torch.fft.ifftn(x)

    def fftfreq(self, n, d):
        freq = torch.fft.fftfreq(n, d=d, device=self.device, dtype=self.real_dtype)
        return 2 * np.pi * freq  # keep as real float32

    # ── grid helpers ─────────────────────────────────────────

    def linspace(self, start, stop, n, endpoint=True):
        if endpoint:
            return torch.linspace(start, stop, n, device=self.device, dtype=self.real_dtype)
        # torch.linspace always includes endpoint; replicate numpy's endpoint=False
        step = (stop - start) / n
        return torch.arange(n, device=self.device, dtype=self.real_dtype) * step + start

    def meshgrid(self, *arrays, indexing="xy"):
        return torch.meshgrid(*arrays, indexing=indexing)

    # ── utilities ────────────────────────────────────────────

    def synchronize(self):
        """Block until all MPS operations complete."""
        torch.mps.synchronize()
