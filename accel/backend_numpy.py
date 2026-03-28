"""
NumPy / SciPy CPU backend.

Uses scipy.fft with workers=-1 for multi-threaded FFT (≈1.6x over np.fft).
"""

import numpy as np
import scipy.fft

from .backend import ArrayBackend


class NumpyBackend(ArrayBackend):

    name = "numpy"

    def __init__(self, use_scipy_fft=True, workers=-1):
        self._use_scipy = use_scipy_fft
        self._workers = workers

    # ── array creation / conversion ──────────────────────────

    def array(self, data):
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)

    def to_numpy(self, arr):
        return np.asarray(arr)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def empty_like(self, x):
        return np.empty_like(x)

    def copy(self, x):
        return x.copy()

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    # ── math (element-wise) ──────────────────────────────────

    def exp(self, x):
        return np.exp(x)

    def cos(self, x):
        return np.cos(x)

    def sin(self, x):
        return np.sin(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def abs(self, x):
        return np.abs(x)

    def real(self, x):
        return np.real(x)

    def conj(self, x):
        return np.conj(x)

    def where(self, cond, a, b):
        return np.where(cond, a, b)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    # ── reductions ───────────────────────────────────────────

    def sum(self, x):
        return np.sum(x)

    def vdot(self, a, b):
        return np.vdot(a, b)

    def max(self, x):
        return np.max(x)

    # ── FFT ──────────────────────────────────────────────────

    def fftn(self, x):
        if self._use_scipy:
            return scipy.fft.fftn(x, workers=self._workers)
        return np.fft.fftn(x)

    def ifftn(self, x):
        if self._use_scipy:
            return scipy.fft.ifftn(x, workers=self._workers)
        return np.fft.ifftn(x)

    def fftfreq(self, n, d):
        return 2 * np.pi * np.fft.fftfreq(n, d=d)

    # ── grid helpers ─────────────────────────────────────────

    def linspace(self, start, stop, n, endpoint=True):
        return np.linspace(start, stop, n, endpoint=endpoint)

    def meshgrid(self, *arrays, indexing="xy"):
        return np.meshgrid(*arrays, indexing=indexing)
