"""
Abstract backend interface for array/FFT operations.

Enables switching between CPU (NumPy/SciPy) and GPU (PyTorch MPS) with
a single parameter.  Each backend implements this protocol.
"""

from abc import ABC, abstractmethod


class ArrayBackend(ABC):
    """Protocol for array & FFT operations used by the Dirac propagator."""

    name: str  # "numpy" or "torch_mps"

    # ── array creation / conversion ──────────────────────────

    @abstractmethod
    def array(self, data):
        """Convert a NumPy ndarray (or scalar) to backend array."""

    @abstractmethod
    def to_numpy(self, arr):
        """Convert backend array back to a NumPy ndarray."""

    @abstractmethod
    def zeros(self, shape, dtype=None):
        ...

    @abstractmethod
    def zeros_like(self, x):
        ...

    @abstractmethod
    def empty_like(self, x):
        ...

    @abstractmethod
    def copy(self, x):
        ...

    @abstractmethod
    def stack(self, arrays, axis=0):
        ...

    # ── math (element-wise) ──────────────────────────────────

    @abstractmethod
    def exp(self, x):
        ...

    @abstractmethod
    def cos(self, x):
        ...

    @abstractmethod
    def sin(self, x):
        ...

    @abstractmethod
    def sqrt(self, x):
        ...

    @abstractmethod
    def abs(self, x):
        ...

    @abstractmethod
    def real(self, x):
        ...

    @abstractmethod
    def conj(self, x):
        ...

    @abstractmethod
    def where(self, cond, a, b):
        ...

    @abstractmethod
    def clip(self, x, lo, hi):
        ...

    # ── reductions ───────────────────────────────────────────

    @abstractmethod
    def sum(self, x):
        ...

    @abstractmethod
    def vdot(self, a, b):
        """Conjugate inner product (flattened): sum(conj(a) * b)."""

    @abstractmethod
    def max(self, x):
        ...

    # ── FFT ──────────────────────────────────────────────────

    @abstractmethod
    def fftn(self, x):
        """2-D FFT (last two dims)."""

    @abstractmethod
    def ifftn(self, x):
        """2-D inverse FFT (last two dims)."""

    @abstractmethod
    def fftfreq(self, n, d):
        """Return 1-D frequency array (as backend array)."""

    # ── grid helpers ─────────────────────────────────────────

    @abstractmethod
    def linspace(self, start, stop, n, endpoint=True):
        ...

    @abstractmethod
    def meshgrid(self, *arrays, indexing="xy"):
        ...
