"""
Accelerated 2D Dirac simulation with CPU/GPU backend abstraction.

Usage:
    from accel import get_backend
    xp = get_backend("auto")       # auto-detect CUDA > MPS > scipy
    xp = get_backend("torch_cuda") # NVIDIA GPU (CUDA)
    xp = get_backend("torch_mps")  # Apple GPU (MPS)
"""

from .backend_numpy import NumpyBackend

try:
    from .backend_torch import TorchMPSBackend
    _HAS_TORCH_MPS = True
except ImportError:
    TorchMPSBackend = None
    _HAS_TORCH_MPS = False

try:
    from .backend_torch_cuda import TorchCUDABackend
    _HAS_TORCH_CUDA = True
except ImportError:
    TorchCUDABackend = None
    _HAS_TORCH_CUDA = False


def get_backend(name="auto"):
    """
    Create a backend by name.

    Parameters
    ----------
    name : str
        "auto"        — try CUDA, then MPS, then scipy
        "torch_cuda"  — PyTorch CUDA (NVIDIA GPU)
        "torch_mps"   — PyTorch MPS (Apple GPU)
        "numpy_scipy" — NumPy + scipy.fft multi-threaded
        "numpy"       — pure NumPy (baseline)
    """
    if name == "auto":
        # 1. NVIDIA CUDA
        if _HAS_TORCH_CUDA:
            try:
                import torch
                if torch.cuda.is_available():
                    print("Auto-selected backend: torch_cuda")
                    return TorchCUDABackend()
            except Exception:
                pass
        # 2. Apple MPS
        if _HAS_TORCH_MPS:
            try:
                import torch
                if torch.backends.mps.is_available():
                    print("Auto-selected backend: torch_mps")
                    return TorchMPSBackend()
            except Exception:
                pass
        # 3. CPU fallback
        return NumpyBackend(use_scipy_fft=True)

    if name == "torch_cuda":
        if not _HAS_TORCH_CUDA:
            raise ImportError("PyTorch not installed")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available on this node")
        return TorchCUDABackend()

    if name == "torch_mps":
        if not _HAS_TORCH_MPS:
            raise ImportError("PyTorch not installed or MPS not available")
        return TorchMPSBackend()

    if name == "numpy_scipy":
        return NumpyBackend(use_scipy_fft=True)

    if name == "numpy":
        return NumpyBackend(use_scipy_fft=False)

    raise ValueError(f"Unknown backend: {name!r}. "
                     f"Choose from: auto, torch_cuda, torch_mps, numpy_scipy, numpy")
