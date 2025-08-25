"""Dynamic loading of CUDA extensions for LUT with Bilateral Grid."""

import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import torch

# Cache for loaded extensions
_lut_transform = None
_bilateral_slicing = None

def _extension_dir() -> Path:
    """Get extension source directory."""
    return Path(__file__).parent / "kernel_code"

def _check_ninja() -> bool:
    """Check if ninja is available for compilation."""
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", "import ninja"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def _build_extension_if_needed() -> bool:
    """Build extensions if source exists and CUDA is available."""
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, using CPU fallbacks", UserWarning)
        return False

    ext_dir = _extension_dir()
    if not ext_dir.exists():
        warnings.warn("Extension source directory not found", UserWarning)
        return False

    if not _check_ninja():
        warnings.warn(
            "ninja not available, falling back to CPU implementation. "
            "Install ninja for CUDA extensions: pip install ninja",
            UserWarning
        )
        return False

    try:
        from torch.utils.cpp_extension import load

        # Build lut_transform
        lut_dir = ext_dir / "lut_transform"
        if lut_dir.exists():
            global _lut_transform
            if _lut_transform is None:
                print("Building lut_transform extension...")
                _lut_transform = load(
                    name="lut_transform",
                    sources=[
                        str(lut_dir / "src/lut_transform.cpp"),
                        str(lut_dir / "src/trilinear_cpu.cpp"),
                        str(lut_dir / "src/trilinear_cuda.cu"),
                        str(lut_dir / "src/tetrahedral_cpu.cpp"),
                        str(lut_dir / "src/tetrahedral_cuda.cu"),
                    ],
                    verbose=False,
                    with_cuda=True
                )

        # Build bilateral_slicing
        bilateral_dir = ext_dir / "bilateral_slicing"
        if bilateral_dir.exists():
            global _bilateral_slicing
            if _bilateral_slicing is None:
                print("Building bilateral_slicing extension...")
                _bilateral_slicing = load(
                    name="bilateral_slicing",
                    sources=[
                        str(bilateral_dir / "src/bilateral_slicing.cpp"),
                        str(bilateral_dir / "src/trilinear_slice_cpu.cpp"),
                        str(bilateral_dir / "src/trilinear_slice_cuda.cu"),
                        str(bilateral_dir / "src/tetrahedral_slice_cpu.cpp"),
                        str(bilateral_dir / "src/tetrahedral_slice_cuda.cu"),
                    ],
                    verbose=False,
                    with_cuda=True
                )

        return True

    except Exception as e:
        warnings.warn(f"Extension building failed: {e}. Using CPU fallback implementations", UserWarning)
        return False

def get_lut_transform() -> Any:
    """Get lut_transform extension."""
    global _lut_transform

    if _lut_transform is not None:
        return _lut_transform

    # Try to import existing extension
    try:
        import lut_transform
        _lut_transform = lut_transform
        return _lut_transform
    except ImportError:
        pass

    # Try to build extension
    if _build_extension_if_needed():
        return _lut_transform

    # Fall back to CPU implementation
    from .cpu_fallbacks import LutTransformCPU
    _lut_transform = LutTransformCPU()
    return _lut_transform

def get_bilateral_slicing() -> Any:
    """Get bilateral_slicing extension."""
    global _bilateral_slicing

    if _bilateral_slicing is not None:
        return _bilateral_slicing

    # Try to import existing extension
    try:
        import bilateral_slicing
        _bilateral_slicing = bilateral_slicing
        return _bilateral_slicing
    except ImportError:
        pass

    # Try to build extension
    if _build_extension_if_needed():
        return _bilateral_slicing

    # Fall back to CPU implementation
    from .cpu_fallbacks import BilateralSlicingCPU
    _bilateral_slicing = BilateralSlicingCPU()
    return _bilateral_slicing

