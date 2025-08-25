"""CPU fallback implementations for CUDA extensions."""

import torch


class LutTransformCPU:
    """CPU implementation of LUT transform operations."""

    def tri_forward(self, lut: torch.Tensor, x: torch.Tensor, output: torch.Tensor) -> None:
        """Trilinear LUT forward pass on CPU."""
        # This is a simplified CPU implementation
        # In practice, you'd implement the actual trilinear interpolation here
        raise NotImplementedError(
            "CUDA extension not available. "
            "Install with: pip install lut-with-bgrid[cuda]"
        )

    def tri_backward(self, grad_output: torch.Tensor, lut: torch.Tensor, x: torch.Tensor,
                    grad_lut: torch.Tensor, grad_img: torch.Tensor) -> None:
        """Trilinear LUT backward pass on CPU."""
        raise NotImplementedError(
            "CUDA extension not available. "
            "Install with: pip install lut-with-bgrid[cuda]"
        )

    def tetra_forward(self, lut: torch.Tensor, x: torch.Tensor, output: torch.Tensor) -> None:
        """Tetrahedral LUT forward pass on CPU."""
        raise NotImplementedError(
            "CUDA extension not available. "
            "Install with: pip install lut-with-bgrid[cuda]"
        )

    def tetra_backward(self, grad_output: torch.Tensor, lut: torch.Tensor, x: torch.Tensor,
                      grad_lut: torch.Tensor, grad_img: torch.Tensor) -> None:
        """Tetrahedral LUT backward pass on CPU."""
        raise NotImplementedError(
            "CUDA extension not available. "
            "Install with: pip install lut-with-bgrid[cuda]"
        )


class BilateralSlicingCPU:
    """CPU implementation of bilateral slicing operations."""

    def tri_forward(self, grid: torch.Tensor, x: torch.Tensor, output: torch.Tensor) -> None:
        """Trilinear bilateral grid forward pass on CPU."""
        raise NotImplementedError(
            "CUDA extension not available. "
            "Install with: pip install lut-with-bgrid[cuda]"
        )

    def tri_backward(self, grad_output: torch.Tensor, grid: torch.Tensor, x: torch.Tensor,
                    grad_grid: torch.Tensor, grad_img: torch.Tensor) -> None:
        """Trilinear bilateral grid backward pass on CPU."""
        raise NotImplementedError(
            "CUDA extension not available. "
            "Install with: pip install lut-with-bgrid[cuda]"
        )

    def tetra_forward(self, grid: torch.Tensor, x: torch.Tensor, output: torch.Tensor) -> None:
        """Tetrahedral bilateral grid forward pass on CPU."""
        raise NotImplementedError(
            "CUDA extension not available. "
            "Install with: pip install lut-with-bgrid[cuda]"
        )

    def tetra_backward(self, grad_output: torch.Tensor, grid: torch.Tensor, x: torch.Tensor,
                      grad_grid: torch.Tensor, grad_img: torch.Tensor) -> None:
        """Tetrahedral bilateral grid backward pass on CPU."""
        raise NotImplementedError(
            "CUDA extension not available. "
            "Install with: pip install lut-with-bgrid[cuda]"
        )
