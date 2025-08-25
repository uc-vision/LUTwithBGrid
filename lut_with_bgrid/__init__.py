"""LUT with Bilateral Grid - Image-adaptive 3D Lookup Tables for Real-time Image Enhancement."""

__version__ = "0.1.0"

from .cpp_ext_interface import (
    trilinear_lut_transform,
    tetrahedral_lut_transform,
    trilinear_slice_function,
    tetrahedral_slice_function,
)
from .models import (
    LUTwithBGrid,
    Gen_3D_LUT,
    Gen_bilateral_grids,
    Slice,
    Backbone,
    resnet18_224,
)
from .datasets import (
    ImageDataset_sRGB,
    ImageDataset_XYZ,
    ImageDataset_PPR10k,
)
from .model_losses import (
    TV_3D,
    DeltaE_loss,
)

__all__ = [
    # Main functionality
    "LUTwithBGrid",
    "trilinear_lut_transform",
    "tetrahedral_lut_transform",
    "trilinear_slice_function",
    "tetrahedral_slice_function",
    # Models
    "Gen_3D_LUT",
    "Gen_bilateral_grids",
    "Slice",
    "Backbone",
    "resnet18_224",
    # Datasets
    "ImageDataset_sRGB",
    "ImageDataset_XYZ",
    "ImageDataset_PPR10k",
    # Losses
    "TV_3D",
    "DeltaE_loss",
]

