"""NinaXander shared model and result utilities."""

from .adapter import LatentAdapter, ResBlock, blocks_of, r2
from .b_to_a_chimera import (
    BToAChimera,
    ResidualOffsets,
    detect_residual_offsets,
    fit_b_to_a_affine_maps,
    relative_error,
)

__all__ = [
    "LatentAdapter",
    "ResBlock",
    "ResidualOffsets",
    "BToAChimera",
    "blocks_of",
    "detect_residual_offsets",
    "fit_b_to_a_affine_maps",
    "r2",
    "relative_error",
]
