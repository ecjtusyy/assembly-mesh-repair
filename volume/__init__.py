"""四面体体网格生成与验收。"""

from volume.gmsh_tetra import TetrahedralizationOptions, tetrahedralize
from volume.tetgen_tetra import (
    StrictTetrahedralizationOptions,
    tetrahedralize_strict,
)
from volume.tetra_mesh import TetraMesh

__all__ = [
    "StrictTetrahedralizationOptions",
    "TetraMesh",
    "TetrahedralizationOptions",
    "tetrahedralize",
    "tetrahedralize_strict",
]
