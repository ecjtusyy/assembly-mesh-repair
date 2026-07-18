"""四面体网格数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TetraMesh:
    V: np.ndarray
    T: np.ndarray
    boundary_faces: np.ndarray

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float64).reshape((-1, 3))
        self.T = np.asarray(self.T, dtype=np.int64).reshape((-1, 4))
        self.boundary_faces = np.asarray(
            self.boundary_faces, dtype=np.int64
        ).reshape((-1, 3))
        if not np.all(np.isfinite(self.V)):
            raise ValueError("体网格顶点中存在 NaN 或 Inf")
        if len(self.T) == 0:
            raise ValueError("Gmsh 没有生成一阶四面体")
        for name, cells in (("T", self.T), ("boundary_faces", self.boundary_faces)):
            if len(cells) and (int(cells.min()) < 0 or int(cells.max()) >= len(self.V)):
                raise ValueError(f"{name} 中存在越界顶点索引")
