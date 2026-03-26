# -*- coding: utf-8 -*-
"""Lightweight mesh container used by the Python-side cleanup pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Mesh:
    V: np.ndarray
    F: np.ndarray

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float64).reshape((-1, 3))
        self.F = np.asarray(self.F, dtype=np.int64).reshape((-1, 3))
        self._dirty = False

    def copy(self) -> "Mesh":
        return Mesh(self.V.copy(), self.F.copy())

    def mark_dirty(self) -> None:
        self._dirty = True

    def add_vertex(self, point: Sequence[float]) -> int:
        p = np.asarray(point, dtype=np.float64).reshape((1, 3))
        self.V = np.vstack([self.V, p]) if self.V.size else p.copy()
        self.mark_dirty()
        return int(self.V.shape[0] - 1)

    def replace_faces(self, removed_indices: Iterable[int], new_faces: Iterable[Sequence[int]]) -> None:
        removed = set(int(i) for i in removed_indices)
        keep_mask = np.array([fi not in removed for fi in range(self.F.shape[0])], dtype=bool)
        kept = self.F[keep_mask]
        new_faces_list = list(new_faces)
        if new_faces_list:
            new_faces_arr = np.asarray(new_faces_list, dtype=np.int64).reshape((-1, 3))
        else:
            new_faces_arr = np.zeros((0, 3), dtype=np.int64)
        if kept.size and new_faces_arr.size:
            self.F = np.vstack([kept, new_faces_arr])
        elif kept.size:
            self.F = kept.copy()
        else:
            self.F = new_faces_arr.copy()
        self.mark_dirty()

    @property
    def num_vertices(self) -> int:
        return int(self.V.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.F.shape[0])
