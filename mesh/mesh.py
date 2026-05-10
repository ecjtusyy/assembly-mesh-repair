# -*- coding: utf-8 -*-
"""Python 侧清理流程使用的轻量网格容器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _empty_vertices() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float64)


def _empty_faces() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.int64)


def _normalize_vertices(V: np.ndarray) -> np.ndarray:
    """统一并检查顶点数组。"""
    V = np.asarray(V, dtype=np.float64)

    if V.size == 0:
        return _empty_vertices()

    # 允许单个顶点写成 [x, y, z]。
    if V.ndim == 1:
        if V.size != 3:
            raise ValueError(f"单个顶点必须有 3 个坐标，当前长度为 {V.size}")
        V = V.reshape((1, 3))

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V 必须是形状为 (N, 3) 的数组，当前是 {V.shape}")

    if not np.all(np.isfinite(V)):
        raise ValueError("V 中存在 NaN 或 Inf")

    return V


def _normalize_faces(F: np.ndarray, vertex_count: int) -> np.ndarray:
    """统一并检查三角面数组。"""
    F = np.asarray(F, dtype=np.int64)

    if F.size == 0:
        return _empty_faces()

    # 允许单个三角面写成 [a, b, c]。
    if F.ndim == 1:
        if F.size != 3:
            raise ValueError(f"单个三角面必须有 3 个顶点索引，当前长度为 {F.size}")
        F = F.reshape((1, 3))

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F 必须是形状为 (M, 3) 的数组，当前是 {F.shape}")

    min_idx = int(F.min())
    max_idx = int(F.max())

    if min_idx < 0 or max_idx >= vertex_count:
        raise ValueError(
            f"F 中存在越界顶点索引: "
            f"min={min_idx}, max={max_idx}, 顶点数={vertex_count}"
        )

    return F


@dataclass
class Mesh:
    V: np.ndarray
    F: np.ndarray

    def __post_init__(self) -> None:
        self.V = _normalize_vertices(self.V)
        self.F = _normalize_faces(self.F, vertex_count=self.V.shape[0])
        self._dirty = False

    def copy(self) -> "Mesh":
        """复制一份独立网格。"""
        copied = Mesh(self.V.copy(), self.F.copy())
        copied._dirty = self._dirty
        return copied

    def mark_dirty(self) -> None:
        """标记网格拓扑或几何已经被修改。"""
        self._dirty = True

    @property
    def dirty(self) -> bool:
        return bool(self._dirty)

    def add_vertex(self, point: Sequence[float]) -> int:
        """添加一个顶点，返回新顶点索引。"""
        p = _normalize_vertices(np.asarray(point, dtype=np.float64))

        if p.shape[0] != 1:
            raise ValueError(f"add_vertex 只能添加单个顶点，当前得到 {p.shape[0]} 个")

        self.V = np.vstack([self.V, p]) if self.V.size else p.copy()
        self.mark_dirty()

        return int(self.V.shape[0] - 1)

    def replace_faces(
        self,
        removed_indices: Iterable[int],
        new_faces: Iterable[Sequence[int]],
    ) -> None:
        """删除指定面，并把 new_faces 追加到剩余面之后。"""
        removed = {int(i) for i in removed_indices}

        if removed:
            min_removed = min(removed)
            max_removed = max(removed)

            if min_removed < 0 or max_removed >= self.num_faces:
                raise ValueError(
                    f"删除面编号越界: "
                    f"min={min_removed}, max={max_removed}, 当前面数={self.num_faces}"
                )

        keep_mask = np.array(
            [face_id not in removed for face_id in range(self.num_faces)],
            dtype=bool,
        )
        kept_faces = self.F[keep_mask]

        new_faces_list = list(new_faces)
        new_faces_arr = _normalize_faces(new_faces_list, vertex_count=self.num_vertices)

        if kept_faces.size and new_faces_arr.size:
            self.F = np.vstack([kept_faces, new_faces_arr])
        elif kept_faces.size:
            self.F = kept_faces.copy()
        else:
            self.F = new_faces_arr.copy()

        self.mark_dirty()

    @property
    def num_vertices(self) -> int:
        return int(self.V.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.F.shape[0])