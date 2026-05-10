# -*- coding: utf-8 -*-
"""Python 侧拓扑清理工具。

当前主流程只保留这条清理路线：
- 顶点焊接
- 删除退化三角面
- 删除重复三角面
- 删除孤立顶点并压缩索引

旧的路线不在这里使用：
- Python 侧三角形相交检测
- Python 侧拆边
- T-junction 传播
- 平滑修复
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from mesh.mesh import Mesh


@dataclass
class CleanupReport:
    V_before: int
    V_after: int
    F_before: int
    F_after: int
    merged_vertices: int = 0
    degenerate_removed: int = 0
    duplicate_removed: int = 0
    isolated_removed: int = 0

    def as_dict(self) -> dict:
        """转换成方便写入 JSON 的普通字典。"""
        return {
            "V_before": self.V_before,
            "V_after": self.V_after,
            "F_before": self.F_before,
            "F_after": self.F_after,
            "merged_vertices": self.merged_vertices,
            "degenerate_removed": self.degenerate_removed,
            "duplicate_removed": self.duplicate_removed,
            "isolated_removed": self.isolated_removed,
        }


class _UnionFind:
    """并查集，用来合并距离足够近的顶点。"""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])

        return int(x)

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return ra

        # 保持结果稳定：rank 相同时，让编号更小的代表元获胜。
        if self.rank[ra] < self.rank[rb] or (
            self.rank[ra] == self.rank[rb] and ra > rb
        ):
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        return ra


_NEIGHBOR_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def _empty_vertex_map(n: int = 0) -> np.ndarray:
    return np.zeros((n,), dtype=np.int64)


def _bbox_diag(V: np.ndarray) -> float:
    """计算顶点包围盒对角线长度。"""
    V = np.asarray(V, dtype=np.float64)

    if V.size == 0:
        return 0.0

    lo = V.min(axis=0)
    hi = V.max(axis=0)

    return float(np.linalg.norm(hi - lo))


def area_threshold_from_mesh(mesh: Mesh) -> float:
    """根据网格尺度给退化面判断提供一个面积阈值。"""
    diag = _bbox_diag(mesh.V)

    if diag == 0.0:
        return 0.0

    return float(np.finfo(np.float64).eps * diag * diag)


def face_key(face: Sequence[int]) -> Tuple[int, int, int]:
    """生成三角面的重复判断键；忽略顶点绕序。"""
    a, b, c = int(face[0]), int(face[1]), int(face[2])
    return tuple(sorted((a, b, c)))


def _iter_neighbor_cells(cell: Tuple[int, int, int]) -> Iterator[Tuple[int, int, int]]:
    """遍历当前网格桶及其 26 个邻居桶。"""
    cx, cy, cz = cell

    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        yield cx + dx, cy + dy, cz + dz


def _cell_key(point: np.ndarray, eps_v: float) -> Tuple[int, int, int]:
    """把一个点映射到空间哈希桶。"""
    return tuple(np.floor(point / eps_v).astype(np.int64).tolist())


def _triangle_double_area(V: np.ndarray, tri: Sequence[int]) -> float:
    """返回三角形面积的 2 倍。"""
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])

    pa = V[a]
    pb = V[b]
    pc = V[c]

    return float(np.linalg.norm(np.cross(pb - pa, pc - pa)))


def _exact_vertex_weld(mesh: Mesh) -> np.ndarray:
    """eps_v 为 0 时，只合并坐标完全相同的顶点。"""
    V = np.asarray(mesh.V, dtype=np.float64)
    F = np.asarray(mesh.F, dtype=np.int64)

    if V.size == 0:
        return _empty_vertex_map()

    key_to_new: Dict[Tuple[float, float, float], int] = {}
    kept_vertices: List[np.ndarray] = []
    old2new = np.empty((V.shape[0],), dtype=np.int64)

    for old_idx, point in enumerate(V):
        key = (float(point[0]), float(point[1]), float(point[2]))
        new_idx = key_to_new.get(key)

        if new_idx is None:
            new_idx = len(kept_vertices)
            key_to_new[key] = new_idx
            kept_vertices.append(point)

        old2new[old_idx] = new_idx

    new_V = np.asarray(kept_vertices, dtype=np.float64)
    new_F = old2new[F] if F.size else F.copy()

    if new_V.shape[0] != V.shape[0]:
        mesh.V = new_V
        mesh.F = new_F
        mesh.mark_dirty()

    return old2new


def vertex_weld(mesh: Mesh, eps_v: float) -> np.ndarray:
    """焊接重复或近重复顶点，并同步重映射所有三角面。"""
    eps_v = float(eps_v)

    if eps_v < 0:
        raise ValueError(f"eps_v 不能为负数，当前为 {eps_v}")

    V = np.asarray(mesh.V, dtype=np.float64)
    F = np.asarray(mesh.F, dtype=np.int64)
    n = int(V.shape[0])

    if n == 0:
        return _empty_vertex_map()

    if eps_v == 0:
        return _exact_vertex_weld(mesh)

    uf = _UnionFind(n)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}

    for i, point in enumerate(V):
        key = _cell_key(point, eps_v)

        # 只查当前桶和邻居桶，避免 O(N^2) 全局两两比较。
        for neighbor_key in _iter_neighbor_cells(key):
            for j in buckets.get(neighbor_key, []):
                if np.linalg.norm(point - V[j]) <= eps_v:
                    uf.union(i, j)

        buckets.setdefault(key, []).append(i)

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)

    root_to_rep: Dict[int, int] = {}
    for old_idx, root in enumerate(roots.tolist()):
        rep = root_to_rep.get(root)
        if rep is None or old_idx < rep:
            root_to_rep[root] = old_idx

    reps = np.array(sorted(root_to_rep.values()), dtype=np.int64)
    rep_to_new = {int(rep): new_idx for new_idx, rep in enumerate(reps.tolist())}

    old2new = np.empty((n,), dtype=np.int64)
    for old_idx in range(n):
        rep = root_to_rep[int(roots[old_idx])]
        old2new[old_idx] = rep_to_new[int(rep)]

    if reps.shape[0] != n:
        mesh.V = V[reps]
        mesh.F = old2new[F] if F.size else F.copy()
        mesh.mark_dirty()

    return old2new


def remove_degenerate_faces(mesh: Mesh, area_eps: float | None = None) -> int:
    """删除重复顶点面和近零面积面。"""
    F = np.asarray(mesh.F, dtype=np.int64)

    if F.size == 0:
        return 0

    V = np.asarray(mesh.V, dtype=np.float64)

    if area_eps is None:
        area_eps = area_threshold_from_mesh(mesh)

    area_eps = float(area_eps)
    if area_eps < 0:
        raise ValueError(f"area_eps 不能为负数，当前为 {area_eps}")

    keep_mask = np.ones((F.shape[0],), dtype=bool)

    for face_idx, tri in enumerate(F):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])

        if a == b or b == c or a == c:
            keep_mask[face_idx] = False
            continue

        area2 = _triangle_double_area(V, tri)
        if not np.isfinite(area2) or area2 <= 2.0 * area_eps:
            keep_mask[face_idx] = False

    removed = int((~keep_mask).sum())

    if removed:
        mesh.F = F[keep_mask]
        mesh.mark_dirty()

    return removed


def remove_duplicate_faces(mesh: Mesh) -> int:
    """删除重复三角面；重复判断忽略绕序。"""
    F = np.asarray(mesh.F, dtype=np.int64)

    if F.size == 0:
        return 0

    keep_indices: List[int] = []
    seen: Dict[Tuple[int, int, int], int] = {}

    for face_idx, tri in enumerate(F):
        key = face_key(tri)

        if key in seen:
            continue

        seen[key] = face_idx
        keep_indices.append(face_idx)

    removed = int(F.shape[0] - len(keep_indices))

    if removed:
        mesh.F = F[np.asarray(keep_indices, dtype=np.int64)]
        mesh.mark_dirty()

    return removed


def compact_vertices(mesh: Mesh) -> np.ndarray:
    """删除孤立顶点，并把面索引压缩到连续区间。"""
    V = np.asarray(mesh.V, dtype=np.float64)
    F = np.asarray(mesh.F, dtype=np.int64)

    if V.size == 0:
        return _empty_vertex_map()

    old2new = np.full((V.shape[0],), -1, dtype=np.int64)

    if F.size == 0:
        mesh.V = np.zeros((0, 3), dtype=np.float64)
        mesh.F = np.zeros((0, 3), dtype=np.int64)
        mesh.mark_dirty()
        return old2new

    used = np.unique(F.reshape(-1))
    old2new[used] = np.arange(used.shape[0], dtype=np.int64)

    if used.shape[0] != V.shape[0] or not np.array_equal(used, np.arange(V.shape[0])):
        mesh.V = V[used]
        mesh.F = old2new[F]
        mesh.mark_dirty()

    return old2new


def cleanup_topology(
    mesh: Mesh,
    eps_v: float = 0.0,
    area_eps: float | None = None,
) -> CleanupReport:
    """执行主流程使用的非移动拓扑清理。"""
    report = CleanupReport(
        V_before=mesh.num_vertices,
        V_after=mesh.num_vertices,
        F_before=mesh.num_faces,
        F_after=mesh.num_faces,
    )

    vertex_weld(mesh, eps_v=eps_v)
    report.merged_vertices = int(report.V_before - mesh.num_vertices)

    report.degenerate_removed = remove_degenerate_faces(mesh, area_eps=area_eps)
    report.duplicate_removed = remove_duplicate_faces(mesh)

    before_compact = mesh.num_vertices
    compact_vertices(mesh)
    report.isolated_removed = int(before_compact - mesh.num_vertices)

    report.V_after = mesh.num_vertices
    report.F_after = mesh.num_faces

    return report


def mesh_has_duplicate_faces(mesh: Mesh) -> bool:
    """判断网格中是否存在重复三角面。"""
    F = np.asarray(mesh.F, dtype=np.int64)

    if F.size == 0:
        return False

    keys = {face_key(tri) for tri in F}
    return len(keys) != int(F.shape[0])


def mesh_has_degenerate_faces(mesh: Mesh, area_eps: float | None = None) -> bool:
    """判断网格中是否存在退化三角面。"""
    V = np.asarray(mesh.V, dtype=np.float64)
    F = np.asarray(mesh.F, dtype=np.int64)

    if F.size == 0:
        return False

    if area_eps is None:
        area_eps = area_threshold_from_mesh(mesh)

    area_eps = float(area_eps)
    if area_eps < 0:
        raise ValueError(f"area_eps 不能为负数，当前为 {area_eps}")

    for tri in F:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])

        if a == b or b == c or a == c:
            return True

        area2 = _triangle_double_area(V, tri)
        if not np.isfinite(area2) or area2 <= 2.0 * area_eps:
            return True

    return False