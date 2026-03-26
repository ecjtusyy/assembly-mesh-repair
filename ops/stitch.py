# -*- coding: utf-8 -*-
"""Topology-preserving cleanup utilities.

This module contains the new Python-side welding and cleanup path used before and after
CGAL autorefinement. The old teaching-only stitch / split / T-junction route is no
longer on the main repair path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

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
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        # Deterministic: smaller representative wins when ranks tie.
        if self.rank[ra] < self.rank[rb] or (self.rank[ra] == self.rank[rb] and ra > rb):
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


_NEIGHBOR_OFFSETS = tuple(product((-1, 0, 1), repeat=3))



def _bbox_diag(V: np.ndarray) -> float:
    if V.size == 0:
        return 0.0
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    return float(np.linalg.norm(hi - lo))



def area_threshold_from_mesh(mesh: Mesh) -> float:
    diag = _bbox_diag(mesh.V)
    if diag == 0.0:
        return 0.0
    return float(np.finfo(np.float64).eps * diag * diag)



def face_key(face: Sequence[int]) -> Tuple[int, int, int]:
    """Canonical duplicate key ignoring winding."""
    a, b, c = (int(face[0]), int(face[1]), int(face[2]))
    return tuple(sorted((a, b, c)))



def _iter_neighbor_cells(cell: Tuple[int, int, int]) -> Iterator[Tuple[int, int, int]]:
    cx, cy, cz = cell
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        yield (cx + dx, cy + dy, cz + dz)



def vertex_weld(mesh: Mesh, eps_v: float) -> np.ndarray:
    """Weld duplicate / near-duplicate vertices and remap all faces.

    Returns ``old2new_map`` mapping the original vertex indices to the retained compacted
    vertex indices.
    """
    V = np.asarray(mesh.V, dtype=np.float64)
    F = np.asarray(mesh.F, dtype=np.int64)
    n = int(V.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if eps_v < 0:
        raise ValueError(f"eps_v must be >= 0, got {eps_v}")
    if eps_v == 0:
        uniq, inverse = np.unique(V, axis=0, return_inverse=True)
        mesh.V = uniq.astype(np.float64, copy=False)
        mesh.F = inverse[F] if F.size else F.copy()
        mesh.mark_dirty()
        return inverse.astype(np.int64, copy=False)

    uf = _UnionFind(n)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}

    for i, p in enumerate(V):
        key = tuple(np.floor(p / eps_v).astype(np.int64).tolist())
        for neighbor_key in _iter_neighbor_cells(key):
            for j in buckets.get(neighbor_key, []):
                if np.linalg.norm(p - V[j]) <= eps_v:
                    uf.union(i, j)
        buckets.setdefault(key, []).append(i)

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    root_to_rep: Dict[int, int] = {}
    for i, r in enumerate(roots.tolist()):
        rep = root_to_rep.get(r)
        if rep is None or i < rep:
            root_to_rep[r] = i
    reps = np.array(sorted(root_to_rep.values()), dtype=np.int64)
    rep_to_new = {int(rep): idx for idx, rep in enumerate(reps.tolist())}

    old2new = np.empty((n,), dtype=np.int64)
    for i in range(n):
        rep = root_to_rep[int(roots[i])]
        old2new[i] = rep_to_new[int(rep)]

    mesh.V = V[reps]
    mesh.F = old2new[F] if F.size else F.copy()
    mesh.mark_dirty()
    return old2new



def remove_degenerate_faces(mesh: Mesh, area_eps: float | None = None) -> int:
    """Remove faces with repeated indices or near-zero area."""
    F = np.asarray(mesh.F, dtype=np.int64)
    if F.size == 0:
        return 0
    V = np.asarray(mesh.V, dtype=np.float64)
    if area_eps is None:
        area_eps = area_threshold_from_mesh(mesh)

    keep_mask = np.ones((F.shape[0],), dtype=bool)
    for i, tri in enumerate(F):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a == b or b == c or a == c:
            keep_mask[i] = False
            continue
        pa, pb, pc = V[a], V[b], V[c]
        area2 = np.linalg.norm(np.cross(pb - pa, pc - pa))
        if not np.isfinite(area2) or area2 <= 2.0 * area_eps:
            keep_mask[i] = False

    removed = int((~keep_mask).sum())
    if removed:
        mesh.F = F[keep_mask]
        mesh.mark_dirty()
    return removed



def remove_duplicate_faces(mesh: Mesh) -> int:
    """Remove duplicate triangles using a canonical key that ignores winding."""
    F = np.asarray(mesh.F, dtype=np.int64)
    if F.size == 0:
        return 0
    keep_indices: List[int] = []
    seen: Dict[Tuple[int, int, int], int] = {}
    for i, tri in enumerate(F):
        key = face_key(tri)
        if key in seen:
            continue
        seen[key] = i
        keep_indices.append(i)
    removed = int(F.shape[0] - len(keep_indices))
    if removed:
        mesh.F = F[np.asarray(keep_indices, dtype=np.int64)]
        mesh.mark_dirty()
    return removed



def compact_vertices(mesh: Mesh) -> np.ndarray:
    """Remove isolated vertices and compact face indices to ``[0, n)``."""
    V = np.asarray(mesh.V, dtype=np.float64)
    F = np.asarray(mesh.F, dtype=np.int64)
    if V.size == 0:
        return np.zeros((0,), dtype=np.int64)
    old2new = np.full((V.shape[0],), -1, dtype=np.int64)
    if F.size == 0:
        mesh.V = np.zeros((0, 3), dtype=np.float64)
        mesh.F = np.zeros((0, 3), dtype=np.int64)
        mesh.mark_dirty()
        return old2new

    used = np.unique(F.reshape(-1))
    old2new[used] = np.arange(used.shape[0], dtype=np.int64)
    mesh.V = V[used]
    mesh.F = old2new[F]
    mesh.mark_dirty()
    return old2new



def cleanup_topology(mesh: Mesh, eps_v: float = 0.0, area_eps: float | None = None) -> CleanupReport:
    """Run the non-moving cleanup sequence used by the new main pipeline."""
    report = CleanupReport(
        V_before=mesh.num_vertices,
        V_after=mesh.num_vertices,
        F_before=mesh.num_faces,
        F_after=mesh.num_faces,
    )
    old2new = vertex_weld(mesh, eps_v=eps_v)
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
    F = np.asarray(mesh.F, dtype=np.int64)
    if F.size == 0:
        return False
    return len({face_key(tri) for tri in F}) != int(F.shape[0])



def mesh_has_degenerate_faces(mesh: Mesh, area_eps: float | None = None) -> bool:
    V = np.asarray(mesh.V, dtype=np.float64)
    F = np.asarray(mesh.F, dtype=np.int64)
    if F.size == 0:
        return False
    if area_eps is None:
        area_eps = area_threshold_from_mesh(mesh)
    for tri in F:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        if len({a, b, c}) != 3:
            return True
        pa, pb, pc = V[a], V[b], V[c]
        area2 = np.linalg.norm(np.cross(pb - pa, pc - pa))
        if not np.isfinite(area2) or area2 <= 2.0 * area_eps:
            return True
    return False
