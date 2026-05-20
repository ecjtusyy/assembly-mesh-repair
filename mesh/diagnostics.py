# -*- coding: utf-8 -*-
"""
判断当前网格中有没有拓扑问题
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np


Edge = Tuple[int, int]


def _normalize_faces(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.int64)
    if F.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F 必须是形状为 (n_faces, 3) 的三角面数组，当前是 {F.shape}")
    return F


def _normalize_vertices(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.float64)
    if V.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V 必须是形状为 (n_vertices, 3) 的数组，当前是 {V.shape}")
    if not np.all(np.isfinite(V)):
        raise ValueError("V 中存在 NaN 或 Inf")
    return V


def _edge_key(u: int, v: int) -> Edge:
    a, b = int(u), int(v)
    return (a, b) if a <= b else (b, a)


def build_edge_faces(F: np.ndarray) -> Dict[Edge, List[int]]:
    """
    遍历每个三角面
        → 拿到三条边
        → 把边变成无向边
        → 记录这条边属于哪个 face
    """
    faces = _normalize_faces(F)
    edge_faces: Dict[Edge, List[int]] = {}

    for face_id, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault(_edge_key(u, v), []).append(int(face_id))

    return edge_faces


def count_degenerate_faces(
    V: np.ndarray,
    F: np.ndarray,
    eps_area: float = 1e-30,
) -> dict[str, object]:
    """统计退化三角形数量。

    使用叉积范数平方判断面积退化；两个顶点索引相同也直接视为退化。
    """
    vertices = _normalize_vertices(V)
    faces = _normalize_faces(F)
    eps_area = float(eps_area)
    sample: list[int] = []
    count = 0

    for face_id, tri in enumerate(faces):
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        is_degenerate = a == b or b == c or a == c

        if not is_degenerate:
            pa, pb, pc = vertices[a], vertices[b], vertices[c]
            cross = np.cross(pb - pa, pc - pa)
            area_measure = float(np.dot(cross, cross))
            is_degenerate = (not np.isfinite(area_measure)) or area_measure <= eps_area

        if is_degenerate:
            count += 1
            if len(sample) < 20:
                sample.append(int(face_id))

    return {"count": int(count), "sample_face_ids": sample}


def count_duplicate_faces(F: np.ndarray) -> dict[str, object]:
    """按无向三角面顶点集合统计重复面。"""
    faces = _normalize_faces(F)
    seen: dict[tuple[int, int, int], int] = {}
    sample: list[int] = []
    count = 0

    for face_id, tri in enumerate(faces):
        key = tuple(sorted((int(tri[0]), int(tri[1]), int(tri[2]))))
        if key in seen:
            count += 1
            if len(sample) < 20:
                sample.append(int(face_id))
        else:
            seen[key] = int(face_id)

    return {"count": int(count), "sample_face_ids": sample}


def connected_components_by_edges(F: np.ndarray) -> dict[str, object]:
    """按共享边统计 face edge-connected components。"""
    faces = _normalize_faces(F)
    n_faces = int(faces.shape[0])
    if n_faces == 0:
        return {
            "component_count": 0,
            "component_sizes": [],
            "face_component_id": [],
        }

    edge_faces = build_edge_faces(faces)
    neighbors: list[set[int]] = [set() for _ in range(n_faces)]
    for incident in edge_faces.values():
        if len(incident) < 2:
            continue
        for i in incident:
            for j in incident:
                if i != j:
                    neighbors[i].add(j)

    face_component_id = [-1] * n_faces
    component_sizes: list[int] = []

    for start in range(n_faces):
        if face_component_id[start] != -1:
            continue

        comp_id = len(component_sizes)
        q: deque[int] = deque([start])
        face_component_id[start] = comp_id
        size = 0

        while q:
            cur = q.popleft()
            size += 1
            for nxt in neighbors[cur]:
                if face_component_id[nxt] == -1:
                    face_component_id[nxt] = comp_id
                    q.append(nxt)

        component_sizes.append(int(size))

    return {
        "component_count": int(len(component_sizes)),
        "component_sizes": component_sizes,
        "face_component_id": [int(x) for x in face_component_id],
    }


def _sample_edges(edges: list[Edge]) -> list[list[int]]:
    return [[int(a), int(b)] for a, b in edges[:20]]


def topology_summary(V: np.ndarray, F: np.ndarray) -> dict[str, object]:
    """返回 JSON 可序列化的基础拓扑诊断。"""
    vertices = _normalize_vertices(V)
    faces = _normalize_faces(F)
    edge_faces = build_edge_faces(faces)

    boundary_edges = [edge for edge, ids in edge_faces.items() if len(ids) == 1]
    nonmanifold_edges = [edge for edge, ids in edge_faces.items() if len(ids) > 2]
    max_edge_incidence = max((len(ids) for ids in edge_faces.values()), default=0)

    dup = count_duplicate_faces(faces)
    deg = count_degenerate_faces(vertices, faces)
    comps = connected_components_by_edges(faces)

    return {
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "edges": int(len(edge_faces)),
        "boundary_edges": int(len(boundary_edges)),
        "nonmanifold_edges": int(len(nonmanifold_edges)),
        "max_edge_incidence": int(max_edge_incidence),
        "duplicate_faces": int(dup["count"]),
        "degenerate_faces": int(deg["count"]),
        "edge_component_count": int(comps["component_count"]),
        "edge_component_sizes": [int(x) for x in comps["component_sizes"]],
        "sample_boundary_edges": _sample_edges(boundary_edges),
        "sample_nonmanifold_edges": _sample_edges(nonmanifold_edges),
        "sample_duplicate_faces": [int(x) for x in dup["sample_face_ids"]],
        "sample_degenerate_faces": [int(x) for x in deg["sample_face_ids"]],
    }
