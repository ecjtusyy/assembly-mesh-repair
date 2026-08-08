"""网格拓扑诊断：根据 V/F 统计边界边、非流形边、重复面、退化面和连通分量。"""

from __future__ import annotations

from collections import deque

import numpy as np


Edge = tuple[int, int]
FaceKey = tuple[int, int, int]


def _as_faces(F: np.ndarray) -> np.ndarray:
    """把输入转换成三角面数组，并检查形状是否为 (n_faces, 3)。"""
    faces = np.asarray(F, dtype=np.int64)

    if faces.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"F 必须是形状为 (n_faces, 3) 的三角面数组，当前是 {faces.shape}")

    return faces


def _as_vertices(V: np.ndarray) -> np.ndarray:
    """把输入转换成顶点坐标数组，并检查形状是否为 (n_vertices, 3)。"""
    vertices = np.asarray(V, dtype=np.float64)

    if vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"V 必须是形状为 (n_vertices, 3) 的顶点数组，当前是 {vertices.shape}")

    if not np.all(np.isfinite(vertices)):
        raise ValueError("V 中存在 NaN 或 Inf")

    return vertices


def _validate_face_indices(faces: np.ndarray, vertex_count: int) -> None:
    """检查面索引是否落在顶点数组范围内。"""
    if len(faces) == 0:
        return

    min_index = int(faces.min())
    max_index = int(faces.max())
    if min_index < 0 or max_index >= vertex_count:
        raise ValueError(
            f"F 中存在越界顶点索引：最小值 {min_index}，最大值 {max_index}，"
            f"顶点数 {vertex_count}"
        )


def _edge_key(u: int, v: int) -> Edge:
    """把有向边 u-v 统一成无向边 key，方便统计同一条边出现次数。"""
    a, b = int(u), int(v)
    return (a, b) if a <= b else (b, a)


def _face_key(face: np.ndarray) -> FaceKey:
    """把三角面顶点编号排序，用于判断不同朝向下的重复面。"""
    a, b, c = (int(face[0]), int(face[1]), int(face[2]))
    return tuple(sorted((a, b, c)))


def _face_edges(face: np.ndarray) -> tuple[Edge, Edge, Edge]:
    """把一个三角面拆成三条无向边。"""
    a, b, c = (int(face[0]), int(face[1]), int(face[2]))
    return (
        _edge_key(a, b),
        _edge_key(b, c),
        _edge_key(c, a),
    )


def build_edge_faces(F: np.ndarray) -> dict[Edge, list[int]]:
    """建立边到面的反查表：edge -> 使用这条边的 face_id 列表。

    核心逻辑：
        遍历每个三角面；
        把该面拆成三条无向边；
        将当前 face_id 追加到对应边的列表里。
    """
    faces = _as_faces(F)
    edge_faces: dict[Edge, list[int]] = {}

    for face_id, face in enumerate(faces):
        for edge in _face_edges(face):
            edge_faces.setdefault(edge, []).append(int(face_id))

    return edge_faces


def count_nonmanifold_vertices(F: np.ndarray) -> dict[str, object]:
    """统计拥有多个独立面扇的顶点。"""

    faces = _as_faces(F)
    if len(faces) == 0:
        return {"count": 0, "sample_vertex_ids": []}

    edge_faces = build_edge_faces(faces)
    incident: dict[int, list[int]] = {}
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            incident.setdefault(int(vertex_id), []).append(face_id)

    bad_vertices: list[int] = []
    for vertex_id, face_ids in incident.items():
        neighbors = {face_id: set() for face_id in face_ids}
        for face_id in face_ids:
            for edge in _face_edges(faces[face_id]):
                if vertex_id not in edge:
                    continue
                uses = edge_faces[edge]
                if len(uses) == 2:
                    a, b = uses
                    neighbors[a].add(b)
                    neighbors[b].add(a)

        seen: set[int] = set()
        fans = 0
        for start in face_ids:
            if start in seen:
                continue
            fans += 1
            stack = [start]
            seen.add(start)
            while stack:
                current = stack.pop()
                for next_id in neighbors[current]:
                    if next_id not in seen:
                        seen.add(next_id)
                        stack.append(next_id)

        if fans > 1:
            bad_vertices.append(vertex_id)

    return {
        "count": int(len(bad_vertices)),
        "sample_vertex_ids": [int(x) for x in bad_vertices[:20]],
    }


def count_degenerate_faces(
    V: np.ndarray,
    F: np.ndarray,
    eps_area: float = 1e-30,
) -> dict[str, object]:
    """统计退化三角面。

    两类情况记为退化：
        1. 三角面内部出现重复顶点编号；
        2. 三个点几何上几乎共线，叉积范数平方小于 eps_area。
    """
    vertices = _as_vertices(V)
    faces = _as_faces(F)
    _validate_face_indices(faces, len(vertices))

    eps_area = float(eps_area)
    count = 0
    sample_face_ids: list[int] = []

    for face_id, face in enumerate(faces):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        is_degenerate = a == b or b == c or a == c

        if not is_degenerate:
            pa, pb, pc = vertices[a], vertices[b], vertices[c]
            cross = np.cross(pb - pa, pc - pa)
            area2 = float(np.dot(cross, cross))
            is_degenerate = (not np.isfinite(area2)) or area2 <= eps_area

        if is_degenerate:
            count += 1
            if len(sample_face_ids) < 20:
                sample_face_ids.append(int(face_id))

    return {
        "count": int(count),
        "sample_face_ids": sample_face_ids,
    }


def count_duplicate_faces(F: np.ndarray) -> dict[str, object]:
    """统计重复三角面。

    面的朝向不参与判断：
        [0, 1, 2] 和 [2, 1, 0] 排序后都是 (0, 1, 2)，因此视为重复面。
    """
    faces = _as_faces(F)
    seen: dict[FaceKey, int] = {}

    count = 0
    sample_face_ids: list[int] = []

    for face_id, face in enumerate(faces):
        key = _face_key(face)

        if key in seen:
            count += 1
            if len(sample_face_ids) < 20:
                sample_face_ids.append(int(face_id))
            continue

        seen[key] = int(face_id)

    return {
        "count": int(count),
        "sample_face_ids": sample_face_ids,
    }


def connected_components_by_edges(F: np.ndarray) -> dict[str, object]:
    """按共享边关系统计 face edge-connected components。

    规则：
        两个三角面只要共享一条边，就属于同一个连通分量。
        这里先用 edge_faces 找相邻面，再用 BFS 给每个 face 标 component_id。
    """
    faces = _as_faces(F)
    n_faces = int(faces.shape[0])

    if n_faces == 0:
        return {
            "component_count": 0,
            "component_sizes": [],
            "face_component_id": [],
        }

    edge_faces = build_edge_faces(faces)
    neighbors = _build_face_neighbors(edge_faces, n_faces)

    face_component_id = [-1] * n_faces
    component_sizes: list[int] = []

    for start in range(n_faces):
        if face_component_id[start] != -1:
            continue

        comp_id = len(component_sizes)
        comp_size = _mark_component(
            start=start,
            comp_id=comp_id,
            neighbors=neighbors,
            face_component_id=face_component_id,
        )
        component_sizes.append(int(comp_size))

    return {
        "component_count": int(len(component_sizes)),
        "component_sizes": component_sizes,
        "face_component_id": [int(x) for x in face_component_id],
    }


def _build_face_neighbors(
    edge_faces: dict[Edge, list[int]],
    n_faces: int,
) -> list[set[int]]:
    """根据 edge -> faces 反查表，建立每个 face 的相邻 face 集合。"""
    neighbors: list[set[int]] = [set() for _ in range(n_faces)]

    for incident_faces in edge_faces.values():
        if len(incident_faces) < 2:
            continue

        for face_i in incident_faces:
            for face_j in incident_faces:
                if face_i != face_j:
                    neighbors[face_i].add(face_j)

    return neighbors


def _mark_component(
    *,
    start: int,
    comp_id: int,
    neighbors: list[set[int]],
    face_component_id: list[int],
) -> int:
    """从 start face 开始 BFS，把同一连通分量的 face 标成 comp_id。"""
    q: deque[int] = deque([start])
    face_component_id[start] = comp_id

    size = 0
    while q:
        face_id = q.popleft()
        size += 1

        for next_face in neighbors[face_id]:
            if face_component_id[next_face] == -1:
                face_component_id[next_face] = comp_id
                q.append(next_face)

    return size


def _sample_edges(edges: list[Edge]) -> list[list[int]]:
    """截取前 20 条边样本，并转换成 JSON 友好的 list。"""
    return [[int(a), int(b)] for a, b in edges[:20]]


def topology_summary(V: np.ndarray, F: np.ndarray) -> dict[str, object]:
    """返回当前 V/F 网格的基础拓扑诊断结果。

    主要统计：
        边界边：只被 1 个三角面使用的边；
        非流形边：被 3 个或更多三角面使用的边；
        重复面：顶点集合相同的三角面；
        退化面：重复顶点或面积接近 0 的三角面；
        连通分量：按共享边连接起来的 face component。
    """
    vertices = _as_vertices(V)
    faces = _as_faces(F)
    _validate_face_indices(faces, len(vertices))

    edge_faces = build_edge_faces(faces)
    boundary_edges, nonmanifold_edges, max_edge_incidence = _classify_edges(edge_faces)

    duplicates = count_duplicate_faces(faces)
    degenerates = count_degenerate_faces(vertices, faces)
    components = connected_components_by_edges(faces)
    nonmanifold_vertices = count_nonmanifold_vertices(faces)

    return {
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "edges": int(len(edge_faces)),
        "boundary_edges": int(len(boundary_edges)),
        "nonmanifold_edges": int(len(nonmanifold_edges)),
        "nonmanifold_vertices": int(nonmanifold_vertices["count"]),
        "max_edge_incidence": int(max_edge_incidence),
        "duplicate_faces": int(duplicates["count"]),
        "degenerate_faces": int(degenerates["count"]),
        "edge_component_count": int(components["component_count"]),
        "edge_component_sizes": [int(x) for x in components["component_sizes"]],
        "face_component_id": [int(x) for x in components["face_component_id"]],
        "sample_boundary_edges": _sample_edges(boundary_edges),
        "sample_nonmanifold_edges": _sample_edges(nonmanifold_edges),
        "sample_nonmanifold_vertices": nonmanifold_vertices["sample_vertex_ids"],
        "sample_duplicate_faces": [int(x) for x in duplicates["sample_face_ids"]],
        "sample_degenerate_faces": [int(x) for x in degenerates["sample_face_ids"]],
    }


def _classify_edges(edge_faces: dict[Edge, list[int]]) -> tuple[list[Edge], list[Edge], int]:
    """根据每条边关联的 face 数量，分类边界边和非流形边。"""
    boundary_edges: list[Edge] = []
    nonmanifold_edges: list[Edge] = []
    max_incidence = 0

    for edge, face_ids in edge_faces.items():
        incidence = len(face_ids)
        max_incidence = max(max_incidence, incidence)

        if incidence == 1:
            boundary_edges.append(edge)
        elif incidence > 2:
            nonmanifold_edges.append(edge)

    return boundary_edges, nonmanifold_edges, int(max_incidence)
