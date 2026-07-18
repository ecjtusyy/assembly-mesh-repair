"""三角网格自相交诊断。"""

from __future__ import annotations

import numpy as np


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = b - a
    second = c - a
    return float(first[0] * second[1] - first[1] * second[0])


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> bool:
    return bool(
        np.all(p >= np.minimum(a, b) - eps)
        and np.all(p <= np.maximum(a, b) + eps)
    )


def _segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    eps: float,
) -> bool:
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)
    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True
    return bool(
        abs(o1) <= eps and _on_segment(a, b, c, eps)
        or abs(o2) <= eps and _on_segment(a, b, d, eps)
        or abs(o3) <= eps and _on_segment(c, d, a, eps)
        or abs(o4) <= eps and _on_segment(c, d, b, eps)
    )


def _point_in_triangle_2d(point: np.ndarray, tri: np.ndarray, eps: float) -> bool:
    signs = np.array(
        [
            _orient2d(tri[0], tri[1], point),
            _orient2d(tri[1], tri[2], point),
            _orient2d(tri[2], tri[0], point),
        ]
    )
    return bool(np.all(signs >= -eps) or np.all(signs <= eps))


def _coplanar_intersection(
    first: np.ndarray,
    second: np.ndarray,
    normal: np.ndarray,
    eps: float,
) -> bool:
    drop_axis = int(np.argmax(np.abs(normal)))
    a = np.delete(first, drop_axis, axis=1)
    b = np.delete(second, drop_axis, axis=1)
    for edge_a in ((0, 1), (1, 2), (2, 0)):
        for edge_b in ((0, 1), (1, 2), (2, 0)):
            if _segments_intersect_2d(
                a[edge_a[0]], a[edge_a[1]], b[edge_b[0]], b[edge_b[1]], eps
            ):
                return True
    return _point_in_triangle_2d(a[0], b, eps) or _point_in_triangle_2d(
        b[0], a, eps
    )


def _segment_triangle_intersection(
    start: np.ndarray,
    end: np.ndarray,
    tri: np.ndarray,
    eps: float,
) -> bool:
    direction = end - start
    edge1 = tri[1] - tri[0]
    edge2 = tri[2] - tri[0]
    cross = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, cross))
    scale = float(np.linalg.norm(direction) * np.linalg.norm(edge1) * np.linalg.norm(edge2))
    if scale == 0.0 or abs(determinant) <= eps * scale:
        return False
    inverse = 1.0 / determinant
    offset = start - tri[0]
    u = inverse * float(np.dot(offset, cross))
    q = np.cross(offset, edge1)
    v = inverse * float(np.dot(direction, q))
    position = inverse * float(np.dot(edge2, q))
    return bool(
        u >= -eps
        and v >= -eps
        and u + v <= 1.0 + eps
        and position >= -eps
        and position <= 1.0 + eps
    )


def triangles_intersect(first: np.ndarray, second: np.ndarray, eps: float) -> bool:
    """判断两个已归一化的三角形是否相交。"""

    normal_a = np.cross(first[1] - first[0], first[2] - first[0])
    normal_b = np.cross(second[1] - second[0], second[2] - second[0])
    length_a = float(np.linalg.norm(normal_a))
    length_b = float(np.linalg.norm(normal_b))
    if length_a <= eps or length_b <= eps:
        return False

    distance_b = (second - first[0]) @ normal_a / length_a
    distance_a = (first - second[0]) @ normal_b / length_b
    if np.all(distance_b > eps) or np.all(distance_b < -eps):
        return False
    if np.all(distance_a > eps) or np.all(distance_a < -eps):
        return False

    parallel = float(np.linalg.norm(np.cross(normal_a, normal_b))) <= (
        eps * length_a * length_b
    )
    if parallel and np.max(np.abs(distance_b)) <= eps:
        return _coplanar_intersection(first, second, normal_a, eps)

    edges = ((0, 1), (1, 2), (2, 0))
    return any(
        _segment_triangle_intersection(first[a], first[b], second, eps)
        for a, b in edges
    ) or any(
        _segment_triangle_intersection(second[a], second[b], first, eps)
        for a, b in edges
    )


def self_intersection_report(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_tests: int = 2_000_000,
    sample_size: int = 20,
) -> dict[str, object]:
    """用包围盒扫描和三角形精确测试统计非相邻面自相交。"""

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if len(triangles) < 2:
        return {
            "count": 0,
            "tested_pairs": 0,
            "candidate_pairs": 0,
            "truncated": False,
            "sample_face_pairs": [],
        }

    minimum = points.min(axis=0)
    diagonal = float(np.linalg.norm(points.max(axis=0) - minimum))
    if diagonal == 0.0:
        diagonal = 1.0
    normalized = (points - minimum) / diagonal
    tri_points = normalized[triangles]
    box_min = tri_points.min(axis=1)
    box_max = tri_points.max(axis=1)
    axis = int(np.argmax(box_max.max(axis=0) - box_min.min(axis=0)))
    order = np.argsort(box_min[:, axis], kind="stable")
    eps = 1e-12
    adjacency_eps = 1e-7

    active: list[int] = []
    tested = 0
    candidates = 0
    intersections: list[list[int]] = []
    count = 0
    truncated = False
    for current_value in order:
        current = int(current_value)
        active = [
            face_id
            for face_id in active
            if box_max[face_id, axis] >= box_min[current, axis] - eps
        ]
        for other in active:
            if not (
                np.all(box_max[other] >= box_min[current] - eps)
                and np.all(box_max[current] >= box_min[other] - eps)
            ):
                continue
            candidates += 1
            if np.intersect1d(triangles[other], triangles[current]).size:
                continue
            vertex_distances = np.linalg.norm(
                tri_points[other][:, None, :] - tri_points[current][None, :, :],
                axis=2,
            )
            if np.any(vertex_distances <= adjacency_eps):
                continue
            if tested >= max_tests:
                truncated = True
                break
            tested += 1
            if triangles_intersect(tri_points[other], tri_points[current], eps):
                count += 1
                if len(intersections) < sample_size:
                    intersections.append([other, current])
        if truncated:
            break
        active.append(current)

    return {
        "count": int(count),
        "tested_pairs": int(tested),
        "candidate_pairs": int(candidates),
        "truncated": truncated,
        "sample_face_pairs": intersections,
    }
