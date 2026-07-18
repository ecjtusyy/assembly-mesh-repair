"""四面体有效性、质量和几何偏差验收。"""

from __future__ import annotations

import numpy as np
import trimesh

from mesh.io_obj import ObjMesh
from volume.tetra_mesh import TetraMesh


_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def signed_six_volumes(mesh: TetraMesh) -> np.ndarray:
    points = mesh.V[mesh.T]
    return np.einsum(
        "ij,ij->i",
        points[:, 1] - points[:, 0],
        np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]),
    )


def mean_ratio_quality(mesh: TetraMesh, six_volumes: np.ndarray | None = None) -> np.ndarray:
    """返回 0 到 1 的四面体 mean-ratio，正四面体为 1。"""

    six = signed_six_volumes(mesh) if six_volumes is None else six_volumes
    points = mesh.V[mesh.T]
    edge_squared = np.zeros(len(mesh.T), dtype=np.float64)
    for first, second in _TET_EDGES:
        delta = points[:, second] - points[:, first]
        edge_squared += np.einsum("ij,ij->i", delta, delta)
    volume = np.abs(six) / 6.0
    numerator = 12.0 * np.power(3.0 * volume, 2.0 / 3.0)
    return np.divide(
        numerator,
        edge_squared,
        out=np.zeros_like(numerator),
        where=edge_squared > 0.0,
    )


def _derived_boundary_faces(tets: np.ndarray) -> np.ndarray:
    faces = np.vstack(
        [
            tets[:, [0, 2, 1]],
            tets[:, [0, 1, 3]],
            tets[:, [1, 2, 3]],
            tets[:, [2, 0, 3]],
        ]
    )
    keys = np.sort(faces, axis=1)
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    return unique[counts == 1]


def _boundary_mismatch(mesh: TetraMesh) -> tuple[int, int]:
    derived = {tuple(row) for row in _derived_boundary_faces(mesh.T)}
    exported = {tuple(row) for row in np.sort(mesh.boundary_faces, axis=1)}
    return len(derived - exported), len(exported - derived)


def _closest_distances(surface: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    values: list[np.ndarray] = []
    for start in range(0, len(points), 1000):
        _, distance, _ = trimesh.proximity.closest_point_naive(
            surface, points[start : start + 1000]
        )
        values.append(np.asarray(distance, dtype=np.float64))
    return np.concatenate(values)


def geometry_deviation(mesh: TetraMesh, source: ObjMesh) -> dict[str, float]:
    """计算源表面与体网格边界的双向离散 Hausdorff 近似。"""

    source_surface = trimesh.Trimesh(source.V, source.F, process=False)
    boundary_surface = trimesh.Trimesh(
        mesh.V, mesh.boundary_faces, process=False
    )
    boundary_vertex_ids = np.unique(mesh.boundary_faces)
    first = _closest_distances(source_surface, mesh.V[boundary_vertex_ids])
    second = _closest_distances(boundary_surface, source.V)
    combined = np.concatenate([first, second])
    diagonal = float(np.linalg.norm(source.V.max(axis=0) - source.V.min(axis=0)))
    absolute_max = float(combined.max(initial=0.0))
    return {
        "max": absolute_max,
        "rms": float(np.sqrt(np.mean(combined * combined))) if len(combined) else 0.0,
        "relative_max": absolute_max / diagonal if diagonal > 0.0 else 0.0,
        "sample_count": int(len(combined)),
    }


def quality_report(
    mesh: TetraMesh,
    source: ObjMesh,
    *,
    min_quality: float,
    max_relative_deviation: float,
    max_relative_volume_error: float = 1e-6,
) -> tuple[dict[str, object], np.ndarray]:
    """生成硬有效性和软质量阈值分开的验收报告。"""

    six = signed_six_volumes(mesh)
    quality = mean_ratio_quality(mesh, six)
    diagonal = float(np.linalg.norm(mesh.V.max(axis=0) - mesh.V.min(axis=0)))
    volume_tolerance = max(diagonal**3 * 1e-14, np.finfo(float).tiny)
    inverted = np.flatnonzero(six < -6.0 * volume_tolerance)
    zero = np.flatnonzero(np.abs(six) <= 6.0 * volume_tolerance)
    _, counts = np.unique(np.sort(mesh.T, axis=1), axis=0, return_counts=True)
    duplicate_count = int(np.sum(counts - 1))
    missing_boundary, extra_boundary = _boundary_mismatch(mesh)
    deviation = geometry_deviation(mesh, source)
    tet_volume = float(np.sum(np.abs(six)) / 6.0)
    surface_volume = abs(
        float(trimesh.Trimesh(source.V, source.F, process=False).volume)
    )
    volume_relative_error = (
        abs(tet_volume - surface_volume) / surface_volume
        if surface_volume > 0.0
        else float("inf")
    )

    hard_errors: list[str] = []
    if len(inverted):
        hard_errors.append("inverted_tetrahedra")
    if len(zero):
        hard_errors.append("zero_volume_tetrahedra")
    if duplicate_count:
        hard_errors.append("duplicate_tetrahedra")
    if missing_boundary or extra_boundary:
        hard_errors.append("boundary_mismatch")

    threshold_errors: list[str] = []
    quality_min = float(quality.min())
    if quality_min < min_quality:
        threshold_errors.append("tetra_quality_below_threshold")
    if float(deviation["relative_max"]) > max_relative_deviation:
        threshold_errors.append("geometry_deviation_above_threshold")
    if volume_relative_error > max_relative_volume_error:
        threshold_errors.append("volume_error_above_threshold")

    report: dict[str, object] = {
        "success": not hard_errors and not threshold_errors,
        "hard_valid": not hard_errors,
        "errors": hard_errors + threshold_errors,
        "hard_errors": hard_errors,
        "threshold_errors": threshold_errors,
        "vertices": int(len(mesh.V)),
        "tetrahedra": int(len(mesh.T)),
        "boundary_faces": int(len(mesh.boundary_faces)),
        "inverted_tetrahedra": int(len(inverted)),
        "zero_volume_tetrahedra": int(len(zero)),
        "duplicate_tetrahedra": duplicate_count,
        "missing_boundary_faces": int(missing_boundary),
        "extra_boundary_faces": int(extra_boundary),
        "mean_ratio": {
            "minimum": quality_min,
            "p01": float(np.quantile(quality, 0.01)),
            "p05": float(np.quantile(quality, 0.05)),
            "median": float(np.median(quality)),
            "maximum": float(quality.max(initial=0.0)),
            "threshold": float(min_quality),
            "below_threshold": int(np.count_nonzero(quality < min_quality)),
        },
        "geometry_deviation": {
            **deviation,
            "relative_threshold": float(max_relative_deviation),
        },
        "surface_volume": surface_volume,
        "tetra_volume": tet_volume,
        "volume_relative_error": float(volume_relative_error),
        "volume_relative_error_threshold": float(max_relative_volume_error),
        "sample_inverted_ids": inverted[:20].astype(int).tolist(),
        "sample_zero_volume_ids": zero[:20].astype(int).tolist(),
    }
    return report, quality
