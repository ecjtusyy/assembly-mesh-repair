"""严格边界锁定验收。"""

from __future__ import annotations

import hashlib

import numpy as np

from mesh.io_obj import ObjMesh
from volume.tetra_mesh import TetraMesh


def triangle_mean_ratio(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """返回 0 到 1 的三角形 mean-ratio，正三角形为 1。"""

    points = np.asarray(vertices, dtype=np.float64)[faces]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    third = points[:, 2] - points[:, 1]
    double_area = np.linalg.norm(np.cross(first, second), axis=1)
    edge_squared = (
        np.einsum("ij,ij->i", first, first)
        + np.einsum("ij,ij->i", second, second)
        + np.einsum("ij,ij->i", third, third)
    )
    return np.divide(
        2.0 * np.sqrt(3.0) * double_area,
        edge_squared,
        out=np.zeros_like(double_area),
        where=edge_squared > 0.0,
    )


def boundary_quality_report(source: ObjMesh) -> dict[str, object]:
    quality = triangle_mean_ratio(source.V, source.F)
    return {
        "faces": int(len(source.F)),
        "minimum": float(quality.min()),
        "p01": float(np.quantile(quality, 0.01)),
        "p05": float(np.quantile(quality, 0.05)),
        "median": float(np.median(quality)),
        "maximum": float(quality.max()),
        "below_1e-6": int(np.count_nonzero(quality < 1e-6)),
        "sample_below_1e-6": np.flatnonzero(quality < 1e-6)[:20]
        .astype(int)
        .tolist(),
    }


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    rows = np.sort(np.asarray(values, dtype=np.int64), axis=1)
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def boundary_lock_report(mesh: TetraMesh, source: ObjMesh) -> dict[str, object]:
    """逐点逐面检查输出边界是否就是输入边界。"""

    input_count = int(len(source.V))
    enough_nodes = len(mesh.V) >= input_count
    prefix_equal = bool(
        enough_nodes and np.array_equal(mesh.V[:input_count], source.V)
    )
    if enough_nodes and input_count:
        displacement = np.linalg.norm(
            mesh.V[:input_count] - source.V,
            axis=1,
        )
        maximum_displacement = float(displacement.max(initial=0.0))
    else:
        maximum_displacement = float("inf")

    boundary_ids = np.unique(mesh.boundary_faces)
    expected_ids = np.unique(source.F)
    extra_ids = boundary_ids[boundary_ids >= input_count]
    original_boundary_ids = boundary_ids[boundary_ids < input_count]
    missing_ids = np.setdiff1d(expected_ids, original_boundary_ids)
    faces_equal = bool(
        len(mesh.boundary_faces) == len(source.F)
        and np.array_equal(
            _sorted_rows(mesh.boundary_faces),
            _sorted_rows(source.F),
        )
    )

    errors: list[str] = []
    if not prefix_equal:
        errors.append("boundary_vertices_changed")
    if len(extra_ids):
        errors.append("boundary_steiner_vertices")
    if len(missing_ids):
        errors.append("boundary_vertices_missing")
    if not faces_equal:
        errors.append("boundary_faces_changed")

    return {
        "success": not errors,
        "errors": errors,
        "vertices_bitwise_equal": prefix_equal,
        "faces_equal_ignoring_orientation": faces_equal,
        "maximum_boundary_vertex_displacement": maximum_displacement,
        "input_boundary_vertices": int(len(expected_ids)),
        "output_boundary_vertices": int(len(boundary_ids)),
        "boundary_steiner_vertices": int(len(extra_ids)),
        "missing_boundary_vertices": int(len(missing_ids)),
        "sample_boundary_steiner_ids": extra_ids[:20].astype(int).tolist(),
        "sample_missing_boundary_ids": missing_ids[:20].astype(int).tolist(),
        "input_vertex_sha256": _array_hash(source.V),
        "output_original_vertex_sha256": _array_hash(mesh.V[:input_count]),
        "input_face_sha256": _array_hash(_sorted_rows(source.F)),
        "output_boundary_face_sha256": _array_hash(
            _sorted_rows(mesh.boundary_faces)
        ),
    }
