"""三角表面网格质量指标。"""

from __future__ import annotations

import numpy as np


_REFERENCE_INVERSE = np.linalg.inv(
    np.array(
        [
            [1.0, 0.5],
            [0.0, np.sqrt(3.0) / 2.0],
        ],
        dtype=np.float64,
    )
)


def triangle_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回最小角、mean-ratio 和等边参考条件数。"""

    points = np.asarray(vertices, dtype=np.float64)[faces]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    third = points[:, 2] - points[:, 1]

    edge_squared = np.stack(
        [
            np.einsum("ij,ij->i", first, first),
            np.einsum("ij,ij->i", third, third),
            np.einsum("ij,ij->i", second, second),
        ],
        axis=1,
    )
    pairs = ((first, second), (-first, third), (-second, -third))
    cosines: list[np.ndarray] = []
    for left, right in pairs:
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cosine = np.divide(
            np.einsum("ij,ij->i", left, right),
            denominator,
            out=np.ones(len(faces), dtype=np.float64),
            where=denominator > 0.0,
        )
        cosines.append(cosine)
    angles = np.degrees(
        np.arccos(np.clip(np.stack(cosines, axis=1), -1.0, 1.0))
    )
    minimum_angle = angles.min(axis=1)

    double_area = np.linalg.norm(np.cross(first, second), axis=1)
    squared_sum = edge_squared.sum(axis=1)
    mean_ratio = np.divide(
        2.0 * np.sqrt(3.0) * double_area,
        squared_sum,
        out=np.zeros_like(double_area),
        where=squared_sum > 0.0,
    )

    physical = np.stack([first, second], axis=2)
    jacobian = physical @ _REFERENCE_INVERSE
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = np.divide(
        singular_values[:, 0],
        singular_values[:, 1],
        out=np.full(len(faces), np.inf, dtype=np.float64),
        where=singular_values[:, 1] > 0.0,
    )
    return minimum_angle, mean_ratio, condition


def surface_quality_report(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_angle: float = 15.0,
    min_mean_ratio: float = 0.2,
    max_condition: float = 10.0,
) -> dict[str, object]:
    """生成表面质量报告。"""

    angles, mean_ratio, condition = triangle_quality(vertices, faces)
    bad = (
        (angles < min_angle)
        | (mean_ratio < min_mean_ratio)
        | (condition > max_condition)
    )
    bad_ids = np.flatnonzero(bad)
    return {
        "success": not len(bad_ids),
        "faces": int(len(faces)),
        "thresholds": {
            "minimum_angle_degrees": float(min_angle),
            "minimum_mean_ratio": float(min_mean_ratio),
            "maximum_shape_condition": float(max_condition),
        },
        "minimum_angle_degrees": {
            "minimum": float(angles.min(initial=180.0)),
            "p01": float(np.quantile(angles, 0.01)) if len(angles) else 180.0,
            "p05": float(np.quantile(angles, 0.05)) if len(angles) else 180.0,
        },
        "mean_ratio": {
            "minimum": float(mean_ratio.min(initial=1.0)),
            "p01": float(np.quantile(mean_ratio, 0.01)) if len(mean_ratio) else 1.0,
            "p05": float(np.quantile(mean_ratio, 0.05)) if len(mean_ratio) else 1.0,
        },
        "shape_condition": {
            "maximum": float(condition.max(initial=1.0)),
            "p95": float(np.quantile(condition, 0.95)) if len(condition) else 1.0,
            "p99": float(np.quantile(condition, 0.99)) if len(condition) else 1.0,
        },
        "below_minimum_angle": int(np.count_nonzero(angles < min_angle)),
        "below_minimum_mean_ratio": int(
            np.count_nonzero(mean_ratio < min_mean_ratio)
        ),
        "above_maximum_condition": int(
            np.count_nonzero(condition > max_condition)
        ),
        "bad_faces": int(len(bad_ids)),
        "sample_bad_face_ids": bad_ids[:20].astype(int).tolist(),
    }
