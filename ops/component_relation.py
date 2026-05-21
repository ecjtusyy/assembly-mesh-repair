# -*- coding: utf-8 -*-
"""
先遍历所有的面，将各面包含的顶点编号按其所属的 component 进行去重归集；
随后再遍历这些 component，用归集好的顶点编号回查顶点表 V，提取真实 3D 坐标，从而计算出每个 component 的包围盒 bbox。
"""
from __future__ import annotations

import numpy as np


def _empty_bbox() -> dict[str, object]:
    """返回一个空 bbox。

    用在某个 component 没有顶点的情况。
    这种 component 不能计算 min/max，所以统一返回 None。
    """
    return {"min": None, "max": None, "diag": 0.0}


def _bbox(points: np.ndarray) -> dict[str, object]:
    """根据一组点坐标计算轴对齐包围盒。

    输入 points 是某个 component 用到的所有顶点坐标。
    输出包括：
    - min：x、y、z 三个方向的最小坐标；
    - max：x、y、z 三个方向的最大坐标；
    - diag：bbox 对角线长度，用来粗略表示这个 component 的空间尺度。
    """
    if points.size == 0:
        return _empty_bbox()

    lo = points.min(axis=0)
    hi = points.max(axis=0)

    return {
        "min": lo.astype(float).tolist(),
        "max": hi.astype(float).tolist(),
        "diag": float(np.linalg.norm(hi - lo)),
    }


def _bbox_overlap(a: dict[str, object], b: dict[str, object], eps: float = 0.0) -> bool:
    """判断两个 bbox 是否发生重叠。

    判断方法很简单：
    只要两个 bbox 在 x、y、z 三个方向上的区间都相交，
    就认为它们的 bbox 有重叠。

    注意：
    bbox 重叠只说明“可能有空间关系”，
    不能说明两个实体真的发生体积重叠。
    """
    if a["min"] is None or b["min"] is None:
        return False

    amin = np.asarray(a["min"], dtype=np.float64)
    amax = np.asarray(a["max"], dtype=np.float64)
    bmin = np.asarray(b["min"], dtype=np.float64)
    bmax = np.asarray(b["max"], dtype=np.float64)

    return bool(np.all(amax + eps >= bmin) and np.all(bmax + eps >= amin))


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, comp_ids: list[int]) -> None:
    """检查输入数据是否合法。

    这里主要检查四件事：
    1. V 必须是 n×3 的顶点坐标数组；
    2. F 必须是 m×3 的三角面数组；
    3. face_component_id 的数量必须等于面数；
    4. F 里的顶点编号不能越界，component id 不能为负数。
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"V 必须是 (n, 3) 数组，当前 shape={vertices.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"F 必须是 (m, 3) 三角面数组，当前 shape={faces.shape}")

    if faces.shape[0] != len(comp_ids):
        raise ValueError(
            f"face_component_id 数量必须等于面数：faces={faces.shape[0]}, ids={len(comp_ids)}"
        )

    if faces.size > 0:
        if faces.min() < 0 or faces.max() >= vertices.shape[0]:
            raise ValueError("F 中存在越界顶点索引")

    if comp_ids and min(comp_ids) < 0:
        raise ValueError("component id 不能为负数")


def classify_component_relations(
    V: np.ndarray,
    F: np.ndarray,
    face_component_id: list[int] | np.ndarray,
) -> dict[str, object]:
    """统计装配体中各个 component 的空间关系。

    输入：
    - V：所有顶点坐标；
    - F：所有三角面，每一行是一个三角形的 3 个顶点编号；
    - face_component_id：每个面的 component 编号。

    核心思路：
    1. 遍历每个面；
    2. 根据这个面的 component id，把它归到对应 component；
    3. 收集每个 component 用到的所有顶点；
    4. 用这些顶点计算每个 component 的 bbox；
    5. 两两比较 bbox，找出可能存在空间重叠关系的 component 对。
    """
    vertices = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)
    comp_ids = [int(x) for x in face_component_id]

    _validate_mesh(vertices, faces, comp_ids)

    if not comp_ids:
        return {
            "component_count": 0,
            "component_sizes": [],
            "bbox_per_component": [],
            "bbox_overlap_pairs": [],
            "suspected_multi_body_assembly": False,
            "recommended_policy_choices": ["report_only"],
        }

    component_count = max(comp_ids) + 1
    component_sizes = [0] * component_count
    component_vertices: list[set[int]] = [set() for _ in range(component_count)]

    # 遍历每个面：统计每个 component 有多少个面、用了哪些顶点。
    for face_id, comp_id in enumerate(comp_ids):
        component_sizes[comp_id] += 1
        component_vertices[comp_id].update(int(v_id) for v_id in faces[face_id])

    bbox_per_component: list[dict[str, object]] = []

    # 对每个 component：取出它用到的顶点坐标，然后计算 bbox。
    for comp_id, vertex_ids in enumerate(component_vertices):
        points = (
            vertices[sorted(vertex_ids)]
            if vertex_ids
            else np.empty((0, 3), dtype=np.float64)
        )

        bbox_per_component.append(
            {
                "component": int(comp_id),
                "faces": int(component_sizes[comp_id]),
                "vertices": int(len(vertex_ids)),
                "bbox": _bbox(points),
            }
        )

    bbox_overlap_pairs: list[dict[str, object]] = []

    # 两两比较 component 的 bbox，记录可能有空间重叠关系的 component 对。
    for i in range(component_count):
        for j in range(i + 1, component_count):
            if _bbox_overlap(
                bbox_per_component[i]["bbox"],
                bbox_per_component[j]["bbox"],
            ):
                bbox_overlap_pairs.append(
                    {
                        "a": int(i),
                        "b": int(j),
                        "relation": "bbox_overlap_uncertain",
                    }
                )

    return {
        "component_count": int(component_count),
        "component_sizes": [int(size) for size in component_sizes],
        "bbox_per_component": bbox_per_component,
        "bbox_overlap_pairs": bbox_overlap_pairs,
        "suspected_multi_body_assembly": bool(component_count > 1),
        "recommended_policy_choices": [
            "report_only",
            "merge_as_one_body",
            "keep_as_contact_bodies",
            "external_boundary_only",
        ],
    }