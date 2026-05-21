# -*- coding: utf-8 -*-
"""测试装配体 component 关系诊断函数。"""

import numpy as np
import pytest

from ops.component_relation import classify_component_relations


def test_empty_mesh_returns_empty_report():
    """测试空 mesh：没有面时，应返回空 component 诊断结果。"""

    V = np.empty((0, 3), dtype=np.float64)
    F = np.empty((0, 3), dtype=np.int64)

    report = classify_component_relations(V, F, [])

    assert report["component_count"] == 0
    assert report["component_sizes"] == []
    assert report["bbox_per_component"] == []
    assert report["bbox_overlap_pairs"] == []
    assert report["suspected_multi_body_assembly"] is False
    assert report["recommended_policy_choices"] == ["report_only"]


def test_single_component_bbox():
    """测试单个 component：应正确统计面数、顶点数和 bbox。"""

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array([[0, 1, 2]], dtype=np.int64)

    report = classify_component_relations(V, F, [0])
    bbox_info = report["bbox_per_component"][0]

    assert report["component_count"] == 1
    assert report["component_sizes"] == [1]
    assert report["suspected_multi_body_assembly"] is False

    assert bbox_info["component"] == 0
    assert bbox_info["faces"] == 1
    assert bbox_info["vertices"] == 3
    assert bbox_info["bbox"]["min"] == [0.0, 0.0, 0.0]
    assert bbox_info["bbox"]["max"] == [2.0, 3.0, 0.0]
    assert bbox_info["bbox"]["diag"] == pytest.approx(np.sqrt(13.0))


def test_two_components_without_bbox_overlap():
    """测试两个分离 component：bbox 不相交时，不应产生 overlap pair。"""

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.int64,
    )

    report = classify_component_relations(V, F, [0, 1])

    assert report["component_count"] == 2
    assert report["component_sizes"] == [1, 1]
    assert report["bbox_overlap_pairs"] == []
    assert report["suspected_multi_body_assembly"] is True


def test_two_components_with_bbox_overlap():
    """测试两个 component 的 bbox 相交：应记录 bbox_overlap_uncertain。"""

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.int64,
    )

    report = classify_component_relations(V, F, [0, 1])

    assert report["component_count"] == 2
    assert report["bbox_overlap_pairs"] == [
        {
            "a": 0,
            "b": 1,
            "relation": "bbox_overlap_uncertain",
        }
    ]


def test_component_collects_unique_vertices_from_multiple_faces():
    """测试同一 component 有多个面时，顶点编号应去重后再统计 bbox。"""

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )

    report = classify_component_relations(V, F, [0, 0])
    bbox_info = report["bbox_per_component"][0]

    assert report["component_count"] == 1
    assert report["component_sizes"] == [2]
    assert bbox_info["vertices"] == 4
    assert bbox_info["bbox"]["min"] == [0.0, 0.0, 0.0]
    assert bbox_info["bbox"]["max"] == [1.0, 1.0, 0.0]


def test_face_component_id_count_must_match_face_count():
    """测试 face_component_id 数量错误时，应直接报错。"""

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="face_component_id 数量必须等于面数"):
        classify_component_relations(V, F, [0, 1])


def test_negative_component_id_is_invalid():
    """测试负数 component id：这种分组编号没有意义，应直接报错。"""

    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="component id 不能为负数"):
        classify_component_relations(V, F, [-1])