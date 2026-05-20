from __future__ import annotations

import json

import numpy as np
import pytest

from mesh.diagnostics import (
    build_edge_faces,
    connected_components_by_edges,
    count_degenerate_faces,
    count_duplicate_faces,
    topology_summary,
)


def test_build_edge_faces_uses_undirected_edges() -> None:
    F = np.array([
        [0, 1, 2],
        [2, 1, 3],
    ], dtype=int)

    edge_faces = build_edge_faces(F)

    assert edge_faces[(1, 2)] == [0, 1]
    assert edge_faces[(0, 1)] == [0]
    assert edge_faces[(1, 3)] == [1]


def test_clean_tetrahedron_is_closed_and_manifold() -> None:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    F = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ], dtype=int)

    summary = topology_summary(V, F)

    assert summary["vertices"] == 4
    assert summary["faces"] == 4
    assert summary["boundary_edges"] == 0
    assert summary["nonmanifold_edges"] == 0
    assert summary["max_edge_incidence"] == 2
    assert summary["duplicate_faces"] == 0
    assert summary["degenerate_faces"] == 0
    assert summary["edge_component_count"] == 1
    assert summary["edge_component_sizes"] == [4]


def test_single_triangle_has_three_boundary_edges() -> None:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    F = np.array([[0, 1, 2]], dtype=int)

    summary = topology_summary(V, F)

    assert summary["boundary_edges"] == 3
    assert summary["nonmanifold_edges"] == 0
    assert summary["max_edge_incidence"] == 1
    assert summary["edge_component_count"] == 1


def test_two_triangles_sharing_one_edge_are_one_component() -> None:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    F = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=int)

    summary = topology_summary(V, F)
    comps = connected_components_by_edges(F)

    assert summary["boundary_edges"] == 4
    assert summary["nonmanifold_edges"] == 0
    assert summary["max_edge_incidence"] == 2
    assert comps["component_count"] == 1
    assert comps["component_sizes"] == [2]
    assert comps["face_component_id"] == [0, 0]


def test_three_triangles_sharing_one_edge_detects_nonmanifold_edge() -> None:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    F = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [0, 1, 4],
    ], dtype=int)

    summary = topology_summary(V, F)

    assert summary["nonmanifold_edges"] == 1
    assert summary["max_edge_incidence"] == 3
    assert summary["sample_nonmanifold_edges"] == [[0, 1]]


def test_disconnected_triangles_form_two_edge_components() -> None:
    F = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ], dtype=int)

    comps = connected_components_by_edges(F)

    assert comps["component_count"] == 2
    assert comps["component_sizes"] == [1, 1]
    assert comps["face_component_id"] == [0, 1]


def test_count_duplicate_faces_ignores_orientation() -> None:
    F = np.array([
        [0, 1, 2],
        [2, 1, 0],
        [0, 2, 3],
    ], dtype=int)

    result = count_duplicate_faces(F)

    assert result["count"] == 1
    assert result["sample_face_ids"] == [1]


def test_count_degenerate_faces_detects_repeated_vertex_index() -> None:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    F = np.array([[0, 1, 1]], dtype=int)

    result = count_degenerate_faces(V, F)

    assert result["count"] == 1
    assert result["sample_face_ids"] == [0]


def test_count_degenerate_faces_detects_collinear_points() -> None:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    F = np.array([[0, 1, 2]], dtype=int)

    result = count_degenerate_faces(V, F)

    assert result["count"] == 1
    assert result["sample_face_ids"] == [0]


def test_empty_inputs_return_zero_summary() -> None:
    V = np.empty((0, 3), dtype=float)
    F = np.empty((0, 3), dtype=int)

    summary = topology_summary(V, F)
    comps = connected_components_by_edges(F)

    assert summary["vertices"] == 0
    assert summary["faces"] == 0
    assert summary["edges"] == 0
    assert summary["boundary_edges"] == 0
    assert summary["nonmanifold_edges"] == 0
    assert summary["max_edge_incidence"] == 0
    assert comps == {
        "component_count": 0,
        "component_sizes": [],
        "face_component_id": [],
    }


def test_topology_summary_is_json_serializable() -> None:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    F = np.array([[0, 1, 2]], dtype=np.int64)

    summary = topology_summary(V, F)

    json.dumps(summary, ensure_ascii=False)


def test_invalid_face_shape_raises_value_error() -> None:
    F = np.array([0, 1, 2], dtype=int)

    with pytest.raises(ValueError, match="F 必须是形状"):
        build_edge_faces(F)


def test_invalid_vertex_shape_raises_value_error() -> None:
    V = np.array([0.0, 1.0, 2.0])
    F = np.array([[0, 1, 2]], dtype=int)

    with pytest.raises(ValueError, match="V 必须是形状"):
        topology_summary(V, F)


def test_nan_vertex_raises_value_error() -> None:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, np.nan, 0.0],
        [0.0, 1.0, 0.0],
    ])
    F = np.array([[0, 1, 2]], dtype=int)

    with pytest.raises(ValueError, match="NaN 或 Inf"):
        topology_summary(V, F)