"""测试三角表面自相交诊断。"""

import numpy as np

from mesh.intersections import self_intersection_report
from mesh.io_obj import ObjMesh
from ops.validation import validate_mesh


def test_crossing_triangles_are_reported():
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.5, -1.0],
            [0.0, -0.5, 1.0],
            [0.0, 0.5, 0.0],
        ]
    )
    report = self_intersection_report(vertices, np.array([[0, 1, 2], [3, 4, 5]]))

    assert report["count"] == 1
    assert report["sample_face_pairs"] == [[0, 1]]
    assert report["truncated"] is False


def test_adjacent_triangles_are_not_self_intersections():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    report = self_intersection_report(vertices, np.array([[0, 1, 2], [1, 3, 2]]))

    assert report["count"] == 0
    assert report["adjacent_count"] == 0


def test_coplanar_fold_over_shared_edge_is_reported():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3]])

    report = self_intersection_report(vertices, faces)

    assert report["count"] == 1
    assert report["adjacent_count"] == 1
    assert report["sample_face_pairs"] == [[0, 1]]


def test_overlap_beyond_a_shared_vertex_is_reported():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, 0.0],
            [0.8, 0.1, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 3, 4]])

    report = self_intersection_report(vertices, faces)

    assert report["count"] == 1
    assert report["adjacent_count"] == 1


def test_noncoplanar_faces_may_share_an_edge():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3]])

    report = self_intersection_report(vertices, faces)

    assert report["count"] == 0
    assert report["adjacent_count"] == 0


def test_validation_names_adjacent_face_overlap():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
        ]
    )
    mesh = ObjMesh(vertices, np.array([[0, 1, 2], [1, 0, 3]]))

    report = validate_mesh(
        mesh,
        require_volume=False,
        check_self_intersections=True,
    )

    assert report["success"] is False
    assert "adjacent_face_overlaps" in report["errors"]
