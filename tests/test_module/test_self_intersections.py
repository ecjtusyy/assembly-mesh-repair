"""测试三角表面自相交诊断。"""

import numpy as np

from mesh.intersections import self_intersection_report


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
