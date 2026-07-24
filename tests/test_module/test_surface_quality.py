"""测试表面三角形质量指标。"""

import numpy as np

from mesh.surface_quality import surface_quality_report, triangle_quality


def test_equilateral_triangle_has_unit_reference_quality():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
        ]
    )

    angle, mean_ratio, condition = triangle_quality(
        vertices,
        np.array([[0, 1, 2]]),
    )

    np.testing.assert_allclose(angle[0], 60.0, atol=1e-12)
    np.testing.assert_allclose(mean_ratio[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(condition[0], 1.0, atol=1e-12)


def test_skinny_triangle_fails_all_default_thresholds():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 1.0, 0.0]]
    )

    report = surface_quality_report(vertices, np.array([[0, 1, 2]]))

    assert report["success"] is False
    assert report["below_minimum_angle"] == 1
    assert report["below_minimum_mean_ratio"] == 1
    assert report["above_maximum_condition"] == 1
