"""测试局部表面修复。"""

import numpy as np

from mesh.io_obj import ObjMesh
from ops.pipeline_impl import repair_mesh_data


def _three_faces_on_one_edge() -> ObjMesh:
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    F = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    return ObjMesh(V, F)


def test_surface_splits_nonmanifold_edge():
    output, report = repair_mesh_data(_three_faces_on_one_edge(), mode="surface")
    topology = report.output_validation["topology"]

    assert report.status == "success"
    assert topology["nonmanifold_edges"] == 0
    assert topology["nonmanifold_vertices"] == 0
    assert len(output.F) == 3


def test_surface_repair_is_idempotent():
    first, _ = repair_mesh_data(_three_faces_on_one_edge(), mode="surface")
    second, _ = repair_mesh_data(first, mode="surface")

    assert len(first.V) == len(second.V)
    assert len(first.F) == len(second.F)


def test_only_surface_mode_reports_open_surface_limitations():
    mesh = _three_faces_on_one_edge()

    _, surface_report = repair_mesh_data(mesh, mode="surface")
    _, assembly_report = repair_mesh_data(mesh, mode="assembly")

    assert any("surface 模式允许边界" in item for item in surface_report.warnings)
    assert all("surface 模式" not in item for item in assembly_report.warnings)


def test_small_hole_is_filled():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3]])

    output, report = repair_mesh_data(
        ObjMesh(V, F),
        mode="surface",
        fill_holes=True,
    )

    assert len(output.F) == 4
    assert report.output_validation["is_watertight"] is True


def test_surface_connects_t_junction():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )
    F = np.array([[0, 1, 2], [0, 3, 4], [3, 1, 4]])

    output, report = repair_mesh_data(ObjMesh(V, F), mode="surface")

    assert len(output.F) == 4
    assert report.output_validation["topology"]["nonmanifold_vertices"] == 0
    assert report.part_reports[0]["changes"][
        "split_cross_component_t_junction_faces"
    ] == 1
