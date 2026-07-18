"""测试 TetGen 严格边界四面体流程。"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tetgen")

from mesh.io_obj import ObjMesh
from ops.pipeline_impl import repair_mesh_data
from pipeline import build_parser
from volume.boundary import boundary_lock_report
from volume.tetgen_tetra import (
    StrictTetrahedralizationOptions,
    tetrahedralize_strict,
)
from volume.tetra_mesh import TetraMesh


def _tetrahedron() -> ObjMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    )
    return ObjMesh(
        vertices,
        faces,
        face_object=["body"] * 4,
        face_group=["wall"] * 4,
        face_material=["steel"] * 4,
    )


def test_valid_solid_bypasses_rebuild_and_preserves_labels():
    source = _tetrahedron()

    output, report = repair_mesh_data(source, mode="solid")

    assert np.array_equal(output.V, source.V)
    assert np.array_equal(output.F, source.F)
    assert output.face_object == source.face_object
    assert output.face_group == source.face_group
    assert output.face_material == source.face_material
    assert "跳过修复和布尔重建" in report.warnings[0]


def test_strict_tetgen_keeps_boundary_bitwise_equal(tmp_path: Path):
    source = _tetrahedron()
    msh_path = tmp_path / "strict.msh"
    vtk_path = tmp_path / "strict.vtk"

    mesh, report = tetrahedralize_strict(
        source,
        msh_path,
        vtk_path,
        options=StrictTetrahedralizationOptions(),
        domain_name="steel",
    )

    assert report["success"] is True
    assert report["boundary_mode"] == "strict"
    assert report["boundary_lock"]["success"] is True
    assert report["boundary_lock"]["vertices_bitwise_equal"] is True
    assert report["boundary_lock"]["faces_equal_ignoring_orientation"] is True
    assert report["boundary_lock"]["maximum_boundary_vertex_displacement"] == 0.0
    assert report["boundary_lock"]["boundary_steiner_vertices"] == 0
    assert np.array_equal(mesh.V[: len(source.V)], source.V)
    assert msh_path.stat().st_size > 0
    assert vtk_path.stat().st_size > 0
    msh_text = msh_path.read_text(encoding="utf-8")
    assert '2 1 "wall"' in msh_text
    assert '3 2 "steel"' in msh_text
    assert report["boundary_physical_groups"] == ["wall"]
    assert report["boundary_labels"] == {
        "objects": ["body"],
        "groups": ["wall"],
        "materials": ["steel"],
    }
    lock = report["boundary_lock"]
    assert lock["input_vertex_sha256"] == lock["output_original_vertex_sha256"]
    assert lock["input_face_sha256"] == lock["output_boundary_face_sha256"]


def test_strict_tetgen_rejects_unreachable_size_without_changing_boundary(
    tmp_path: Path,
):
    source = _tetrahedron()

    _, report = tetrahedralize_strict(
        source,
        tmp_path / "small.msh",
        tmp_path / "small.vtk",
        options=StrictTetrahedralizationOptions(target_size=0.3),
    )

    assert report["success"] is False
    assert report["target_size_satisfied"] is False
    assert "tetra_size_above_target" in report["threshold_errors"]
    assert report["boundary_lock"]["success"] is True


def test_boundary_lock_rejects_moved_vertex():
    source = _tetrahedron()
    moved = source.V.copy()
    moved[0, 0] = 1e-9
    mesh = TetraMesh(moved, np.array([[0, 1, 2, 3]]), source.F)

    report = boundary_lock_report(mesh, source)

    assert report["success"] is False
    assert "boundary_vertices_changed" in report["errors"]
    assert report["maximum_boundary_vertex_displacement"] == pytest.approx(1e-9)


def test_cli_defaults_to_strict_boundary_mode():
    args = build_parser().parse_args(
        ["--input", "model.obj", "--output_dir", "out", "--tetrahedralize"]
    )

    assert args.tetra_mode == "strict"
