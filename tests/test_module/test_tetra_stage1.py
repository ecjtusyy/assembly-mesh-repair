"""测试第一阶段的真实四面体生成和质量验收。"""

from pathlib import Path

import numpy as np
import pytest

try:
    import gmsh  # noqa: F401
except (ImportError, OSError):
    pytest.skip("Gmsh 或系统动态库不可用", allow_module_level=True)

from mesh.io_obj import ObjMesh
from ops.pipeline_impl import repair_mesh_data
from volume.gmsh_tetra import TetrahedralizationOptions, tetrahedralize
from volume.quality import mean_ratio_quality, quality_report
from volume.tetra_mesh import TetraMesh


DATA = Path(__file__).resolve().parents[1] / "data"


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
    return ObjMesh(vertices, faces)


def test_regular_tetra_mean_ratio_is_bounded():
    source = _tetrahedron()
    mesh = TetraMesh(source.V, np.array([[0, 1, 2, 3]]), source.F)

    quality = mean_ratio_quality(mesh)

    assert 0.0 < quality[0] <= 1.0


def test_inverted_tetra_fails_hard_validation():
    source = _tetrahedron()
    mesh = TetraMesh(source.V, np.array([[0, 2, 1, 3]]), source.F)

    report, _ = quality_report(
        mesh,
        source,
        min_quality=0.0,
        max_relative_deviation=1.0,
    )

    assert report["hard_valid"] is False
    assert report["inverted_tetrahedra"] == 1
    assert "inverted_tetrahedra" in report["errors"]


def test_gmsh_generates_valid_volume_and_files(tmp_path: Path):
    surface, _ = repair_mesh_data(_tetrahedron(), mode="solid")
    msh_path = tmp_path / "volume.msh"
    vtk_path = tmp_path / "quality.vtk"

    mesh, report = tetrahedralize(
        surface,
        msh_path,
        vtk_path,
        options=TetrahedralizationOptions(target_size=0.3),
    )

    assert len(mesh.T) > 1
    assert report["success"] is True
    assert report["hard_valid"] is True
    assert report["inverted_tetrahedra"] == 0
    assert report["zero_volume_tetrahedra"] == 0
    assert report["duplicate_tetrahedra"] == 0
    assert report["missing_boundary_faces"] == 0
    assert report["extra_boundary_faces"] == 0
    assert report["mean_ratio"]["minimum"] >= 0.05
    assert report["geometry_deviation"]["relative_max"] <= 1e-8
    assert report["volume_relative_error"] <= 1e-12
    assert msh_path.stat().st_size > 0
    assert vtk_path.stat().st_size > 0


def test_real_shared_surface_model_tetrahedralizes(tmp_path: Path):
    from mesh.io_obj import load_obj_data

    source = load_obj_data(DATA / "整体元素土块底土（存在共享接触面）.obj")
    surface, _ = repair_mesh_data(source, mode="solid")

    _, report = tetrahedralize(
        surface,
        tmp_path / "real.msh",
        tmp_path / "real.vtk",
    )

    assert report["success"] is True
    assert report["hard_valid"] is True
    assert report["mean_ratio"]["below_threshold"] == 0
    assert report["geometry_deviation"]["relative_max"] <= 1e-6
    assert report["volume_relative_error"] <= 1e-6
