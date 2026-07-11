"""测试 Gmsh 均匀细分。"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh")

from mesh.io_obj import ObjMesh, load_obj_data
from ops.pipeline_impl import repair_mesh_data


DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = [
    "土块加底土（相互穿透、存在体积重叠）.obj",
    "基坑1.0（存在多部分贴合和局部重叠）.obj",
    "基坑单元格未合并（存在多部分贴合和局部重叠）.obj",
    "整体元素土块底土（存在共享接触面）.obj",
]


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
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
    )
    return ObjMesh(vertices, faces)


def test_each_level_splits_one_triangle_into_four():
    output, report = repair_mesh_data(
        _tetrahedron(),
        mode="solid",
        uniform_refine_levels=2,
    )

    assert report.pre_refine_faces == 4
    assert report.post_refine_faces == 4 * 4**2
    assert len(output.F) == 64
    assert report.output_validation["success"] is True
    assert report.output_validation["is_volume"] is True
    assert np.isclose(
        report.output_validation["volume"],
        report.pre_refine_validation["volume"],
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("filename", MODELS)
def test_real_models_stay_manifold_after_uniform_refine(filename):
    output, report = repair_mesh_data(
        load_obj_data(DATA / filename),
        mode="solid",
        uniform_refine_levels=1,
    )

    topology = report.output_validation["topology"]
    assert report.post_refine_faces == report.pre_refine_faces * 4
    assert len(output.F) == report.post_refine_faces
    assert topology["nonmanifold_edges"] == 0
    assert topology["nonmanifold_vertices"] == 0
    assert report.output_validation["is_watertight"] is True
    assert report.output_validation["is_winding_consistent"] is True
    assert report.output_validation["is_volume"] is True
    assert np.isclose(
        report.output_validation["volume"],
        report.pre_refine_validation["volume"],
        rtol=1e-12,
        atol=1e-12,
    )
