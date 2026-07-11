"""测试仓库中的四个真实模型。"""

from pathlib import Path

import pytest

from mesh.io_obj import load_obj_data
from ops.pipeline_impl import repair_mesh_data


DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = [
    "土块加底土（相互穿透、存在体积重叠）.obj",
    "基坑1.0（存在多部分贴合和局部重叠）.obj",
    "基坑单元格未合并（存在多部分贴合和局部重叠）.obj",
    "整体元素土块底土（存在共享接触面）.obj",
]


@pytest.mark.parametrize("filename", MODELS)
def test_real_model_solid_union(filename):
    output, report = repair_mesh_data(load_obj_data(DATA / filename), mode="solid")
    validation = report.output_validation

    assert report.approximate_rebuild is False
    assert validation["success"] is True
    assert validation["is_watertight"] is True
    assert validation["is_winding_consistent"] is True
    assert validation["is_volume"] is True
    assert len(output.F) > 0


@pytest.mark.parametrize("filename", MODELS)
def test_real_model_assembly_keeps_parts(filename):
    output, report = repair_mesh_data(load_obj_data(DATA / filename), mode="assembly")
    expected = int(report.input_topology["edge_component_count"])

    assert len(report.part_reports) == expected
    assert len(set(output.face_object)) == expected
    assert report.output_validation["success"] is True


def test_approximate_rebuild_is_explicit_fallback():
    mesh = load_obj_data(DATA / "tri_cross.obj")
    output, report = repair_mesh_data(
        mesh,
        mode="solid",
        approximate_rebuild=True,
        rebuild_resolution=5000,
    )

    assert report.approximate_rebuild is True
    assert report.output_validation["is_volume"] is True
    assert len(output.F) > len(mesh.F)
