"""测试仓库中的四个真实模型。"""

from pathlib import Path

import numpy as np
import pytest

from mesh.io_obj import load_obj_data
from ops.pipeline_impl import (
    _canonicalize_pre_union_coordinates,
    repair_mesh_data,
)


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
    try:
        import point_cloud_utils  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"point-cloud-utils 当前环境不可用：{exc}")

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


def test_pre_union_coordinate_canonicalization_preserves_bbox():
    mesh = load_obj_data(DATA / "基坑1.0（存在多部分贴合和局部重叠）.obj")

    output, report = _canonicalize_pre_union_coordinates(mesh, 3e-8)

    assert report["enabled"] is True
    assert report["bounding_box_preserved"] is True
    assert report["moved_vertices"] > 0
    assert report["maximum_relative_displacement"] <= 3e-8
    assert np.array_equal(output.V.min(axis=0), mesh.V.min(axis=0))
    assert np.array_equal(output.V.max(axis=0), mesh.V.max(axis=0))


def test_pit_pre_union_canonicalization_removes_numeric_seams():
    source = load_obj_data(DATA / "基坑1.0（存在多部分贴合和局部重叠）.obj")

    output, report = repair_mesh_data(
        source,
        mode="solid",
        pre_union_snap_relative=3e-8,
    )

    near_x = np.unique(output.V[:, 0])
    near_x = near_x[(near_x > 1.49998) & (near_x < 1.50002)]
    canonicalization = report.coordinate_canonicalization

    assert report.output_validation["success"] is True
    assert canonicalization["enabled"] is True
    assert canonicalization["bounding_box_preserved"] is True
    assert near_x.tolist() == [1.5]


def test_pre_union_coordinate_tolerance_cannot_be_negative():
    source = load_obj_data(DATA / "基坑1.0（存在多部分贴合和局部重叠）.obj")

    with pytest.raises(ValueError, match="pre_union_snap_rel"):
        repair_mesh_data(
            source,
            mode="solid",
            pre_union_snap_relative=-1e-8,
        )
