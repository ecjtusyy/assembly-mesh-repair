"""测试保持几何的表面质量重剖分。"""

import numpy as np
import pytest
import trimesh

pytest.importorskip("gmsh")

from mesh.io_obj import ObjMesh
from ops.gmsh_quality_remesh import remesh_planar_surface
from ops.pipeline_impl import repair_mesh_data


def _long_box() -> ObjMesh:
    box = trimesh.creation.box(extents=[10.0, 1.0, 1.0])
    return ObjMesh(
        np.asarray(box.vertices),
        np.asarray(box.faces),
        face_object=["box"] * len(box.faces),
        face_group=["wall"] * len(box.faces),
        face_material=["steel"] * len(box.faces),
    )


def _sharp_tetrahedron() -> ObjMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    )
    return ObjMesh(vertices, faces)


def test_retriangulates_rectangle_without_changing_geometry():
    source = _long_box()

    output, report = remesh_planar_surface(source)

    assert report["before"]["minimum_angle_degrees"]["minimum"] < 6.0
    assert report["before"]["bad_faces"] == 8
    assert report["after"]["minimum_angle_degrees"]["minimum"] > 30.0
    assert report["after"]["bad_faces"] == 0
    assert report["geometry"]["success"] is True
    assert report["geometry"]["volume_error_relative"] < 1e-12
    assert (
        report["geometry"]["maximum_feature_curve_error_relative"] < 1e-12
    )
    assert len(output.F) > len(source.F)
    assert set(output.face_object) == {"box"}
    assert set(output.face_group) == {"wall"}
    assert set(output.face_material) == {"steel"}


def test_pipeline_records_quality_and_geometry_evidence():
    output, report = repair_mesh_data(
        _long_box(),
        mode="solid",
        quality_surface_remesh=True,
    )

    assert report.status == "success"
    assert report.quality_surface_remesh is True
    assert report.surface_remesh_report["success"] is True
    assert report.pre_surface_faces == 12
    assert report.post_surface_faces == len(output.F)
    assert report.output_validation["is_volume"] is True


def test_does_not_claim_quality_when_geometry_has_a_sharp_corner():
    with pytest.raises(
        RuntimeError,
        match="surface_quality_below_threshold",
    ):
        remesh_planar_surface(_sharp_tetrahedron())
