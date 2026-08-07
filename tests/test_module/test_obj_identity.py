"""测试 OBJ 零件身份读写。"""

import numpy as np

from mesh.io_obj import ObjMesh, load_obj_data, save_obj_data
from ops.pipeline_impl import repair_mesh_data


def test_obj_keeps_object_group_and_material(tmp_path):
    source = tmp_path / "source.obj"
    source.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "o plate",
                "g top",
                "usemtl steel",
                "f 1 2 3 4",
            ]
        ),
        encoding="utf-8",
    )

    mesh = load_obj_data(source)
    assert len(mesh.F) == 2
    assert mesh.face_object == ["plate", "plate"]
    assert mesh.face_group == ["top", "top"]
    assert mesh.face_material == ["steel", "steel"]

    output = tmp_path / "output.obj"
    save_obj_data(output, mesh)
    loaded = load_obj_data(output)
    assert loaded.face_object == mesh.face_object
    assert loaded.face_group == mesh.face_group
    assert loaded.face_material == mesh.face_material


def test_assembly_keeps_face_labels_when_topology_is_unchanged():
    mesh = ObjMesh(
        V=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        F=np.array([[0, 1, 2], [0, 2, 3]]),
        face_object=["plate", "plate"],
        face_group=["right", "left"],
        face_material=["steel", "coating"],
    )

    output, report = repair_mesh_data(mesh, mode="assembly")

    assert report.output_validation["success"] is True
    assert output.face_group == mesh.face_group
    assert output.face_material == mesh.face_material


def test_assembly_keeps_uniform_labels_after_face_cleanup():
    mesh = ObjMesh(
        V=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        F=np.array([[0, 1, 2], [2, 1, 0]]),
        face_object=["plate", "plate"],
        face_group=["wall", "wall"],
        face_material=["steel", "steel"],
    )

    output, report = repair_mesh_data(mesh, mode="assembly")

    assert report.part_reports[0]["changes"]["removed_duplicate_faces"] == 1
    assert output.face_group == ["wall"]
    assert output.face_material == ["steel"]
