"""测试 OBJ 零件身份读写。"""

from mesh.io_obj import load_obj_data, save_obj_data


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
