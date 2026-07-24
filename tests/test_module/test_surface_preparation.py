"""测试宽松体网格前的安全拓扑清理。"""

import numpy as np

from mesh.io_obj import ObjMesh
from ops.validation import validate_mesh
from volume.gmsh_tetra import _prepare_surface


def _tetrahedron_with_a_split_edge() -> ObjMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [5e-9, 0.0, 0.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 4],
            [4, 2, 1],
            [0, 4, 3],
            [4, 1, 3],
            [1, 2, 3],
            [0, 3, 2],
        ]
    )
    return ObjMesh(vertices, faces)


def test_prepare_surface_removes_short_edge_only_after_validation():
    source = _tetrahedron_with_a_split_edge()

    output, report = _prepare_surface(source)

    validation = validate_mesh(
        output,
        require_volume=True,
        check_self_intersections=True,
    )
    assert validation["success"] is True
    assert report["method"] == "validated_topology_cleanup"
    assert report["vertices_removed"] == 1
    assert report["faces_removed"] == 2


def test_prepare_surface_accepts_non_contiguous_arrays():
    source = _tetrahedron_with_a_split_edge()
    non_contiguous = ObjMesh(
        np.asfortranarray(source.V),
        np.asfortranarray(source.F),
    )

    output, report = _prepare_surface(non_contiguous)

    assert report["validation"]["success"] is True
    assert len(output.F) > 0


def test_prepare_surface_rejects_adjacent_face_overlap():
    mesh = ObjMesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        ),
        np.array([[0, 1, 2], [1, 0, 3]]),
    )

    with np.testing.assert_raises_regex(
        RuntimeError,
        "adjacent_face_overlaps",
    ):
        _prepare_surface(mesh)
