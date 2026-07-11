"""使用 Gmsh 对三角网格做均匀细分。"""

from __future__ import annotations

import uuid

import numpy as np

from mesh.io_obj import ObjMesh


def _copy_mesh(mesh: ObjMesh) -> ObjMesh:
    return ObjMesh(
        mesh.V.copy(),
        mesh.F.copy(),
        mesh.face_object.copy(),
        mesh.face_group.copy(),
        mesh.face_material.copy(),
    )


def _read_triangles(gmsh, surface_tag: int) -> tuple[np.ndarray, np.ndarray]:
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes(
        2,
        surface_tag,
        includeBoundary=True,
    )
    node_tags = np.asarray(node_tags, dtype=np.uint64)
    vertices = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    node_index = {int(tag): index for index, tag in enumerate(node_tags)}

    element_types, _, element_nodes = gmsh.model.mesh.getElements(2, surface_tag)
    triangle_nodes = None
    for element_type, nodes in zip(element_types, element_nodes):
        if int(element_type) == 2:
            triangle_nodes = np.asarray(nodes, dtype=np.uint64).reshape(-1, 3)
            break

    if triangle_nodes is None:
        raise RuntimeError("Gmsh 细分结果中没有一阶三角形")

    faces = np.asarray(
        [[node_index[int(tag)] for tag in face] for face in triangle_nodes],
        dtype=np.int64,
    )
    return vertices, faces


def uniform_refine(mesh: ObjMesh, levels: int = 1) -> ObjMesh:
    """每一级把一个三角形均匀拆成四个三角形。"""

    levels = int(levels)
    if levels < 0:
        raise ValueError("uniform_refine_levels 不能为负数")
    if levels == 0:
        return _copy_mesh(mesh)
    if len(mesh.V) == 0 or len(mesh.F) == 0:
        raise ValueError("空网格不能进行 Gmsh 均匀细分")

    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Gmsh 均匀细分不可用，请安装 requirements-gmsh.txt；"
            "Linux 还需要 libGLU.so.1 和 libXft.so.2"
        ) from exc

    started_here = not bool(gmsh.isInitialized())
    if started_here:
        gmsh.initialize()

    surface_tag = 1
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"uniform_refine_{uuid.uuid4().hex}")
        gmsh.model.addDiscreteEntity(2, surface_tag)

        node_tags = np.arange(1, len(mesh.V) + 1, dtype=np.uint64)
        element_tags = np.arange(1, len(mesh.F) + 1, dtype=np.uint64)
        triangle_nodes = np.asarray(mesh.F + 1, dtype=np.uint64)

        gmsh.model.mesh.addNodes(
            2,
            surface_tag,
            node_tags,
            np.asarray(mesh.V, dtype=np.float64).reshape(-1),
        )
        gmsh.model.mesh.addElementsByType(
            surface_tag,
            2,
            element_tags,
            triangle_nodes.reshape(-1),
        )

        for _ in range(levels):
            gmsh.model.mesh.refine()

        vertices, faces = _read_triangles(gmsh, surface_tag)
    finally:
        if started_here:
            gmsh.finalize()
        else:
            gmsh.model.remove()

    return ObjMesh(
        vertices,
        faces,
        face_object=["refined"] * len(faces),
        face_group=[f"gmsh_uniform_l{levels}"] * len(faces),
        face_material=[""] * len(faces),
    )

