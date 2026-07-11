"""网格零件拆分和合并。"""

from __future__ import annotations

import numpy as np

from mesh.diagnostics import connected_components_by_edges
from mesh.io_obj import ObjMesh


def _submesh(mesh: ObjMesh, face_ids: list[int], name: str) -> ObjMesh:
    faces = mesh.F[face_ids]
    used = np.unique(faces.reshape(-1))
    old_to_new = -np.ones(len(mesh.V), dtype=np.int64)
    old_to_new[used] = np.arange(len(used), dtype=np.int64)

    return ObjMesh(
        V=mesh.V[used].copy(),
        F=old_to_new[faces],
        face_object=[name] * len(face_ids),
        face_group=[mesh.face_group[i] for i in face_ids],
        face_material=[mesh.face_material[i] for i in face_ids],
    )


def split_edge_components(mesh: ObjMesh) -> list[ObjMesh]:
    """按共享边拆分零件，不根据坐标焊接。"""

    if len(mesh.F) == 0:
        return []

    result = connected_components_by_edges(mesh.F)
    ids = result["face_component_id"]
    count = int(result["component_count"])

    parts: list[ObjMesh] = []
    for component_id in range(count):
        face_ids = [i for i, value in enumerate(ids) if value == component_id]
        source_names = {mesh.face_object[i] for i in face_ids}
        source_name = next(iter(source_names)) if len(source_names) == 1 else "mixed"
        name = f"part_{component_id:03d}_{source_name}"
        parts.append(_submesh(mesh, face_ids, name))

    return parts


def combine_parts(parts: list[ObjMesh]) -> ObjMesh:
    """把多个独立零件写入同一个 V/F。"""

    if not parts:
        return ObjMesh(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
        )

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    objects: list[str] = []
    groups: list[str] = []
    materials: list[str] = []
    offset = 0

    for part in parts:
        vertices.append(part.V)
        faces.append(part.F + offset)
        objects.extend(part.face_object)
        groups.extend(part.face_group)
        materials.extend(part.face_material)
        offset += len(part.V)

    return ObjMesh(
        V=np.vstack(vertices),
        F=np.vstack(faces),
        face_object=objects,
        face_group=groups,
        face_material=materials,
    )
