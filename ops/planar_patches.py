"""识别共面三角片区和边界环。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from mesh.io_obj import ObjMesh


Edge = tuple[int, int]
Label = tuple[str, str, str]


@dataclass(frozen=True)
class PlanarPatch:
    face_ids: tuple[int, ...]
    loops: tuple[tuple[int, ...], ...]
    normal: np.ndarray
    origin: np.ndarray
    area: float
    perimeter: float
    label: Label

    @property
    def target_size(self) -> float:
        return 2.0 * self.area / self.perimeter


def _face_labels(mesh: ObjMesh) -> list[Label]:
    return list(
        zip(mesh.face_object, mesh.face_group, mesh.face_material, strict=True)
    )


def _edge_faces(faces: np.ndarray) -> dict[Edge, list[int]]:
    result: dict[Edge, list[int]] = defaultdict(list)
    for face_id, face in enumerate(faces):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[first]), int(face[second]))))
            result[edge].append(face_id)
    return result


def _boundary_loops(
    faces: np.ndarray,
    face_ids: set[int],
    edge_faces: dict[Edge, list[int]],
) -> tuple[tuple[int, ...], ...]:
    halfedges: list[tuple[int, int]] = []
    for face_id in face_ids:
        face = faces[face_id]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            start = int(face[first])
            end = int(face[second])
            incident = edge_faces[tuple(sorted((start, end)))]
            if len(incident) != 2 or any(value not in face_ids for value in incident):
                halfedges.append((start, end))

    outgoing: dict[int, list[int]] = defaultdict(list)
    for start, end in halfedges:
        outgoing[start].append(end)
    if any(len(values) != 1 for values in outgoing.values()):
        raise RuntimeError("共面片区边界不是简单闭环")

    unused = set(halfedges)
    loops: list[tuple[int, ...]] = []
    while unused:
        start, end = next(iter(unused))
        loop = [start]
        current = (start, end)
        while True:
            if current not in unused:
                raise RuntimeError("共面片区边界环发生重复或断裂")
            unused.remove(current)
            loop.append(current[1])
            if current[1] == loop[0]:
                break
            next_values = outgoing.get(current[1], [])
            if len(next_values) != 1:
                raise RuntimeError("共面片区边界环无法闭合")
            current = (current[1], next_values[0])
        loops.append(tuple(loop[:-1]))
    return tuple(loops)


def _sort_loops(
    vertices: np.ndarray,
    loops: tuple[tuple[int, ...], ...],
    normal: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    axis_u = vertices[loops[0][1]] - vertices[loops[0][0]]
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)

    def signed_area(loop: tuple[int, ...]) -> float:
        points = vertices[np.asarray(loop, dtype=np.int64)]
        xy = np.column_stack((points @ axis_u, points @ axis_v))
        shifted = np.roll(xy, -1, axis=0)
        return 0.5 * float(
            np.sum(xy[:, 0] * shifted[:, 1] - shifted[:, 0] * xy[:, 1])
        )

    areas = [signed_area(loop) for loop in loops]
    outer = int(np.argmax(np.abs(areas)))
    ordered = [loops[outer]]
    ordered.extend(loop for index, loop in enumerate(loops) if index != outer)
    return tuple(ordered)


def detect_planar_patches(
    mesh: ObjMesh,
    *,
    angle_tolerance_degrees: float = 1e-5,
    distance_tolerance_relative: float = 1e-10,
) -> tuple[list[PlanarPatch], np.ndarray]:
    """沿共面且标签一致的内部边合并三角形。"""

    triangles = mesh.V[mesh.F]
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_area = np.linalg.norm(raw_normals, axis=1)
    if np.any(double_area <= 0.0):
        raise ValueError("共面片区识别前必须清除退化三角形")
    normals = raw_normals / double_area[:, None]
    labels = _face_labels(mesh)
    edge_faces = _edge_faces(mesh.F)
    candidates: list[list[int]] = [[] for _ in range(len(mesh.F))]
    diagonal = float(np.linalg.norm(mesh.V.max(axis=0) - mesh.V.min(axis=0)))
    distance_tolerance = max(
        diagonal * distance_tolerance_relative,
        np.finfo(np.float64).eps * max(diagonal, 1.0) * 64.0,
    )
    cosine_tolerance = np.cos(np.deg2rad(angle_tolerance_degrees))

    for incident in edge_faces.values():
        if len(incident) != 2:
            continue
        first, second = incident
        if labels[first] != labels[second]:
            continue
        if float(np.dot(normals[first], normals[second])) < cosine_tolerance:
            continue
        offset = triangles[second] - triangles[first, 0]
        if float(np.max(np.abs(offset @ normals[first]))) > distance_tolerance:
            continue
        candidates[first].append(second)
        candidates[second].append(first)

    face_patch = np.full(len(mesh.F), -1, dtype=np.int64)
    patches: list[PlanarPatch] = []
    for start in range(len(mesh.F)):
        if face_patch[start] >= 0:
            continue
        patch_id = len(patches)
        stack = [start]
        face_ids: list[int] = []
        face_patch[start] = patch_id
        while stack:
            face_id = stack.pop()
            face_ids.append(face_id)
            for neighbor in candidates[face_id]:
                if face_patch[neighbor] >= 0:
                    continue
                if float(np.dot(normals[start], normals[neighbor])) < cosine_tolerance:
                    continue
                offset = triangles[neighbor] - triangles[start, 0]
                if (
                    float(np.max(np.abs(offset @ normals[start])))
                    > distance_tolerance
                ):
                    continue
                face_patch[neighbor] = patch_id
                stack.append(neighbor)

        face_set = set(face_ids)
        normal = np.sum(raw_normals[face_ids], axis=0)
        normal /= np.linalg.norm(normal)
        loops = _boundary_loops(mesh.F, face_set, edge_faces)
        loops = _sort_loops(mesh.V, loops, normal)
        perimeter = 0.0
        for loop in loops:
            points = mesh.V[np.asarray(loop, dtype=np.int64)]
            perimeter += float(
                np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
            )
        patches.append(
            PlanarPatch(
                face_ids=tuple(face_ids),
                loops=loops,
                normal=normal,
                origin=triangles[start, 0].copy(),
                area=float(double_area[face_ids].sum() / 2.0),
                perimeter=perimeter,
                label=labels[start],
            )
        )
    return patches, face_patch
