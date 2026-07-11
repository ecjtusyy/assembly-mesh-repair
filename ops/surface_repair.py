"""局部表面修复。"""

from __future__ import annotations

from collections import deque

import numpy as np

from mesh.diagnostics import build_edge_faces
from mesh.io_obj import ObjMesh
from mesh.mesh import Mesh
from ops.stitch import (
    area_threshold_from_mesh,
    cleanup_topology,
    compact_vertices,
    remove_degenerate_faces,
    remove_duplicate_faces,
)


def _directed_edges(face: np.ndarray) -> tuple[tuple[int, int], ...]:
    a, b, c = (int(face[0]), int(face[1]), int(face[2]))
    return ((a, b), (b, c), (c, a))


def orient_faces(F: np.ndarray) -> int:
    """用共享边传播面方向，返回翻转面数。"""

    faces = np.asarray(F, dtype=np.int64)
    edge_use: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for face_id, face in enumerate(faces):
        for u, v in _directed_edges(face):
            edge = (u, v) if u < v else (v, u)
            direction = 1 if (u, v) == edge else -1
            edge_use.setdefault(edge, []).append((face_id, direction))

    neighbors: list[list[tuple[int, int]]] = [[] for _ in range(len(faces))]
    for uses in edge_use.values():
        if len(uses) != 2:
            continue
        (a, da), (b, db) = uses
        relation = 1 if da == db else 0
        neighbors[a].append((b, relation))
        neighbors[b].append((a, relation))

    parity = -np.ones(len(faces), dtype=np.int8)
    for start in range(len(faces)):
        if parity[start] != -1:
            continue
        parity[start] = 0
        queue: deque[int] = deque([start])
        while queue:
            face_id = queue.popleft()
            for next_id, relation in neighbors[face_id]:
                expected = int(parity[face_id]) ^ relation
                if parity[next_id] == -1:
                    parity[next_id] = expected
                    queue.append(next_id)

    flip_ids = np.flatnonzero(parity == 1)
    faces[flip_ids] = faces[flip_ids][:, [0, 2, 1]]
    return int(len(flip_ids))


def _vertex_fans(
    F: np.ndarray,
    vertex_id: int,
    edge_faces: dict[tuple[int, int], list[int]],
    incident: list[int],
) -> list[list[int]]:
    if not incident:
        return []

    neighbors = {face_id: set() for face_id in incident}
    checked_edges: set[tuple[int, int]] = set()
    for face_id in incident:
        for u, v in _directed_edges(F[face_id]):
            if vertex_id not in (u, v):
                continue
            edge = (u, v) if u < v else (v, u)
            if edge in checked_edges:
                continue
            checked_edges.add(edge)
            face_ids = edge_faces[edge]
            if len(face_ids) == 2:
                a, b = face_ids
                neighbors[a].add(b)
                neighbors[b].add(a)

    fans: list[list[int]] = []
    seen: set[int] = set()
    for start in incident:
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        fan: list[int] = []
        while queue:
            face_id = queue.pop()
            fan.append(face_id)
            for next_id in neighbors[face_id]:
                if next_id not in seen:
                    seen.add(next_id)
                    queue.append(next_id)
        fans.append(fan)
    return fans


def split_nonmanifold_vertices(mesh: Mesh) -> int:
    """按顶点周围的独立面扇复制顶点。"""

    original_vertices = len(mesh.V)
    new_vertices = [point.copy() for point in mesh.V]
    base_faces = mesh.F.copy()
    faces = base_faces.copy()
    split_count = 0

    edge_faces = build_edge_faces(base_faces)
    incident_by_vertex: list[list[int]] = [[] for _ in range(original_vertices)]
    for face_id, face in enumerate(base_faces):
        for vertex_id in face:
            incident_by_vertex[int(vertex_id)].append(face_id)

    for vertex_id in range(original_vertices):
        fans = _vertex_fans(
            base_faces,
            vertex_id,
            edge_faces,
            incident_by_vertex[vertex_id],
        )
        for fan in fans[1:]:
            new_id = len(new_vertices)
            new_vertices.append(mesh.V[vertex_id].copy())
            for face_id in fan:
                faces[face_id, faces[face_id] == vertex_id] = new_id
            split_count += 1

    if split_count:
        mesh.V = np.asarray(new_vertices, dtype=np.float64)
        mesh.F = faces
        mesh.mark_dirty()
    return split_count


def _split_edge_at_vertex(mesh: Mesh, edge: tuple[int, int], vertex_id: int) -> int:
    """把使用一条边的三角形从已有顶点处切开。"""

    a, b = edge
    new_faces: list[list[int]] = []
    split_faces = 0

    for face in mesh.F:
        directed = _directed_edges(face)
        if (a, b) in directed:
            third = next(int(x) for x in face if x not in edge)
            new_faces.extend([[a, vertex_id, third], [vertex_id, b, third]])
            split_faces += 1
        elif (b, a) in directed:
            third = next(int(x) for x in face if x not in edge)
            new_faces.extend([[b, vertex_id, third], [vertex_id, a, third]])
            split_faces += 1
        else:
            new_faces.append([int(x) for x in face])

    if split_faces:
        mesh.F = np.asarray(new_faces, dtype=np.int64)
        mesh.mark_dirty()
    return split_faces


def repair_t_junctions(mesh: Mesh, eps: float) -> int:
    """把落在已有边内部的顶点接入三角网格。"""

    if len(mesh.V) == 0 or len(mesh.F) == 0:
        return 0

    diag = float(np.linalg.norm(mesh.V.max(axis=0) - mesh.V.min(axis=0)))
    tolerance = max(float(eps), diag * 1e-12, 1e-14)
    total = 0

    for _ in range(10000):
        changed = False
        for a, b in build_edge_faces(mesh.F):
            start = mesh.V[a]
            vector = mesh.V[b] - start
            length2 = float(np.dot(vector, vector))
            if length2 == 0.0:
                continue

            relative = mesh.V - start
            t = relative @ vector / length2
            projected = start + t[:, None] * vector
            distance = np.linalg.norm(mesh.V - projected, axis=1)
            candidates = np.flatnonzero(
                (t > tolerance / np.sqrt(length2))
                & (t < 1.0 - tolerance / np.sqrt(length2))
                & (distance <= tolerance)
            )

            candidates = candidates[(candidates != a) & (candidates != b)]
            if len(candidates) == 0:
                continue

            vertex_id = int(candidates[np.argmin(t[candidates])])
            split_faces = _split_edge_at_vertex(mesh, (a, b), vertex_id)
            if split_faces:
                total += split_faces
                changed = True
                break

        if not changed:
            return total

    raise RuntimeError("T 接头修复次数超过上限")


def _boundary_loops(F: np.ndarray) -> list[list[int]]:
    edge_faces = build_edge_faces(F)
    boundary = {edge for edge, face_ids in edge_faces.items() if len(face_ids) == 1}
    directed: dict[int, int] = {}

    for face in F:
        for u, v in _directed_edges(face):
            edge = (u, v) if u < v else (v, u)
            if edge in boundary:
                if u in directed:
                    return []
                directed[u] = v

    loops: list[list[int]] = []
    used: set[int] = set()
    for start in directed:
        if start in used:
            continue
        loop = [start]
        current = start
        while current in directed:
            used.add(current)
            current = directed[current]
            if current == start:
                break
            if current in used:
                return []
            loop.append(current)
        if current == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_small_holes(mesh: Mesh, max_edges: int = 4) -> int:
    """只填三角形或四边形小孔。"""

    new_faces: list[list[int]] = []
    for loop in _boundary_loops(mesh.F):
        if len(loop) > max_edges:
            continue
        root = loop[0]
        for i in range(1, len(loop) - 1):
            new_faces.append([root, loop[i + 1], loop[i]])

    if new_faces:
        mesh.F = np.vstack([mesh.F, np.asarray(new_faces, dtype=np.int64)])
        mesh.mark_dirty()
    return len(new_faces)


def repair_surface_part(
    part: ObjMesh,
    *,
    eps_v: float,
    fill_holes: bool = False,
    max_hole_edges: int = 4,
) -> tuple[ObjMesh, dict[str, int]]:
    """修复单个零件，不和其它零件焊接。"""

    mesh = Mesh(part.V.copy(), part.F.copy())
    cleanup = cleanup_topology(
        mesh,
        eps_v=eps_v,
        area_eps=area_threshold_from_mesh(mesh),
    )
    t_junction_faces = repair_t_junctions(mesh, eps_v)
    flipped = orient_faces(mesh.F)
    split_vertices = split_nonmanifold_vertices(mesh)
    flipped += orient_faces(mesh.F)
    holes_added = fill_small_holes(mesh, max_hole_edges) if fill_holes else 0

    remove_degenerate_faces(mesh, area_eps=area_threshold_from_mesh(mesh))
    remove_duplicate_faces(mesh)
    compact_vertices(mesh)

    object_name = part.face_object[0] if part.face_object else "part"
    output = ObjMesh(
        mesh.V,
        mesh.F,
        face_object=[object_name] * len(mesh.F),
        face_group=["repaired"] * len(mesh.F),
        face_material=[""] * len(mesh.F),
    )
    report = {
        "merged_vertices": int(cleanup.merged_vertices),
        "removed_degenerate_faces": int(cleanup.degenerate_removed),
        "removed_duplicate_faces": int(cleanup.duplicate_removed),
        "flipped_faces": int(flipped),
        "split_t_junction_faces": int(t_junction_faces),
        "split_vertices": int(split_vertices),
        "added_hole_faces": int(holes_added),
    }
    return output, report
