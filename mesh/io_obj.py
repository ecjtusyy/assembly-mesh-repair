# -*- coding: utf-8 -*-
"""OBJ 读写。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PathLike = str | Path


@dataclass
class ObjMesh:
    """保存三角网格和每个面的 OBJ 身份。"""

    V: np.ndarray
    F: np.ndarray
    face_object: list[str] = field(default_factory=list)
    face_group: list[str] = field(default_factory=list)
    face_material: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.V = _normalize_vertices(self.V)
        self.F = _normalize_faces(self.F, len(self.V))

        n_faces = len(self.F)
        if not self.face_object:
            self.face_object = ["default"] * n_faces
        if not self.face_group:
            self.face_group = ["default"] * n_faces
        if not self.face_material:
            self.face_material = [""] * n_faces

        for name, values in (
            ("face_object", self.face_object),
            ("face_group", self.face_group),
            ("face_material", self.face_material),
        ):
            if len(values) != n_faces:
                raise ValueError(f"{name} 数量必须等于面数")


def _empty_vertices() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float64)


def _empty_faces() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.int64)


def _normalize_vertices(V: np.ndarray) -> np.ndarray:
    vertices = np.asarray(V, dtype=np.float64)
    if vertices.size == 0:
        return _empty_vertices()
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"V 必须是 (N, 3) 数组，当前是 {vertices.shape}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("V 中存在 NaN 或 Inf")
    return vertices


def _normalize_faces(F: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(F, dtype=np.int64)
    if faces.size == 0:
        return _empty_faces()
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"F 必须是 (M, 3) 数组，当前是 {faces.shape}")
    if int(faces.min()) < 0 or int(faces.max()) >= vertex_count:
        raise ValueError("F 中存在越界顶点索引")
    return faces


def _face_index(token: str, vertex_count: int, path: Path, line_no: int) -> int:
    head = token.split("/", 1)[0]
    try:
        raw = int(head)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no} 面索引无效：{token!r}") from exc

    if raw == 0:
        raise ValueError(f"{path}:{line_no} OBJ 索引不能为 0")

    index = raw - 1 if raw > 0 else vertex_count + raw
    if index < 0 or index >= vertex_count:
        raise ValueError(f"{path}:{line_no} 面索引越界：{token!r}")
    return index


def load_obj_data(path: PathLike) -> ObjMesh:
    """读取 OBJ，同时保留 o、g、usemtl。"""

    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_object: list[str] = []
    face_group: list[str] = []
    face_material: list[str] = []

    current_object = "default"
    current_group = "default"
    current_material = ""

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            record = parts[0]

            if record == "v":
                if len(parts) < 4:
                    raise ValueError(f"{path}:{line_no} 顶点行不完整")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if record == "o":
                current_object = " ".join(parts[1:]) or "default"
                continue

            if record == "g":
                current_group = " ".join(parts[1:]) or "default"
                continue

            if record == "usemtl":
                current_material = " ".join(parts[1:])
                continue

            if record != "f":
                continue

            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no} 面的顶点数少于 3")

            ids = [
                _face_index(token, len(vertices), path, line_no)
                for token in parts[1:]
            ]

            for i in range(1, len(ids) - 1):
                faces.append([ids[0], ids[i], ids[i + 1]])
                face_object.append(current_object)
                face_group.append(current_group)
                face_material.append(current_material)

    V = np.asarray(vertices, dtype=np.float64) if vertices else _empty_vertices()
    F = np.asarray(faces, dtype=np.int64) if faces else _empty_faces()
    return ObjMesh(V, F, face_object, face_group, face_material)


def load_obj(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """兼容原接口，只返回 V/F。"""

    mesh = load_obj_data(path)
    return mesh.V, mesh.F


def save_obj(path: PathLike, V: np.ndarray, F: np.ndarray) -> None:
    """保存最小 OBJ。"""

    save_obj_data(path, ObjMesh(V, F))


def save_obj_data(path: PathLike, mesh: ObjMesh) -> None:
    """保存 OBJ，并写出面所属零件。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        for x, y, z in mesh.V:
            stream.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")

        last_object = None
        last_group = None
        last_material = None

        for face_id, (a, b, c) in enumerate(mesh.F):
            object_name = mesh.face_object[face_id]
            group_name = mesh.face_group[face_id]
            material_name = mesh.face_material[face_id]

            if object_name != last_object:
                stream.write(f"o {object_name}\n")
                last_object = object_name
            if group_name != last_group:
                stream.write(f"g {group_name}\n")
                last_group = group_name
            if material_name != last_material:
                if material_name:
                    stream.write(f"usemtl {material_name}\n")
                last_material = material_name

            stream.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


read_obj = load_obj
write_obj = save_obj
