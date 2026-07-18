"""写出求解器通用的 Gmsh 2.2 ASCII 体网格。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from volume.tetra_mesh import TetraMesh


def _safe_name(value: str) -> str:
    return value.replace("\\", "_").replace('"', "_")


def write_msh22(
    path: str | Path,
    mesh: TetraMesh,
    *,
    domain_name: str,
    boundary_names: Sequence[str] | None = None,
) -> list[str]:
    """写出体网格，并返回实际使用的边界物理组。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if boundary_names is None:
        names = ["boundary"] * len(mesh.boundary_faces)
    else:
        names = [_safe_name(value.strip() or "boundary") for value in boundary_names]
        if len(names) != len(mesh.boundary_faces):
            raise ValueError("boundary_names 数量必须等于边界三角形数")

    physical_names = list(dict.fromkeys(names)) or ["boundary"]
    physical_ids = {name: index + 1 for index, name in enumerate(physical_names)}
    domain_id = len(physical_names) + 1
    lines = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        str(len(physical_names) + 1),
    ]
    lines.extend(
        f'2 {physical_ids[name]} "{name}"' for name in physical_names
    )
    lines.extend(
        [
            f'3 {domain_id} "{_safe_name(domain_name or "domain")}"',
            "$EndPhysicalNames",
            "$Nodes",
            str(len(mesh.V)),
        ]
    )
    lines.extend(
        f"{node_id} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}"
        for node_id, point in enumerate(mesh.V, start=1)
    )
    lines.extend(["$EndNodes", "$Elements", str(len(mesh.boundary_faces) + len(mesh.T))])
    element_id = 1
    for face, name in zip(mesh.boundary_faces, names):
        nodes = " ".join(str(int(value) + 1) for value in face)
        physical_id = physical_ids[name]
        lines.append(f"{element_id} 2 2 {physical_id} {physical_id} {nodes}")
        element_id += 1
    for tetrahedron in mesh.T:
        nodes = " ".join(str(int(value) + 1) for value in tetrahedron)
        lines.append(f"{element_id} 4 2 {domain_id} {domain_id} {nodes}")
        element_id += 1
    lines.extend(["$EndElements", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return physical_names
