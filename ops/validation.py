"""修复结果验收。"""

from __future__ import annotations

import trimesh

from mesh.diagnostics import topology_summary
from mesh.io_obj import ObjMesh


def validate_mesh(mesh: ObjMesh, *, require_volume: bool) -> dict[str, object]:
    """同时检查组合拓扑和实体属性。"""

    topology = topology_summary(mesh.V, mesh.F)
    tri = trimesh.Trimesh(vertices=mesh.V, faces=mesh.F, process=False)

    winding = bool(tri.is_winding_consistent) if len(mesh.F) else False
    watertight = bool(tri.is_watertight) if len(mesh.F) else False
    signed_volume = float(tri.volume) if watertight and winding else 0.0
    volume = watertight and winding and signed_volume > 0.0

    errors: list[str] = []
    for key in (
        "degenerate_faces",
        "duplicate_faces",
        "nonmanifold_edges",
        "nonmanifold_vertices",
    ):
        if int(topology[key]) > 0:
            errors.append(key)

    if not winding:
        errors.append("inconsistent_winding")
    if require_volume and not watertight:
        errors.append("not_watertight")
    if require_volume and not volume:
        errors.append("not_positive_volume")

    return {
        "success": not errors,
        "errors": errors,
        "vertices": int(len(mesh.V)),
        "faces": int(len(mesh.F)),
        "topology": topology,
        "is_watertight": watertight,
        "is_winding_consistent": winding,
        "is_volume": volume,
        "volume": signed_volume,
    }


def require_valid(report: dict[str, object], label: str) -> None:
    """验收失败时给出可读错误。"""

    if bool(report.get("success")):
        return
    errors = ", ".join(str(x) for x in report.get("errors", []))
    raise RuntimeError(f"{label} 验收失败：{errors}")
