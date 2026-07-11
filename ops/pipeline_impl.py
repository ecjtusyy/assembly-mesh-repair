"""网格修复主流程。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce

import manifold3d
import numpy as np
import trimesh

from mesh.components import combine_parts, split_edge_components
from mesh.diagnostics import topology_summary
from mesh.io_obj import ObjMesh
from mesh.mesh import Mesh
from ops.stitch import CleanupReport, area_threshold_from_mesh, cleanup_topology
from ops.surface_repair import repair_surface_part, repair_t_junctions
from ops.validation import require_valid, validate_mesh


MeshDict = dict[str, np.ndarray]


@dataclass
class RepairRunReport:
    """单个文件的修复报告。"""

    mode: str
    eps_v_abs: float
    input_topology: dict[str, object]
    part_reports: list[dict[str, object]] = field(default_factory=list)
    pre_refine_validation: dict[str, object] = field(default_factory=dict)
    output_validation: dict[str, object] = field(default_factory=dict)
    approximate_rebuild: bool = False
    uniform_refine_levels: int = 0
    pre_refine_faces: int | None = None
    post_refine_faces: int | None = None
    warnings: list[str] = field(default_factory=list)
    status: str = "unknown"

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "eps_v_abs": self.eps_v_abs,
            "input_topology": self.input_topology,
            "part_reports": self.part_reports,
            "pre_refine_validation": self.pre_refine_validation,
            "output_validation": self.output_validation,
            "approximate_rebuild": self.approximate_rebuild,
            "uniform_refine_levels": self.uniform_refine_levels,
            "pre_refine_faces": self.pre_refine_faces,
            "post_refine_faces": self.post_refine_faces,
            "warnings": self.warnings,
            "status": self.status,
        }


def bbox_diag(V: np.ndarray) -> float:
    points = np.asarray(V, dtype=np.float64)
    if points.size == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def normalize_eps(eps: float, V: np.ndarray, mode: str = "relative_bbox") -> float:
    eps = float(eps)
    if eps < 0:
        raise ValueError("eps_v 不能为负数")
    if mode == "absolute":
        return eps
    if mode == "relative_bbox":
        return eps * bbox_diag(V)
    raise ValueError("eps_mode 只能是 absolute 或 relative_bbox")


def python_cleanup_only(
    mesh_in: MeshDict,
    *,
    eps_v: float = 0.0,
    eps_mode: str = "relative_bbox",
) -> tuple[MeshDict, CleanupReport, float]:
    """保留原来的纯 Python 清理接口。"""

    mesh = Mesh(
        np.asarray(mesh_in["V"], dtype=np.float64).copy(),
        np.asarray(mesh_in["F"], dtype=np.int64).copy(),
    )
    eps_v_abs = normalize_eps(eps_v, mesh.V, eps_mode)
    report = cleanup_topology(
        mesh,
        eps_v=eps_v_abs,
        area_eps=area_threshold_from_mesh(mesh),
    )
    return {"V": mesh.V, "F": mesh.F}, report, eps_v_abs


def _repair_parts(
    mesh: ObjMesh,
    *,
    eps_v_abs: float,
    fill_holes: bool,
    max_hole_edges: int,
    connect_t_junctions: bool = False,
) -> tuple[list[ObjMesh], list[dict[str, object]]]:
    global_t_splits = 0
    source = mesh
    if connect_t_junctions:
        connected = Mesh(mesh.V.copy(), mesh.F.copy())
        global_t_splits = repair_t_junctions(connected, eps_v_abs)
        source = ObjMesh(connected.V, connected.F)

    parts = split_edge_components(source)
    repaired: list[ObjMesh] = []
    reports: list[dict[str, object]] = []

    for part_id, part in enumerate(parts):
        output, changes = repair_surface_part(
            part,
            eps_v=eps_v_abs,
            fill_holes=fill_holes,
            max_hole_edges=max_hole_edges,
        )
        validation = validate_mesh(output, require_volume=False)
        require_valid(validation, f"part_{part_id}")

        repaired.append(output)
        reports.append(
            {
                "part": part_id,
                "input_vertices": int(len(part.V)),
                "input_faces": int(len(part.F)),
                "changes": changes,
                "validation": validation,
            }
        )

    if reports and global_t_splits:
        reports[0]["changes"]["split_cross_component_t_junction_faces"] = int(
            global_t_splits
        )

    return repaired, reports


def _outward_part(part: ObjMesh) -> ObjMesh:
    tri = trimesh.Trimesh(vertices=part.V, faces=part.F, process=False)
    if tri.is_watertight and tri.is_winding_consistent and float(tri.volume) < 0:
        faces = part.F[:, [0, 2, 1]]
        return ObjMesh(
            part.V.copy(),
            faces,
            part.face_object.copy(),
            part.face_group.copy(),
            part.face_material.copy(),
        )
    return part


def _union_parts(parts: list[ObjMesh]) -> ObjMesh:
    solids: list[manifold3d.Manifold] = []
    for part_id, source in enumerate(parts):
        part = _outward_part(source)
        validation = validate_mesh(part, require_volume=True)
        require_valid(validation, f"solid part_{part_id}")
        mesh64 = manifold3d.Mesh64(
            vert_properties=np.asarray(part.V, dtype=np.float64),
            tri_verts=np.asarray(part.F, dtype=np.uint32),
        )
        solid = manifold3d.Manifold(mesh=mesh64)
        if solid.is_empty():
            raise RuntimeError(f"solid part_{part_id} 不能转换为 Manifold")
        solids.append(solid)

    if len(solids) == 1:
        result = solids[0]
    else:
        result = reduce(lambda a, b: a + b, solids)

    if result.is_empty():
        raise RuntimeError("Manifold3D 布尔并集返回空网格")

    result_mesh = result.to_mesh64()
    vertices = np.asarray(result_mesh.vert_properties, dtype=np.float64)
    faces = np.asarray(result_mesh.tri_verts, dtype=np.int64)
    return ObjMesh(
        vertices,
        faces,
        face_object=["solid"] * len(faces),
        face_group=["union"] * len(faces),
        face_material=[""] * len(faces),
    )


def _rebuild_watertight(mesh: ObjMesh, resolution: int) -> ObjMesh:
    try:
        import point_cloud_utils as pcu
    except ImportError as exc:
        raise RuntimeError(
            "近似重建需要安装 point-cloud-utils"
        ) from exc

    if resolution < 1000:
        raise ValueError("rebuild_resolution 不能小于 1000")

    vertices, faces = pcu.make_mesh_watertight(
        mesh.V,
        mesh.F.astype(np.int32),
        resolution=int(resolution),
    )
    return ObjMesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        face_object=["rebuild"] * len(faces),
        face_group=["approximate"] * len(faces),
        face_material=[""] * len(faces),
    )


def _finish_output(
    output: ObjMesh,
    report: RepairRunReport,
    *,
    require_volume: bool,
    uniform_refine_levels: int,
) -> ObjMesh:
    """细分前后各验收一次。"""

    before = validate_mesh(output, require_volume=require_volume)
    require_valid(before, "pre_refine")
    report.pre_refine_validation = before

    levels = int(uniform_refine_levels)
    if levels < 0:
        raise ValueError("uniform_refine_levels 不能为负数")

    report.uniform_refine_levels = levels
    report.pre_refine_faces = int(len(output.F))
    if levels:
        from ops.gmsh_refine import uniform_refine

        output = uniform_refine(output, levels=levels)

    report.post_refine_faces = int(len(output.F))
    report.output_validation = validate_mesh(output, require_volume=require_volume)
    require_valid(report.output_validation, "post_refine")
    return output


def repair_mesh_data(
    mesh: ObjMesh,
    *,
    mode: str = "assembly",
    eps_v: float = 0.0,
    eps_mode: str = "relative_bbox",
    fill_holes: bool = False,
    max_hole_edges: int = 4,
    approximate_rebuild: bool = False,
    rebuild_resolution: int = 50000,
    uniform_refine_levels: int = 0,
) -> tuple[ObjMesh, RepairRunReport]:
    """按用户意图修复网格。"""

    if mode not in {"assembly", "surface", "solid"}:
        raise ValueError("mode 只能是 assembly、surface 或 solid")

    eps_v_abs = normalize_eps(eps_v, mesh.V, eps_mode)
    report = RepairRunReport(
        mode=mode,
        eps_v_abs=eps_v_abs,
        input_topology=topology_summary(mesh.V, mesh.F),
    )

    parts, report.part_reports = _repair_parts(
        mesh,
        eps_v_abs=eps_v_abs,
        fill_holes=fill_holes,
        max_hole_edges=max_hole_edges,
        connect_t_junctions=mode == "surface",
    )

    if mode in {"assembly", "surface"}:
        output = combine_parts(parts)
        output = _finish_output(
            output,
            report,
            require_volume=False,
            uniform_refine_levels=uniform_refine_levels,
        )
        report.warnings.append("surface 模式允许边界，不判断开放曲面的实体内外。")
        report.warnings.append("surface 模式不执行开放曲面的精确自交切分。")
        report.status = "success"
        return output, report

    try:
        output = _union_parts(parts)
    except (RuntimeError, ValueError) as exc:
        if not approximate_rebuild:
            raise RuntimeError(
                f"精确实体合并失败：{exc}。如允许改变几何，可开启 approximate_rebuild。"
            ) from exc

        output = _rebuild_watertight(combine_parts(parts), rebuild_resolution)
        report.approximate_rebuild = True
        report.warnings.append("已使用近似重建，输出顶点不再与输入一一对应。")

    output = _finish_output(
        output,
        report,
        require_volume=True,
        uniform_refine_levels=uniform_refine_levels,
    )
    report.status = "success"
    return output, report


def repair_single_mesh(
    mesh_in: MeshDict,
    *,
    mode: str = "surface",
    eps_v: float = 0.0,
    eps_mode: str = "relative_bbox",
    fill_holes: bool = False,
    max_hole_edges: int = 4,
    approximate_rebuild: bool = False,
    rebuild_resolution: int = 50000,
    uniform_refine_levels: int = 0,
    **_legacy: object,
) -> tuple[MeshDict, RepairRunReport]:
    """兼容原 V/F 调用接口。"""

    if "V" not in mesh_in or "F" not in mesh_in:
        raise KeyError("mesh_in 必须包含 V 和 F")

    output, report = repair_mesh_data(
        ObjMesh(mesh_in["V"], mesh_in["F"]),
        mode=mode,
        eps_v=eps_v,
        eps_mode=eps_mode,
        fill_holes=fill_holes,
        max_hole_edges=max_hole_edges,
        approximate_rebuild=approximate_rebuild,
        rebuild_resolution=rebuild_resolution,
        uniform_refine_levels=uniform_refine_levels,
    )
    return {"V": output.V, "F": output.F}, report
