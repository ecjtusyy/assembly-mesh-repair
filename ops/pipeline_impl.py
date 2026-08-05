"""网格修复主流程。"""

from __future__ import annotations

from copy import deepcopy
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
    quality_surface_remesh: bool = False
    surface_remesh_report: dict[str, object] = field(default_factory=dict)
    coordinate_canonicalization: dict[str, object] = field(default_factory=dict)
    uniform_refine_levels: int = 0
    pre_surface_faces: int | None = None
    post_surface_faces: int | None = None
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
            "quality_surface_remesh": self.quality_surface_remesh,
            "surface_remesh_report": self.surface_remesh_report,
            "coordinate_canonicalization": self.coordinate_canonicalization,
            "uniform_refine_levels": self.uniform_refine_levels,
            "pre_surface_faces": self.pre_surface_faces,
            "post_surface_faces": self.post_surface_faces,
            "pre_refine_faces": self.pre_refine_faces,
            "post_refine_faces": self.post_refine_faces,
            "warnings": self.warnings,
            "status": self.status,
        }


@dataclass(frozen=True)
class _FinishOptions:
    """输出网格的重剖分与细分参数。"""

    uniform_refine_levels: int
    quality_surface_remesh: bool
    surface_target_size: float
    min_surface_angle: float
    min_surface_mean_ratio: float
    max_surface_condition: float
    max_surface_geometry_error_relative: float
    max_surface_faces: int
    surface_smoothing_steps: int


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


def _canonicalize_pre_union_coordinates(
    mesh: ObjMesh,
    relative_tolerance: float,
) -> tuple[ObjMesh, dict[str, object]]:
    """在零件布尔并集前统一近重合坐标面，并保持装配体包围盒。"""

    relative = float(relative_tolerance)
    if relative < 0.0:
        raise ValueError("pre_union_snap_rel 不能为负数")

    diagonal = bbox_diag(mesh.V)
    absolute = diagonal * relative
    source = mesh.V
    vertices = source.copy()
    clusters: list[dict[str, object]] = []
    if absolute > 0.0:
        source_minimum = source.min(axis=0)
        source_maximum = source.max(axis=0)
        for axis in range(3):
            values, counts = np.unique(
                source[:, axis],
                return_counts=True,
            )
            start = 0
            while start < len(values):
                stop = start + 1
                while (
                    stop < len(values)
                    and float(values[stop] - values[start]) <= absolute
                ):
                    stop += 1
                cluster_values = values[start:stop]
                cluster_counts = counts[start:stop]
                if len(cluster_values) > 1:
                    if cluster_values[0] == source_minimum[axis]:
                        representative = float(cluster_values[0])
                    elif cluster_values[-1] == source_maximum[axis]:
                        representative = float(cluster_values[-1])
                    else:
                        representative = float(
                            cluster_values[int(np.argmax(cluster_counts))]
                        )
                    affected = np.isin(vertices[:, axis], cluster_values)
                    vertices[affected, axis] = representative
                    clusters.append(
                        {
                            "axis": "xyz"[axis],
                            "minimum": float(cluster_values[0]),
                            "maximum": float(cluster_values[-1]),
                            "representative": representative,
                            "distinct_coordinates": int(len(cluster_values)),
                            "affected_vertices": int(np.count_nonzero(affected)),
                        }
                    )
                start = stop

    displacement = np.linalg.norm(vertices - source, axis=1)
    maximum = float(displacement.max(initial=0.0))
    output = ObjMesh(
        vertices,
        mesh.F.copy(),
        mesh.face_object.copy(),
        mesh.face_group.copy(),
        mesh.face_material.copy(),
    )
    return output, {
        "enabled": bool(relative > 0.0),
        "relative_tolerance": relative,
        "absolute_tolerance": float(absolute),
        "clusters": clusters,
        "moved_vertices": int(np.count_nonzero(displacement > 0.0)),
        "maximum_displacement": maximum,
        "maximum_relative_displacement": (
            maximum / diagonal if diagonal > 0.0 else 0.0
        ),
        "bounding_box_preserved": bool(
            np.array_equal(source.min(axis=0), vertices.min(axis=0))
            and np.array_equal(source.max(axis=0), vertices.max(axis=0))
        ),
    }


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
        validation = validate_mesh(
            output,
            require_volume=False,
            check_self_intersections=True,
        )
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
        validation = validate_mesh(
            part,
            require_volume=True,
            check_self_intersections=True,
        )
        require_valid(validation, f"solid part_{part_id}")
        mesh64 = manifold3d.Mesh64(
            vert_properties=np.array(
                part.V,
                dtype=np.float64,
                copy=True,
                order="C",
            ),
            tri_verts=np.array(
                part.F,
                dtype=np.uint64,
                copy=True,
                order="C",
            ),
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


def _load_point_cloud_utils():
    """加载近似重建依赖，并保留真实故障原因。"""

    try:
        import point_cloud_utils as pcu
    except ModuleNotFoundError as exc:
        if exc.name == "point_cloud_utils":
            raise RuntimeError(
                "近似重建需要安装 point-cloud-utils"
            ) from exc
        raise RuntimeError(
            f"point-cloud-utils 已安装，但依赖 {exc.name!r} 无法加载"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            f"point-cloud-utils 已安装，但本机二进制依赖无法加载：{exc}"
        ) from exc
    return pcu


def _rebuild_watertight(mesh: ObjMesh, resolution: int) -> ObjMesh:
    pcu = _load_point_cloud_utils()

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
    options: _FinishOptions,
) -> ObjMesh:
    """质量重剖分和均匀细分前后分别验收。"""

    initial = validate_mesh(
        output,
        require_volume=require_volume,
        check_self_intersections=require_volume,
    )
    require_valid(initial, "pre_surface_remesh")
    report.pre_surface_faces = int(len(output.F))
    report.quality_surface_remesh = bool(options.quality_surface_remesh)

    if options.quality_surface_remesh:
        from ops.gmsh_quality_remesh import (
            SurfaceRemeshOptions,
            remesh_planar_surface,
        )

        output, report.surface_remesh_report = remesh_planar_surface(
            output,
            options=SurfaceRemeshOptions(
                target_size=options.surface_target_size,
                min_angle=options.min_surface_angle,
                min_mean_ratio=options.min_surface_mean_ratio,
                max_condition=options.max_surface_condition,
                max_geometry_error_relative=(
                    options.max_surface_geometry_error_relative
                ),
                max_faces=options.max_surface_faces,
                smoothing_steps=options.surface_smoothing_steps,
            ),
        )
        before = validate_mesh(
            output,
            require_volume=require_volume,
            check_self_intersections=require_volume,
        )
        require_valid(before, "post_surface_remesh")
    else:
        before = initial
    report.post_surface_faces = int(len(output.F))
    report.pre_refine_validation = before

    levels = int(options.uniform_refine_levels)
    if levels < 0:
        raise ValueError("uniform_refine_levels 不能为负数")

    report.uniform_refine_levels = levels
    report.pre_refine_faces = int(len(output.F))
    if levels:
        from ops.gmsh_refine import uniform_refine

        output = uniform_refine(output, levels=levels)

    report.post_refine_faces = int(len(output.F))
    if not levels:
        report.output_validation = deepcopy(before)
    else:
        report.output_validation = validate_mesh(
            output,
            require_volume=require_volume,
            check_self_intersections=False,
        )
        report.output_validation["self_intersections"] = {
            **before["self_intersections"],
            "inherited_from_pre_refine": True,
        }
    require_valid(report.output_validation, "post_refine")
    return output


def _try_preserve_valid_solid(
    mesh: ObjMesh,
    report: RepairRunReport,
    options: _FinishOptions,
) -> ObjMesh | None:
    """合法实体无需布尔重建时，复制网格并锁定原始边界。"""

    validation = validate_mesh(
        mesh,
        require_volume=True,
        check_self_intersections=True,
    )
    if not bool(validation["success"]):
        return None

    output = ObjMesh(
        mesh.V.copy(),
        mesh.F.copy(),
        mesh.face_object.copy(),
        mesh.face_group.copy(),
        mesh.face_material.copy(),
    )
    output = _finish_output(
        output,
        report,
        require_volume=True,
        options=options,
    )
    if options.quality_surface_remesh:
        report.warnings.append(
            "输入已经是合法闭合实体，已跳过布尔重建；"
            "表面三角形已在原平面和特征边约束内重剖分。"
        )
    else:
        report.warnings.append(
            "输入已经是合法闭合实体，已跳过修复和布尔重建，"
            "以锁定原始边界。"
        )
    return output


def _finish_non_solid_output(
    parts: list[ObjMesh],
    report: RepairRunReport,
    mode: str,
    options: _FinishOptions,
) -> ObjMesh:
    """合并非实体分件，并完成输出验收。"""

    output = _finish_output(
        combine_parts(parts),
        report,
        require_volume=False,
        options=options,
    )
    if mode == "surface":
        report.warnings.append(
            "surface 模式允许边界，不判断开放曲面的实体内外。"
        )
        report.warnings.append(
            "surface 模式不执行开放曲面的精确自交切分。"
        )
    return output


def _build_solid_output(
    parts: list[ObjMesh],
    report: RepairRunReport,
    *,
    approximate_rebuild: bool,
    rebuild_resolution: int,
    options: _FinishOptions,
) -> ObjMesh:
    """优先精确合并实体，失败后按用户选择决定是否近似重建。"""

    try:
        output = _union_parts(parts)
    except (RuntimeError, ValueError) as exc:
        if not approximate_rebuild:
            raise RuntimeError(
                f"精确实体合并失败：{exc}。如允许改变几何，"
                "可开启 approximate_rebuild。"
            ) from exc

        output = _rebuild_watertight(combine_parts(parts), rebuild_resolution)
        report.approximate_rebuild = True
        report.warnings.append(
            "已使用近似重建，输出顶点不再与输入一一对应。"
        )

    return _finish_output(
        output,
        report,
        require_volume=True,
        options=options,
    )


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
    quality_surface_remesh: bool = False,
    surface_target_size: float = 0.0,
    min_surface_angle: float = 15.0,
    min_surface_mean_ratio: float = 0.2,
    max_surface_condition: float = 10.0,
    max_surface_geometry_error_relative: float = 1e-10,
    max_surface_faces: int = 1_000_000,
    surface_smoothing_steps: int = 5,
    pre_union_snap_relative: float = 0.0,
) -> tuple[ObjMesh, RepairRunReport]:
    """按用户意图修复网格。"""

    if mode not in {"assembly", "surface", "solid"}:
        raise ValueError("mode 只能是 assembly、surface 或 solid")
    if pre_union_snap_relative < 0.0:
        raise ValueError("pre_union_snap_rel 不能为负数")

    finish_options = _FinishOptions(
        uniform_refine_levels=uniform_refine_levels,
        quality_surface_remesh=quality_surface_remesh,
        surface_target_size=surface_target_size,
        min_surface_angle=min_surface_angle,
        min_surface_mean_ratio=min_surface_mean_ratio,
        max_surface_condition=max_surface_condition,
        max_surface_geometry_error_relative=(
            max_surface_geometry_error_relative
        ),
        max_surface_faces=max_surface_faces,
        surface_smoothing_steps=surface_smoothing_steps,
    )
    eps_v_abs = normalize_eps(eps_v, mesh.V, eps_mode)
    report = RepairRunReport(
        mode=mode,
        eps_v_abs=eps_v_abs,
        input_topology=topology_summary(mesh.V, mesh.F),
    )
    report.coordinate_canonicalization = {
        "enabled": False,
        "relative_tolerance": float(pre_union_snap_relative),
    }

    preserve_input_boundary = (
        eps_v_abs == 0.0
        and not fill_holes
        and not approximate_rebuild
        and uniform_refine_levels == 0
    )
    if mode == "solid" and preserve_input_boundary:
        output = _try_preserve_valid_solid(
            mesh,
            report,
            finish_options,
        )
        if output is not None:
            report.status = "success"
            return output, report

    repair_input = mesh
    if mode == "solid" and pre_union_snap_relative > 0.0:
        repair_input, report.coordinate_canonicalization = (
            _canonicalize_pre_union_coordinates(
                mesh,
                pre_union_snap_relative,
            )
        )

    parts, report.part_reports = _repair_parts(
        repair_input,
        eps_v_abs=eps_v_abs,
        fill_holes=fill_holes,
        max_hole_edges=max_hole_edges,
        connect_t_junctions=mode == "surface",
    )

    if mode in {"assembly", "surface"}:
        output = _finish_non_solid_output(
            parts,
            report,
            mode,
            finish_options,
        )
        report.status = "success"
        return output, report

    output = _build_solid_output(
        parts,
        report,
        approximate_rebuild=approximate_rebuild,
        rebuild_resolution=rebuild_resolution,
        options=finish_options,
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
    quality_surface_remesh: bool = False,
    surface_target_size: float = 0.0,
    min_surface_angle: float = 15.0,
    min_surface_mean_ratio: float = 0.2,
    max_surface_condition: float = 10.0,
    max_surface_geometry_error_relative: float = 1e-10,
    max_surface_faces: int = 1_000_000,
    surface_smoothing_steps: int = 5,
    pre_union_snap_relative: float = 0.0,
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
        quality_surface_remesh=quality_surface_remesh,
        surface_target_size=surface_target_size,
        min_surface_angle=min_surface_angle,
        min_surface_mean_ratio=min_surface_mean_ratio,
        max_surface_condition=max_surface_condition,
        max_surface_geometry_error_relative=max_surface_geometry_error_relative,
        max_surface_faces=max_surface_faces,
        surface_smoothing_steps=surface_smoothing_steps,
        pre_union_snap_relative=pre_union_snap_relative,
    )
    return {"V": output.V, "F": output.F}, report
