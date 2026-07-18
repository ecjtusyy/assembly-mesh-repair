"""固定输入边界的 TetGen 单区域四面体生成。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np

from mesh.io_obj import ObjMesh
from ops.validation import require_valid, validate_mesh
from volume.boundary import boundary_lock_report, boundary_quality_report
from volume.msh_io import write_msh22
from volume.quality import quality_report, signed_six_volumes
from volume.tetra_mesh import TetraMesh
from volume.vtk_io import write_quality_vtk


_TETGEN_CWD_LOCK = Lock()


@dataclass(frozen=True)
class StrictTetrahedralizationOptions:
    target_size: float = 0.0
    min_quality: float = 0.05
    max_relative_volume_error: float = 1e-10
    radius_edge_ratio: float = 2.0
    min_dihedral: float = 0.0
    max_steiner_points: int = 10_000
    optimize: bool = True

    def validate(self) -> None:
        if self.target_size < 0.0:
            raise ValueError("target_size 不能为负数")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_tet_quality 必须在 [0, 1] 内")
        if self.max_relative_volume_error < 0.0:
            raise ValueError("max_volume_error_rel 不能为负数")
        if self.radius_edge_ratio <= 1.0:
            raise ValueError("tetgen_radius_edge_ratio 必须大于 1")
        if not 0.0 <= self.min_dihedral < 180.0:
            raise ValueError("tetgen_min_dihedral 必须在 [0, 180) 内")
        if self.max_steiner_points < 0:
            raise ValueError("max_steiner_points 不能为负数")


def _append_boundary_errors(
    report: dict[str, object],
    boundary: dict[str, object],
) -> None:
    if bool(boundary["success"]):
        return
    hard_errors = list(report["hard_errors"])
    for error in boundary["errors"]:
        if error not in hard_errors:
            hard_errors.append(str(error))
    report["hard_errors"] = hard_errors
    report["errors"] = hard_errors + list(report["threshold_errors"])
    report["hard_valid"] = False
    report["success"] = False


def _append_threshold_error(report: dict[str, object], error: str) -> None:
    threshold_errors = list(report["threshold_errors"])
    if error not in threshold_errors:
        threshold_errors.append(error)
    report["threshold_errors"] = threshold_errors
    report["errors"] = list(report["hard_errors"]) + threshold_errors
    report["success"] = False


def _boundary_name(source: ObjMesh, face_id: int) -> str:
    group = source.face_group[face_id].strip()
    if group and group != "default":
        return group
    object_name = source.face_object[face_id].strip()
    if object_name and object_name != "default":
        return object_name
    return "boundary"


def _boundary_names(mesh: TetraMesh, source: ObjMesh) -> list[str]:
    source_names = {
        tuple(sorted(int(value) for value in face)): _boundary_name(source, face_id)
        for face_id, face in enumerate(source.F)
    }
    return [
        source_names.get(
            tuple(sorted(int(value) for value in face)),
            "boundary",
        )
        for face in mesh.boundary_faces
    ]


def tetrahedralize_strict(
    source: ObjMesh,
    msh_path: str | Path,
    vtk_path: str | Path,
    *,
    options: StrictTetrahedralizationOptions | None = None,
    domain_name: str = "domain",
) -> tuple[TetraMesh, dict[str, object]]:
    """冻结输入 V/F，只允许 TetGen 在体内增加节点。"""

    settings = options or StrictTetrahedralizationOptions()
    settings.validate()
    surface_validation = validate_mesh(
        source,
        require_volume=True,
        check_self_intersections=True,
    )
    require_valid(surface_validation, "strict_boundary_input")
    component_count = int(
        surface_validation["topology"]["edge_component_count"]
    )
    if component_count != 1:
        raise RuntimeError(
            "strict_boundary_input 只支持一个连通闭合边界；"
            f"当前检测到 {component_count} 个边界分量"
        )
    try:
        import tetgen
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "严格边界四面体生成需要安装 requirements-tetgen.txt"
        ) from exc

    maximum_volume = (
        settings.target_size**3 / (6.0 * np.sqrt(2.0))
        if settings.target_size > 0.0
        else None
    )

    generator = tetgen.TetGen(
        np.array(source.V, dtype=np.float64, copy=True, order="C"),
        np.array(source.F, dtype=np.int32, copy=True, order="C"),
    )
    try:
        with _TETGEN_CWD_LOCK, TemporaryDirectory(prefix="tetgen-strict-") as work:
            previous = Path.cwd()
            try:
                os.chdir(work)
                nodes, tetrahedra, _, _ = generator.tetrahedralize(
                    plc=True,
                    quality=settings.optimize,
                    nobisect=True,
                    nomergefacet=True,
                    nomergevertex=True,
                    fixedvolume=maximum_volume is not None,
                    maxvolume=maximum_volume if maximum_volume is not None else -1.0,
                    minratio=settings.radius_edge_ratio,
                    mindihedral=settings.min_dihedral,
                    steinerleft=settings.max_steiner_points,
                    quiet=True,
                )
            finally:
                os.chdir(previous)
    except RuntimeError as exc:
        raise RuntimeError(f"TetGen 严格边界生成失败：{exc}") from exc

    mesh = TetraMesh(
        np.asarray(nodes, dtype=np.float64),
        np.asarray(tetrahedra, dtype=np.int64),
        np.asarray(generator.trifaces, dtype=np.int64),
    )
    boundary = boundary_lock_report(mesh, source)
    report, quality = quality_report(
        mesh,
        source,
        min_quality=settings.min_quality,
        max_relative_deviation=0.0,
        max_relative_volume_error=settings.max_relative_volume_error,
    )
    _append_boundary_errors(report, boundary)
    actual_maximum_volume = float(
        np.max(np.abs(signed_six_volumes(mesh))) / 6.0
    )
    size_satisfied = bool(
        maximum_volume is None
        or actual_maximum_volume <= maximum_volume * (1.0 + 1e-8)
    )
    if not size_satisfied:
        _append_threshold_error(report, "tetra_size_above_target")
    msh_output = Path(msh_path)
    vtk_output = Path(vtk_path)
    boundary_groups = write_msh22(
        msh_output,
        mesh,
        domain_name=domain_name,
        boundary_names=_boundary_names(mesh, source),
    )
    write_quality_vtk(vtk_output, mesh, quality)
    report.update(
        {
            "generator": "TetGen",
            "boundary_mode": "strict",
            "single_region": True,
            "domain_name": domain_name or "domain",
            "boundary_physical_groups": boundary_groups,
            "boundary_labels": {
                "objects": sorted(set(source.face_object)),
                "groups": sorted(set(source.face_group)),
                "materials": sorted(
                    name for name in set(source.face_material) if name
                ),
            },
            "target_size": float(settings.target_size),
            "requested_maximum_tetra_volume": (
                float(maximum_volume) if maximum_volume is not None else None
            ),
            "actual_maximum_tetra_volume": actual_maximum_volume,
            "target_size_satisfied": size_satisfied,
            "optimized": bool(settings.optimize),
            "interior_steiner_vertices": int(len(mesh.V) - len(source.V)),
            "boundary_lock": boundary,
            "input_boundary_quality": boundary_quality_report(source),
            "surface_validation": surface_validation,
            "msh_output": str(msh_output),
            "quality_vtk_output": str(vtk_output),
        }
    )
    return mesh, report
