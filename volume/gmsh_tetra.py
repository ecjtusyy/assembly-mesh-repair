"""从闭合三角表面生成并验收单区域四面体网格。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mesh.io_obj import ObjMesh
from mesh.mesh import Mesh
from ops.stitch import area_threshold_from_mesh, cleanup_topology
from ops.validation import require_valid, validate_mesh
from volume.quality import quality_report
from volume.tetra_mesh import TetraMesh
from volume.vtk_io import write_quality_vtk


@dataclass(frozen=True)
class TetrahedralizationOptions:
    target_size: float = 0.0
    surface_angle: float = 40.0
    min_quality: float = 0.05
    max_relative_deviation: float = 1e-6
    max_relative_volume_error: float = 1e-6
    optimize: bool = True

    def validate(self) -> None:
        if self.target_size < 0.0:
            raise ValueError("target_size 不能为负数")
        if not 0.0 < self.surface_angle <= 180.0:
            raise ValueError("surface_angle 必须在 (0, 180] 内")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_tet_quality 必须在 [0, 1] 内")
        if self.max_relative_deviation < 0.0:
            raise ValueError("max_geometry_deviation_rel 不能为负数")
        if self.max_relative_volume_error < 0.0:
            raise ValueError("max_volume_error_rel 不能为负数")


def _map_connectivity(node_tags: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    order = np.argsort(node_tags)
    sorted_tags = node_tags[order]
    positions = np.searchsorted(sorted_tags, connectivity)
    if np.any(positions >= len(sorted_tags)) or not np.array_equal(
        sorted_tags[positions], connectivity
    ):
        raise RuntimeError("Gmsh 单元引用了未知节点")
    return order[positions]


def _extract_mesh(gmsh: object, volume_tag: int) -> TetraMesh:
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    vertices = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))

    element_types, _, element_nodes = gmsh.model.mesh.getElements(3, volume_tag)
    tetra_nodes: np.ndarray | None = None
    for element_type, nodes in zip(element_types, element_nodes):
        if int(element_type) == 4:
            tetra_nodes = np.asarray(nodes, dtype=np.int64).reshape((-1, 4))
            break
    if tetra_nodes is None:
        raise RuntimeError("Gmsh 没有生成一阶四面体单元")

    boundary_parts: list[np.ndarray] = []
    surface_types, _, surface_nodes = gmsh.model.mesh.getElements(2)
    for element_type, nodes in zip(surface_types, surface_nodes):
        if int(element_type) == 2:
            boundary_parts.append(np.asarray(nodes, dtype=np.int64).reshape((-1, 3)))
    if not boundary_parts:
        raise RuntimeError("Gmsh 没有导出体网格边界三角形")

    return TetraMesh(
        vertices,
        _map_connectivity(node_tags, tetra_nodes),
        _map_connectivity(node_tags, np.vstack(boundary_parts)),
    )


def _prepare_surface(source: ObjMesh) -> tuple[ObjMesh, dict[str, object]]:
    """清除布尔输出中的近重复点和极薄三角片。"""

    diagonal = float(np.linalg.norm(source.V.max(axis=0) - source.V.min(axis=0)))
    attempts: list[dict[str, object]] = []
    for relative_tolerance in (1e-7, 5e-8, 1e-8, 5e-9, 0.0):
        working = Mesh(source.V.copy(), source.F.copy())
        changes = cleanup_topology(
            working,
            eps_v=diagonal * relative_tolerance,
            area_eps=area_threshold_from_mesh(working),
        )
        prepared = ObjMesh(working.V, working.F)
        validation = validate_mesh(
            prepared,
            require_volume=True,
            check_self_intersections=True,
        )
        attempts.append(
            {
                "relative_weld_tolerance": relative_tolerance,
                "success": bool(validation["success"]),
                "errors": list(validation["errors"]),
            }
        )
        if bool(validation["success"]):
            return prepared, {
                "relative_weld_tolerance": relative_tolerance,
                "changes": changes.as_dict(),
                "validation": validation,
                "attempts": attempts,
            }

    require_valid(validation, "tetra_surface_cleanup")
    raise AssertionError("unreachable")


def tetrahedralize(
    source: ObjMesh,
    msh_path: str | Path,
    vtk_path: str | Path,
    *,
    options: TetrahedralizationOptions | None = None,
    domain_name: str = "domain",
) -> tuple[TetraMesh, dict[str, object]]:
    """按 Gmsh 官方离散表面重参数化流程生成单区域体网格。"""

    settings = options or TetrahedralizationOptions()
    settings.validate()
    prepared, preparation_report = _prepare_surface(source)
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "四面体生成需要 Gmsh；请安装 requirements-gmsh.txt 及系统 libGLU"
        ) from exc

    msh_output = Path(msh_path)
    vtk_output = Path(vtk_path)
    msh_output.parent.mkdir(parents=True, exist_ok=True)
    vtk_output.parent.mkdir(parents=True, exist_ok=True)
    diagonal = float(np.linalg.norm(source.V.max(axis=0) - source.V.min(axis=0)))
    target_size = settings.target_size or diagonal / 8.0
    if target_size <= 0.0:
        raise ValueError("输入表面的包围盒尺寸必须大于 0")

    started_here = not bool(gmsh.isInitialized())
    if started_here:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_size)
        gmsh.model.add("assembly_mesh_repair")
        surface_tag = gmsh.model.addDiscreteEntity(2)
        node_tags = np.arange(1, len(prepared.V) + 1, dtype=np.int64)
        gmsh.model.mesh.addNodes(2, surface_tag, node_tags, prepared.V.reshape(-1))
        gmsh.model.mesh.addElementsByType(
            surface_tag,
            2,
            [],
            (prepared.F.astype(np.int64) + 1).reshape(-1),
        )
        gmsh.model.mesh.classifySurfaces(
            np.deg2rad(settings.surface_angle),
            True,
            False,
            np.pi,
        )
        gmsh.model.mesh.createGeometry()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        if not surfaces:
            raise RuntimeError("Gmsh 没有从输入三角面识别出边界曲面")
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        volume_tag = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        boundary_group = gmsh.model.addPhysicalGroup(2, surfaces)
        gmsh.model.setPhysicalName(2, boundary_group, "boundary")
        domain_group = gmsh.model.addPhysicalGroup(3, [volume_tag])
        gmsh.model.setPhysicalName(3, domain_group, domain_name or "domain")
        gmsh.model.mesh.generate(3)
        if settings.optimize:
            gmsh.model.mesh.optimize("Netgen")
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(msh_output))
        mesh = _extract_mesh(gmsh, volume_tag)
    finally:
        if started_here and gmsh.isInitialized():
            gmsh.finalize()

    report, quality = quality_report(
        mesh,
        source,
        min_quality=settings.min_quality,
        max_relative_deviation=settings.max_relative_deviation,
        max_relative_volume_error=settings.max_relative_volume_error,
    )
    write_quality_vtk(vtk_output, mesh, quality)
    report.update(
        {
            "generator": "Gmsh",
            "single_region": True,
            "domain_name": domain_name or "domain",
            "boundary_name": "boundary",
            "target_size": float(target_size),
            "surface_angle_degrees": float(settings.surface_angle),
            "optimized": bool(settings.optimize),
            "surface_preparation": preparation_report,
            "msh_output": str(msh_output),
            "quality_vtk_output": str(vtk_output),
        }
    )
    return mesh, report
