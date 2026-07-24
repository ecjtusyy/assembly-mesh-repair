"""保持平面几何的 Gmsh 表面质量重剖分。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import trimesh

from mesh.io_obj import ObjMesh
from mesh.surface_quality import surface_quality_report
from ops.planar_patches import PlanarPatch, detect_planar_patches


@dataclass(frozen=True)
class SurfaceRemeshOptions:
    target_size: float = 0.0
    min_angle: float = 15.0
    min_mean_ratio: float = 0.2
    max_condition: float = 10.0
    max_geometry_error_relative: float = 1e-10
    max_faces: int = 1_000_000
    smoothing_steps: int = 5

    def validate(self) -> None:
        if self.target_size < 0.0:
            raise ValueError("surface_target_size 不能为负数")
        if not 0.0 <= self.min_angle < 60.0:
            raise ValueError("min_surface_angle 必须在 [0, 60) 内")
        if not 0.0 <= self.min_mean_ratio <= 1.0:
            raise ValueError("min_surface_mean_ratio 必须在 [0, 1] 内")
        if self.max_condition < 1.0:
            raise ValueError("max_surface_condition 不能小于 1")
        if self.max_geometry_error_relative < 0.0:
            raise ValueError("max_surface_geometry_error_rel 不能为负数")
        if self.max_faces <= 0:
            raise ValueError("max_surface_faces 必须大于 0")
        if self.smoothing_steps < 0:
            raise ValueError("surface_smoothing_steps 不能为负数")


def _patch_sizes(
    patches: list[PlanarPatch],
    target_size: float,
) -> list[float]:
    if target_size > 0.0:
        values = [target_size] * len(patches)
    else:
        raw = np.asarray([patch.target_size for patch in patches])
        base = float(np.median(raw))
        values = np.clip(raw, base / 4.0, base * 4.0).tolist()
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("共面片区无法计算有效目标尺寸")
    return values


def _point_sizes(
    patches: list[PlanarPatch],
    patch_sizes: list[float],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for patch, size in zip(patches, patch_sizes, strict=True):
        for loop in patch.loops:
            for vertex_id in loop:
                result[vertex_id] = min(result.get(vertex_id, size), size)
    return result


def _estimated_face_upper_bound(
    patches: list[PlanarPatch],
    patch_sizes: list[float],
) -> int:
    equilateral_area = np.sqrt(3.0) / 4.0
    estimate = sum(
        patch.area / (equilateral_area * size * size)
        for patch, size in zip(patches, patch_sizes, strict=True)
    )
    return int(np.ceil(16.0 * estimate))


def _feature_size_report(
    source: ObjMesh,
    patches: list[PlanarPatch],
    patch_sizes: list[float],
) -> dict[str, float]:
    edges = {
        tuple(sorted((first, second)))
        for patch in patches
        for loop in patch.loops
        for first, second in zip(loop, loop[1:] + loop[:1], strict=True)
    }
    lengths = np.asarray(
        [
            np.linalg.norm(source.V[second] - source.V[first])
            for first, second in edges
        ]
    )
    minimum = float(lengths.min())
    target = float(min(patch_sizes))
    return {
        "minimum_length": minimum,
        "p01_length": float(np.quantile(lengths, 0.01)),
        "median_length": float(np.median(lengths)),
        "minimum_length_to_target_size": minimum / target,
    }


def _extract_surface(
    gmsh: object,
    surface_patches: dict[int, int],
    patches: list[PlanarPatch],
) -> tuple[ObjMesh, np.ndarray]:
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    vertices = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))
    order = np.argsort(node_tags)
    sorted_tags = node_tags[order]

    faces: list[np.ndarray] = []
    patch_ids: list[np.ndarray] = []
    labels: list[tuple[str, str, str]] = []
    for surface_tag, patch_id in surface_patches.items():
        element_types, _, element_nodes = gmsh.model.mesh.getElements(2, surface_tag)
        for element_type, nodes in zip(element_types, element_nodes):
            if int(element_type) != 2:
                continue
            connectivity = np.asarray(nodes, dtype=np.int64).reshape((-1, 3))
            positions = np.searchsorted(sorted_tags, connectivity)
            mapped = order[positions]
            points = vertices[mapped]
            normals = np.cross(
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
            )
            flip = (normals @ patches[patch_id].normal) < 0.0
            mapped[flip] = mapped[flip][:, [0, 2, 1]]
            faces.append(mapped)
            patch_ids.append(np.full(len(mapped), patch_id, dtype=np.int64))
            labels.extend([patches[patch_id].label] * len(mapped))

    if not faces:
        raise RuntimeError("Gmsh 表面重剖分没有生成一阶三角形")
    all_faces = np.vstack(faces)
    all_patch_ids = np.concatenate(patch_ids)
    used, inverse = np.unique(all_faces, return_inverse=True)
    compact_faces = inverse.reshape((-1, 3))
    compact_vertices = vertices[used]
    objects, groups, materials = zip(*labels, strict=True)
    return (
        ObjMesh(
            compact_vertices,
            compact_faces,
            list(objects),
            list(groups),
            list(materials),
        ),
        all_patch_ids,
    )


def _feature_error(
    gmsh: object,
    curves: dict[tuple[int, int], int],
    vertices: np.ndarray,
    diagonal: float,
) -> float:
    maximum = 0.0
    scale = max(diagonal, np.finfo(float).tiny)
    for (first, second), curve_tag in curves.items():
        _, coordinates, _ = gmsh.model.mesh.getNodes(
            1,
            curve_tag,
            includeBoundary=True,
        )
        points = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))
        start = vertices[first]
        delta = vertices[second] - start
        squared = float(np.dot(delta, delta))
        if not len(points) or squared <= 0.0:
            continue
        parameter = ((points - start) @ delta) / squared
        projection = start + parameter[:, None] * delta
        distance = np.linalg.norm(points - projection, axis=1)
        outside = np.maximum.reduce(
            [
                np.zeros(len(parameter)),
                -parameter,
                parameter - 1.0,
            ]
        )
        maximum = max(
            maximum,
            float(distance.max(initial=0.0)),
            float(outside.max(initial=0.0) * np.sqrt(squared)),
        )
    return maximum / scale


def _geometry_report(
    source: ObjMesh,
    output: ObjMesh,
    patches: list[PlanarPatch],
    output_patch_ids: np.ndarray,
    *,
    feature_error_relative: float,
    threshold: float,
) -> dict[str, object]:
    diagonal = float(np.linalg.norm(source.V.max(axis=0) - source.V.min(axis=0)))
    scale = max(diagonal, np.finfo(float).tiny)
    triangles = output.V[output.F]
    maximum_plane_error = 0.0
    maximum_area_error = 0.0
    for patch_id, patch in enumerate(patches):
        mask = output_patch_ids == patch_id
        patch_triangles = triangles[mask]
        if not len(patch_triangles):
            maximum_area_error = float("inf")
            continue
        distance = np.abs((patch_triangles - patch.origin) @ patch.normal)
        maximum_plane_error = max(
            maximum_plane_error,
            float(distance.max(initial=0.0)),
        )
        area = 0.5 * float(
            np.linalg.norm(
                np.cross(
                    patch_triangles[:, 1] - patch_triangles[:, 0],
                    patch_triangles[:, 2] - patch_triangles[:, 0],
                ),
                axis=1,
            ).sum()
        )
        relative = abs(area - patch.area) / max(patch.area, np.finfo(float).tiny)
        maximum_area_error = max(maximum_area_error, relative)

    source_trimesh = trimesh.Trimesh(source.V, source.F, process=False)
    output_trimesh = trimesh.Trimesh(output.V, output.F, process=False)
    volume_checked = bool(source_trimesh.is_watertight)
    source_volume = abs(float(source_trimesh.volume))
    output_volume = abs(float(output_trimesh.volume))
    volume_error = (
        abs(output_volume - source_volume)
        / max(source_volume, np.finfo(float).tiny)
        if volume_checked
        else 0.0
    )
    plane_error_relative = maximum_plane_error / scale
    errors: list[str] = []
    if feature_error_relative > threshold:
        errors.append("feature_curve_changed")
    if plane_error_relative > threshold:
        errors.append("surface_left_original_plane")
    if maximum_area_error > threshold:
        errors.append("planar_patch_area_changed")
    if volume_checked and volume_error > threshold:
        errors.append("surface_volume_changed")
    return {
        "success": not errors,
        "errors": errors,
        "relative_threshold": float(threshold),
        "maximum_feature_curve_error_relative": float(feature_error_relative),
        "maximum_plane_error_relative": float(plane_error_relative),
        "maximum_patch_area_error_relative": float(maximum_area_error),
        "volume_checked": volume_checked,
        "volume_error_relative": float(volume_error),
        "source_volume": float(source_volume),
        "output_volume": float(output_volume),
    }


def remesh_planar_surface(
    source: ObjMesh,
    *,
    options: SurfaceRemeshOptions | None = None,
) -> tuple[ObjMesh, dict[str, object]]:
    """把共面片区重建为精确平面后重新划分三角形。"""

    settings = options or SurfaceRemeshOptions()
    settings.validate()
    patches, _ = detect_planar_patches(source)
    patch_sizes = _patch_sizes(patches, settings.target_size)
    feature_sizes = _feature_size_report(source, patches, patch_sizes)
    estimated_faces = _estimated_face_upper_bound(patches, patch_sizes)
    if estimated_faces > settings.max_faces:
        raise RuntimeError(
            f"预计需要约 {estimated_faces} 个表面三角形，"
            f"超过上限 {settings.max_faces}；请增大 surface_target_size"
        )
    point_sizes = _point_sizes(patches, patch_sizes)
    diagonal = float(np.linalg.norm(source.V.max(axis=0) - source.V.min(axis=0)))

    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "表面质量重剖分需要 Gmsh 及系统 libGLU、libXft"
        ) from exc

    started_here = not bool(gmsh.isInitialized())
    if started_here:
        gmsh.initialize()
    feature_error_relative = float("inf")
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(patch_sizes))
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(patch_sizes))
        gmsh.model.add(f"surface_quality_{uuid.uuid4().hex}")

        point_tags = {
            vertex_id: gmsh.model.geo.addPoint(
                *source.V[vertex_id],
                point_sizes[vertex_id],
            )
            for vertex_id in sorted(point_sizes)
        }
        curves: dict[tuple[int, int], int] = {}
        surface_patches: dict[int, int] = {}
        for patch_id, patch in enumerate(patches):
            loop_tags: list[int] = []
            for loop in patch.loops:
                curve_tags: list[int] = []
                for first, second in zip(
                    loop,
                    loop[1:] + loop[:1],
                    strict=True,
                ):
                    key = tuple(sorted((first, second)))
                    if key not in curves:
                        curves[key] = gmsh.model.geo.addLine(
                            point_tags[key[0]],
                            point_tags[key[1]],
                        )
                    curve_tag = curves[key]
                    oriented = (
                        curve_tag if key == (first, second) else -curve_tag
                    )
                    curve_tags.append(oriented)
                loop_tags.append(
                    gmsh.model.geo.addCurveLoop(curve_tags, reorient=True)
                )
            surface_tag = gmsh.model.geo.addPlaneSurface(loop_tags)
            surface_patches[surface_tag] = patch_id

        gmsh.model.geo.synchronize()
        for surface_tag in surface_patches:
            gmsh.model.mesh.setAlgorithm(2, surface_tag, 6)
            gmsh.model.mesh.setSmoothing(
                2,
                surface_tag,
                settings.smoothing_steps,
            )
        gmsh.model.mesh.generate(2)
        dim_tags = [(2, tag) for tag in surface_patches]
        if settings.smoothing_steps:
            gmsh.model.mesh.optimize(
                "Relocate2D",
                False,
                settings.smoothing_steps,
                dim_tags,
            )
        feature_error_relative = _feature_error(
            gmsh,
            curves,
            source.V,
            diagonal,
        )
        output, output_patch_ids = _extract_surface(
            gmsh,
            surface_patches,
            patches,
        )
    finally:
        if started_here and gmsh.isInitialized():
            gmsh.finalize()
        elif gmsh.isInitialized():
            gmsh.model.remove()

    if len(output.F) > settings.max_faces:
        raise RuntimeError(
            f"表面重剖分生成 {len(output.F)} 个三角形，"
            f"超过上限 {settings.max_faces}"
        )
    before = surface_quality_report(
        source.V,
        source.F,
        min_angle=settings.min_angle,
        min_mean_ratio=settings.min_mean_ratio,
        max_condition=settings.max_condition,
    )
    after = surface_quality_report(
        output.V,
        output.F,
        min_angle=settings.min_angle,
        min_mean_ratio=settings.min_mean_ratio,
        max_condition=settings.max_condition,
    )
    geometry = _geometry_report(
        source,
        output,
        patches,
        output_patch_ids,
        feature_error_relative=feature_error_relative,
        threshold=settings.max_geometry_error_relative,
    )
    report = {
        "success": bool(after["success"] and geometry["success"]),
        "generator": "Gmsh PlaneSurface Frontal-Delaunay",
        "planar_patches": int(len(patches)),
        "estimated_face_upper_bound": int(estimated_faces),
        "target_size": (
            float(settings.target_size) if settings.target_size > 0.0 else "auto"
        ),
        "patch_target_size": {
            "minimum": float(min(patch_sizes)),
            "maximum": float(max(patch_sizes)),
            "median": float(np.median(patch_sizes)),
        },
        "protected_feature_edges": feature_sizes,
        "before": before,
        "after": after,
        "geometry": geometry,
    }
    if not report["success"]:
        errors = list(geometry["errors"])
        if not after["success"]:
            errors.append("surface_quality_below_threshold")
        raise RuntimeError(
            "表面质量重剖分验收失败："
            + ", ".join(errors)
            + f"；坏面 {after['bad_faces']}，"
            + "最小角 "
            + f"{after['minimum_angle_degrees']['minimum']:.6g}°，"
            + "最短受保护特征边 "
            + f"{feature_sizes['minimum_length']:.6g}"
        )
    return output, report
