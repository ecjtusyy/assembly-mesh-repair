"""根据自交检测和拓扑诊断，判断网格是否适合进入下一步前处理。"""


def _int_metric(topology, key, default=0):
    """读取 topology 中的整数指标。
    如果字段不存在或不是数字，就使用默认值。"""
    try:
        return int(topology.get(key, default))
    except (TypeError, ValueError):
        return default


def _add_once(items, value):
    """向列表中加入一个不重复的内容。
    如果这个内容已经存在，就不再重复加入。"""
    if value not in items:
        items.append(value)


def build_readiness(self_intersect, topology, mode="surface"):
    """根据自交和拓扑诊断结果生成 readiness 报告。
    主要通过几个关键指标判断网格能否继续 refine 或体网格化。"""

    if mode not in {"surface", "volume"}:
        mode = "surface"

    self_intersection_free = not bool(self_intersect)

    boundary_edges = _int_metric(topology, "boundary_edges")
    nonmanifold_edges = _int_metric(topology, "nonmanifold_edges")
    max_edge_incidence = _int_metric(topology, "max_edge_incidence")
    duplicate_faces = _int_metric(topology, "duplicate_faces")
    degenerate_faces = _int_metric(topology, "degenerate_faces")
    edge_component_count = _int_metric(topology, "edge_component_count")

    blocking_reasons = []
    warnings = [
        "Current MVP only checks mesh preprocessing readiness, not full solver input readiness."
    ]
    recommended_action = []

    if not self_intersection_free:
        _add_once(blocking_reasons, "self_intersections")
        _add_once(recommended_action, "run_autorefine")

    if nonmanifold_edges > 0 or max_edge_incidence > 2:
        _add_once(blocking_reasons, "nonmanifold_edges")
        _add_once(recommended_action, "run_manifold_repair_or_component_split")

    if degenerate_faces > 0:
        _add_once(blocking_reasons, "degenerate_faces")
        _add_once(recommended_action, "remove_degenerate_faces")

    if duplicate_faces > 0:
        _add_once(blocking_reasons, "duplicate_faces")
        _add_once(recommended_action, "remove_duplicate_faces")

    surface_manifold_ready = (
        self_intersection_free
        and nonmanifold_edges == 0
        and max_edge_incidence <= 2
        and degenerate_faces == 0
        and duplicate_faces == 0
    )

    if edge_component_count > 1:
        warnings.append(
            "Multiple edge-connected components detected; inspect assembly relations before merging."
        )
        _add_once(recommended_action, "inspect_component_relations")

    if boundary_edges > 0:
        _add_once(blocking_reasons, "boundary_edges")
        _add_once(recommended_action, "hole_filling_or_reject_for_volume_meshing")

    closed_surface_ready = surface_manifold_ready and boundary_edges == 0

    # 表面加密只需要排除明显的自交、非流形、重复面和退化面。
    gmsh_refine_ready = surface_manifold_ready

    # 体网格化要求表面已经闭合，但最终还需要 CGAL 再验证一次。
    volume_meshing_ready = closed_surface_ready
    if volume_meshing_ready:
        warnings.append(
            "volume_meshing_ready is preliminary: Python diagnostics cannot prove CGAL does_bound_a_volume."
        )
        _add_once(recommended_action, "eligible_for_cgal_volume_validation")

    industrial_solver_ready = False

    return {
        "self_intersection_free": self_intersection_free,
        "surface_manifold_ready": surface_manifold_ready,
        "closed_surface_ready": closed_surface_ready,
        "gmsh_refine_ready": gmsh_refine_ready,
        "volume_meshing_ready": volume_meshing_ready,
        "industrial_solver_ready": industrial_solver_ready,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "recommended_action": recommended_action,
    }