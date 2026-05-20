import json

from ops.readiness import build_readiness


def clean_closed_topology():
    """构造一个干净、闭合的表面网格诊断结果。
    这些指标表示没有边界边、非流形边、重复面和退化面。"""
    return {
        "boundary_edges": 0,
        "nonmanifold_edges": 0,
        "max_edge_incidence": 2,
        "duplicate_faces": 0,
        "degenerate_faces": 0,
        "edge_component_count": 1,
    }


def test_clean_closed_mesh_is_ready():
    """测试干净闭合网格是否能通过 readiness 检查。
    使用一个无自交、无非流形、无边界边的 topology 作为输入。"""
    topology = clean_closed_topology()

    readiness = build_readiness(self_intersect=False, topology=topology)

    assert readiness["self_intersection_free"] is True
    assert readiness["surface_manifold_ready"] is True
    assert readiness["closed_surface_ready"] is True
    assert readiness["gmsh_refine_ready"] is True
    assert readiness["volume_meshing_ready"] is True
    assert readiness["industrial_solver_ready"] is False
    assert readiness["blocking_reasons"] == []
    assert "eligible_for_cgal_volume_validation" in readiness["recommended_action"]


def test_self_intersection_blocks_readiness():
    """测试自交是否会阻止网格继续进入下一步。
    将 self_intersect 设为 True，检查报告中是否给出对应阻塞原因。"""
    topology = clean_closed_topology()

    readiness = build_readiness(self_intersect=True, topology=topology)

    assert readiness["self_intersection_free"] is False
    assert readiness["surface_manifold_ready"] is False
    assert readiness["closed_surface_ready"] is False
    assert readiness["gmsh_refine_ready"] is False
    assert readiness["volume_meshing_ready"] is False
    assert "self_intersections" in readiness["blocking_reasons"]
    assert "run_autorefine" in readiness["recommended_action"]


def test_nonmanifold_edges_block_readiness():
    """测试非流形边是否会阻止表面网格通过检查。
    手动增加 nonmanifold_edges 和 max_edge_incidence 来模拟非流形情况。"""
    topology = clean_closed_topology()
    topology["nonmanifold_edges"] = 3
    topology["max_edge_incidence"] = 4

    readiness = build_readiness(self_intersect=False, topology=topology)

    assert readiness["self_intersection_free"] is True
    assert readiness["surface_manifold_ready"] is False
    assert readiness["closed_surface_ready"] is False
    assert readiness["gmsh_refine_ready"] is False
    assert readiness["volume_meshing_ready"] is False
    assert "nonmanifold_edges" in readiness["blocking_reasons"]
    assert "run_manifold_repair_or_component_split" in readiness["recommended_action"]


def test_boundary_edges_only_block_closed_surface_and_volume():
    """测试边界边是否只影响闭合表面和体网格化判断。
    表面本身可以继续 refine，但不能直接认为适合体网格化。"""
    topology = clean_closed_topology()
    topology["boundary_edges"] = 6

    readiness = build_readiness(self_intersect=False, topology=topology)

    assert readiness["self_intersection_free"] is True
    assert readiness["surface_manifold_ready"] is True
    assert readiness["closed_surface_ready"] is False
    assert readiness["gmsh_refine_ready"] is True
    assert readiness["volume_meshing_ready"] is False
    assert "boundary_edges" in readiness["blocking_reasons"]
    assert "hole_filling_or_reject_for_volume_meshing" in readiness["recommended_action"]


def test_duplicate_faces_block_surface_readiness():
    """测试重复面是否会阻止表面 readiness 通过。
    将 duplicate_faces 设为正数，检查阻塞原因和推荐动作。"""
    topology = clean_closed_topology()
    topology["duplicate_faces"] = 2

    readiness = build_readiness(self_intersect=False, topology=topology)

    assert readiness["surface_manifold_ready"] is False
    assert readiness["closed_surface_ready"] is False
    assert readiness["gmsh_refine_ready"] is False
    assert "duplicate_faces" in readiness["blocking_reasons"]
    assert "remove_duplicate_faces" in readiness["recommended_action"]


def test_degenerate_faces_block_surface_readiness():
    """测试退化面是否会阻止表面 readiness 通过。
    将 degenerate_faces 设为正数，检查阻塞原因和推荐动作。"""
    topology = clean_closed_topology()
    topology["degenerate_faces"] = 1

    readiness = build_readiness(self_intersect=False, topology=topology)

    assert readiness["surface_manifold_ready"] is False
    assert readiness["closed_surface_ready"] is False
    assert readiness["gmsh_refine_ready"] is False
    assert "degenerate_faces" in readiness["blocking_reasons"]
    assert "remove_degenerate_faces" in readiness["recommended_action"]


def test_multiple_components_add_warning():
    """测试多个 edge-connected components 是否会给出提醒。
    多 component 不一定直接阻塞，但需要检查装配体之间的关系。"""
    topology = clean_closed_topology()
    topology["edge_component_count"] = 3

    readiness = build_readiness(self_intersect=False, topology=topology)

    assert readiness["surface_manifold_ready"] is True
    assert readiness["closed_surface_ready"] is True
    assert "inspect_component_relations" in readiness["recommended_action"]

    has_component_warning = any(
        "Multiple edge-connected components" in warning
        for warning in readiness["warnings"]
    )
    assert has_component_warning is True


def test_readiness_result_can_be_saved_as_json():
    """测试 readiness 报告是否可以转成 JSON。
    这样后面 CLI 或日志系统就能直接保存这个结果"""
    topology = clean_closed_topology()

    readiness = build_readiness(self_intersect=False, topology=topology)

    json.dumps(readiness, ensure_ascii=False)


def test_bad_mode_uses_surface_mode():
    """测试非法 mode 输入时程序是否还能继续运行。
    """
    topology = clean_closed_topology()

    readiness = build_readiness(
        self_intersect=False,
        topology=topology,
        mode="bad_mode",
    )

    assert readiness["surface_manifold_ready"] is True
    assert readiness["gmsh_refine_ready"] is True


def test_bad_topology_value_uses_default_value():
    """测试 topology 中出现异常字段值时程序是否还能继续运行。
    简化版代码会使用默认值，避免因为一个字段写错就中断整个测试。"""
    topology = clean_closed_topology()
    topology["boundary_edges"] = "not_an_int"

    readiness = build_readiness(self_intersect=False, topology=topology)

    assert readiness["surface_manifold_ready"] is True
    assert readiness["closed_surface_ready"] is True
    assert readiness["volume_meshing_ready"] is True