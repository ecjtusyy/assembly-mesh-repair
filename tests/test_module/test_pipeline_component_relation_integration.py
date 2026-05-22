"""测试 RepairRunReport 已接入 assembly component 关系诊断。"""

from __future__ import annotations

import json

import numpy as np

from mesh.diagnostics import topology_summary
from ops.cgal_refine import CheckResult, CommandResult
from ops.component_relation import classify_component_relations
from ops.pipeline_impl import RepairRunReport, _build_assembly_diagnostics
from ops.readiness import build_readiness
from ops.stitch import CleanupReport


MeshData = tuple[np.ndarray, np.ndarray]


def _two_separate_triangles() -> MeshData:
    """构造两个空间分离的三角形，用来模拟两个独立 component。"""
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.int64,
    )
    return V, F


def _single_triangle() -> MeshData:
    """构造单个三角形，用来测试单 component fallback。"""
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array([[0, 1, 2]], dtype=np.int64)
    return V, F


def _fake_cleanup_report(V_count: int, F_count: int) -> CleanupReport:
    """构造最小 CleanupReport，只用于填充 RepairRunReport。"""
    return CleanupReport(
        V_before=V_count,
        V_after=V_count,
        F_before=F_count,
        F_after=F_count,
    )


def _fake_check_result(self_intersect: bool = False) -> CheckResult:
    """构造最小 CGAL 检查结果，避免测试里真实调用 CGAL。"""
    command = CommandResult(
        cmd=["check_self_intersections", "dummy.obj"],
        returncode=0,
        stdout="self_intersect=0\ncount=0\n",
        stderr="",
    )
    return CheckResult(
        self_intersect=self_intersect,
        count=0,
        pairs=[],
        command=command,
    )


def _fake_report(
    V: np.ndarray,
    F: np.ndarray,
    assembly_diagnostics: dict[str, object],
) -> RepairRunReport:
    """用真实 RepairRunReport 组装一个最小可序列化 report。"""
    topology = topology_summary(V, F)

    diagnostics = {
        "input": topology,
        "after_pre_cleanup": topology,
        "after_autorefine": topology,
        "after_post_cleanup": topology,
    }

    return RepairRunReport(
        eps_mode="absolute",
        eps_v_input=1e-9,
        eps_v_abs=1e-9,
        pre_cleanup=_fake_cleanup_report(len(V), len(F)),
        pre_check=None,
        autorefined=False,
        post_cleanup=_fake_cleanup_report(len(V), len(F)),
        post_check=_fake_check_result(False),
        final_vertices=int(len(V)),
        final_faces=int(len(F)),
        diagnostics=diagnostics,
        readiness=build_readiness(self_intersect=False, topology=topology),
        status="self_intersection_free_but_not_manifold_ready",
        assembly_diagnostics=assembly_diagnostics,
    )


def test_report_contains_component_relation_for_two_components() -> None:
    """两个分离三角形应作为两个 component 写入 report。"""
    V, F = _two_separate_triangles()

    topology = topology_summary(V, F)
    assembly_diagnostics = classify_component_relations(
        V,
        F,
        topology["face_component_id"],
    )
    report_dict = _fake_report(V, F, assembly_diagnostics).as_dict()

    assert report_dict["assembly_diagnostics"]["component_count"] == 2
    assert report_dict["assembly_diagnostics"]["bbox_overlap_pairs"] == []
    assert report_dict["assembly_diagnostics"]["suspected_multi_body_assembly"] is True

    json.dumps(report_dict, ensure_ascii=False)


def test_single_component_uses_pipeline_fallback() -> None:
    """单 component 不需要 bbox 分类，直接返回 pipeline fallback 结构。"""
    V, F = _single_triangle()

    topology = topology_summary(V, F)
    assembly_diagnostics = _build_assembly_diagnostics(V, F, topology)
    report_dict = _fake_report(V, F, assembly_diagnostics).as_dict()

    assert report_dict["assembly_diagnostics"] == {
        "component_count": 1,
        "bbox_overlap_pairs": [],
        "suspected_multi_body_assembly": False,
    }


def test_topology_summary_exports_json_safe_face_component_id() -> None:
    """topology_summary 必须导出 JSON 安全的 face_component_id。"""
    V, F = _two_separate_triangles()

    topology = topology_summary(V, F)
    face_component_id = topology["face_component_id"]

    assert len(face_component_id) == len(F)
    assert all(isinstance(x, int) for x in face_component_id)

    json.dumps(face_component_id, ensure_ascii=False)