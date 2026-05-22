# -*- coding: utf-8 -*-
"""测试 RepairRunReport 已接入 diagnostics/readiness/status 字段。"""

from __future__ import annotations

import json

import numpy as np

from mesh.diagnostics import topology_summary
from ops.cgal_refine import CheckResult, CommandResult
from ops.pipeline_impl import RepairRunReport
from ops.readiness import build_readiness
from ops.stitch import CleanupReport


def _nonmanifold_edge_mesh() -> tuple[np.ndarray, np.ndarray]:
    """构造一个最小非流形网格：3 个三角面共享同一条边 0-1。"""
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4],
        ],
        dtype=np.int64,
    )
    return V, F


def _fake_cleanup_report() -> CleanupReport:
    """构造最小 CleanupReport，用来填充 RepairRunReport。"""
    return CleanupReport(
        V_before=5,
        V_after=5,
        F_before=3,
        F_after=3,
    )


def _fake_check_result(self_intersect: bool = False) -> CheckResult:
    """构造最小 CGAL 检查结果，避免测试里真的调用 CGAL。"""
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


def _fake_report() -> RepairRunReport:
    """构造带 diagnostics/readiness/status 的主流程报告。"""
    V, F = _nonmanifold_edge_mesh()

    topology = topology_summary(V, F)
    readiness = build_readiness(
        self_intersect=False,
        topology=topology,
    )

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
        pre_cleanup=_fake_cleanup_report(),
        pre_check=None,
        autorefined=False,
        post_cleanup=_fake_cleanup_report(),
        post_check=_fake_check_result(False),
        final_vertices=5,
        final_faces=3,
        diagnostics=diagnostics,
        readiness=readiness,
        status="self_intersection_free_but_not_manifold_ready",
    )


def test_report_exports_diagnostics_readiness_and_status() -> None:
    """检查主流程 report 的新增字段能正确导出为 dict/JSON。"""
    report_dict = _fake_report().as_dict()

    assert report_dict["diagnostics"]["after_post_cleanup"]["nonmanifold_edges"] == 1
    assert report_dict["readiness"]["surface_manifold_ready"] is False
    assert report_dict["status"] == "self_intersection_free_but_not_manifold_ready"

    json.dumps(report_dict, ensure_ascii=False)