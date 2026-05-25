"""测试 CLI 层 Gmsh refine readiness 门禁。

这些测试只验证 pipeline.py 里的调度逻辑：
1. readiness 通过时允许 Gmsh refine；
2. readiness 不通过时默认跳过；
3. 用户显式 force 时允许执行，但必须写 warning。

真实 CGAL、Gmsh、OBJ 文件读写都用 monkeypatch 隔离。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import pipeline
from pipeline import FORCED_GMSH_REFINE_WARNING, should_run_gmsh_refine


class FakeRepairReport:
    """构造带 readiness 的最小 repair report，避免测试依赖 CGAL/Gmsh。"""

    def __init__(self, gmsh_ready: bool):
        """记录当前假报告是否允许进入 Gmsh refine。"""
        self.gmsh_ready = gmsh_ready

    def as_dict(self) -> dict[str, object]:
        """返回 repair_one_obj 需要读取的最小报告字段。"""
        status = (
            "surface_manifold_ready"
            if self.gmsh_ready
            else "self_intersection_free_but_not_manifold_ready"
        )
        return {
            "readiness": {
                "gmsh_refine_ready": self.gmsh_ready,
                "surface_manifold_ready": self.gmsh_ready,
            },
            "status": status,
            "diagnostics": {},
            "assembly_diagnostics": {},
        }


def _mesh() -> dict[str, np.ndarray]:
    """返回一个最小三角面网格。"""
    return {
        "V": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "F": np.array([[0, 1, 2]], dtype=np.int64),
    }


def _args(*, gmsh_refine: bool = True, force: bool = False) -> argparse.Namespace:
    """构造 repair_one_obj 需要的最小 CLI 参数集合。"""
    return argparse.Namespace(
        eps_v=1e-9,
        eps_mode="relative_bbox",
        build_dir="build/cgal",
        checker_timeout=60,
        refine_timeout=300,
        snap_grid_size=23,
        number_of_iterations=5,
        gmsh_refine=gmsh_refine,
        force_gmsh_refine=force,
        gmsh_refine_levels=1,
        gmsh_target_edge_length=None,
        gmsh_target_edge_ratio=None,
        gmsh_max_refine_levels=5,
        gmsh_keep_msh=False,
        gmsh_terminal=0,
    )


def _patch_common(monkeypatch, *, gmsh_ready: bool) -> dict[str, np.ndarray]:
    """替换真实 IO 和 repair 调用，只保留 CLI 门禁逻辑。"""
    mesh = _mesh()

    monkeypatch.setattr(pipeline, "load_obj", lambda _path: (mesh["V"], mesh["F"]))
    monkeypatch.setattr(pipeline, "save_obj", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "repair_single_mesh",
        lambda *_args, **_kwargs: (mesh, FakeRepairReport(gmsh_ready)),
    )

    return mesh


def _patch_fake_refine(monkeypatch) -> dict[str, int]:
    """替换 Gmsh refine，并记录是否真的进入 refine 分支。"""
    calls = {"refine": 0}

    def fake_refine(repaired_mesh, **_kwargs):
        """模拟 refine 成功，只返回最小报告字段。"""
        calls["refine"] += 1
        return repaired_mesh, {
            "是否开启": True,
            "加密后顶点数": int(len(repaired_mesh["V"])),
            "加密后三角形数": int(len(repaired_mesh["F"])),
        }

    monkeypatch.setattr(pipeline, "refine_repaired_mesh_with_gmsh", fake_refine)
    return calls


def test_gmsh_gate_does_not_run_when_not_requested():
    """用户没有开启 --gmsh_refine 时，不应执行也不应跳过。"""
    should_run, skipped, reason = should_run_gmsh_refine(
        gmsh_requested=False,
        gmsh_ready=True,
        force_gmsh_refine=False,
    )

    assert should_run is False
    assert skipped is False
    assert reason is None


def test_gmsh_gate_runs_when_ready():
    """readiness 通过时，应允许执行 Gmsh refine。"""
    should_run, skipped, reason = should_run_gmsh_refine(
        gmsh_requested=True,
        gmsh_ready=True,
        force_gmsh_refine=False,
    )

    assert should_run is True
    assert skipped is False
    assert reason is None


def test_gmsh_gate_skips_when_not_ready_without_force():
    """readiness 不通过且未 force 时，应跳过 Gmsh refine。"""
    should_run, skipped, reason = should_run_gmsh_refine(
        gmsh_requested=True,
        gmsh_ready=False,
        force_gmsh_refine=False,
    )

    assert should_run is False
    assert skipped is True
    assert reason is not None
    assert "force_gmsh_refine" in reason


def test_gmsh_gate_runs_when_forced():
    """readiness 不通过但用户显式 force 时，应允许执行。"""
    should_run, skipped, reason = should_run_gmsh_refine(
        gmsh_requested=True,
        gmsh_ready=False,
        force_gmsh_refine=True,
    )

    assert should_run is True
    assert skipped is False
    assert reason is None


def test_repair_one_obj_does_not_run_gmsh_when_not_requested(monkeypatch, tmp_path):
    """未开启 gmsh_refine 时，repair_one_obj 不应调用 refine 分支。"""
    _patch_common(monkeypatch, gmsh_ready=True)
    calls = _patch_fake_refine(monkeypatch)

    report = pipeline.repair_one_obj(Path("plain.obj"), tmp_path, _args(gmsh_refine=False))
    gmsh_report = report["Gmsh加密"]

    assert calls["refine"] == 0
    assert gmsh_report["是否开启"] is False
    assert gmsh_report["是否执行"] is False
    assert gmsh_report["是否跳过"] is False
    assert gmsh_report["gmsh_refine_skipped"] is False


def test_repair_one_obj_runs_gmsh_when_ready(monkeypatch, tmp_path):
    """repair_one_obj 应在 gmsh_refine_ready=True 时调用 refine 分支。"""
    _patch_common(monkeypatch, gmsh_ready=True)
    calls = _patch_fake_refine(monkeypatch)

    report = pipeline.repair_one_obj(Path("ready.obj"), tmp_path, _args())
    gmsh_report = report["Gmsh加密"]

    assert calls["refine"] == 1
    assert gmsh_report["是否执行"] is True
    assert gmsh_report["是否跳过"] is False
    assert gmsh_report["gmsh_refine_skipped"] is False


def test_repair_one_obj_skips_gmsh_when_not_ready_without_force(monkeypatch, tmp_path):
    """readiness 不通过且未 force 时，repair_one_obj 不应调用 refine 分支。"""
    _patch_common(monkeypatch, gmsh_ready=False)
    calls = _patch_fake_refine(monkeypatch)

    report = pipeline.repair_one_obj(Path("not_ready.obj"), tmp_path, _args(force=False))
    gmsh_report = report["Gmsh加密"]

    assert calls["refine"] == 0
    assert gmsh_report["是否开启"] is True
    assert gmsh_report["是否执行"] is False
    assert gmsh_report["是否跳过"] is True
    assert gmsh_report["gmsh_refine_skipped"] is True
    assert "Use --force_gmsh_refine to override." in gmsh_report["gmsh_refine_skip_reason"]


def test_repair_one_obj_runs_gmsh_when_not_ready_but_forced(monkeypatch, tmp_path):
    """force 时允许执行，但 report 必须写入 warning。"""
    _patch_common(monkeypatch, gmsh_ready=False)
    calls = _patch_fake_refine(monkeypatch)

    report = pipeline.repair_one_obj(Path("forced.obj"), tmp_path, _args(force=True))
    gmsh_report = report["Gmsh加密"]

    assert calls["refine"] == 1
    assert gmsh_report["是否执行"] is True
    assert gmsh_report["是否跳过"] is False
    assert FORCED_GMSH_REFINE_WARNING in gmsh_report["warnings"]
