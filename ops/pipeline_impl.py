# -*- coding: utf-8 -*-
"""网格修复主流程：Python 清理 + CGAL 检测/Autorefine + 再清理 + 最终检测。

当前文件只负责单个网格的主流程编排：
1. 检查输入 V/F；
2. 执行 Python 侧拓扑清理；
3. 调用 CGAL 做自交检测和必要的 autorefine；
4. 再次清理并做最终 CGAL 检测；
5. 输出修复后的 V/F 和可写入 JSON 的 report。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np

from mesh.diagnostics import topology_summary
from mesh.io_obj import load_obj, save_obj
from mesh.mesh import Mesh
from ops.cgal_refine import CheckResult, autorefine_obj, check_self_intersections
from ops.readiness import build_readiness
from ops.stitch import (
    CleanupReport,
    area_threshold_from_mesh,
    cleanup_topology,
    mesh_has_degenerate_faces,
    mesh_has_duplicate_faces,
)


MeshDict = dict[str, np.ndarray]
PathLike = str | Path


@dataclass
class RepairRunReport:
    """记录单个网格从清理、CGAL 修复到最终检查的主流程结果。"""

    eps_mode: str
    eps_v_input: float
    eps_v_abs: float

    pre_cleanup: CleanupReport
    pre_check: CheckResult | None
    autorefined: bool
    post_cleanup: CleanupReport
    post_check: CheckResult

    final_vertices: int
    final_faces: int

    diagnostics: dict[str, object] = field(default_factory=dict)
    readiness: dict[str, object] = field(default_factory=dict)
    status: str = "unknown"

    def as_dict(self) -> dict[str, object]:
        """转成普通 dict，供 CLI 写入 JSON 报告。"""
        return {
            "eps_mode": self.eps_mode,
            "eps_v_input": self.eps_v_input,
            "eps_v_abs": self.eps_v_abs,
            "pre_cleanup": self.pre_cleanup.as_dict(),
            "pre_check": _check_result_to_dict(self.pre_check),
            "autorefined": self.autorefined,
            "post_cleanup": self.post_cleanup.as_dict(),
            "post_check": _check_result_to_dict(self.post_check),
            "final_vertices": self.final_vertices,
            "final_faces": self.final_faces,
            "diagnostics": self.diagnostics,
            "readiness": self.readiness,
            "status": self.status,
        }


def _check_result_to_dict(result: CheckResult | None) -> dict[str, object] | None:
    """把 CGAL 检查结果转成 JSON 友好的 dict。"""
    if result is None:
        return None

    return {
        "self_intersect": bool(result.self_intersect),
        "count": int(result.count),
        "pairs": [[int(a), int(b)] for a, b in result.pairs],
    }


def _require_mesh_dict(mesh_in: MeshDict) -> MeshDict:
    """检查输入网格必须包含 V/F，并统一成 numpy 数组。"""
    if "V" not in mesh_in or "F" not in mesh_in:
        raise KeyError("mesh_in 必须包含 'V' 和 'F' 两个字段")

    return {
        "V": np.asarray(mesh_in["V"], dtype=np.float64),
        "F": np.asarray(mesh_in["F"], dtype=np.int64),
    }


def bbox_diag(V: np.ndarray) -> float:
    """计算顶点包围盒对角线长度。"""
    V = np.asarray(V, dtype=np.float64)

    if V.size == 0:
        return 0.0

    lo = V.min(axis=0)
    hi = V.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def normalize_eps(eps: float, V: np.ndarray, mode: str = "relative_bbox") -> float:
    """把 eps_v 按 absolute/relative_bbox 模式转成绝对长度。"""
    eps = float(eps)

    if eps < 0:
        raise ValueError(f"eps_v 不能为负数，当前为 {eps}")

    if mode == "absolute":
        return eps

    if mode == "relative_bbox":
        return float(eps * bbox_diag(V))

    raise ValueError(f"未知 eps_mode={mode!r}，只能是 'absolute' 或 'relative_bbox'")


def _validate_post_cleanup_mesh(mesh: Mesh) -> None:
    """检查清理后的网格没有退化面、重复面，并且面索引已压缩。"""
    if mesh_has_degenerate_faces(mesh):
        raise RuntimeError("清理后仍然存在退化三角面")

    if mesh_has_duplicate_faces(mesh):
        raise RuntimeError("清理后仍然存在重复三角面")

    if mesh.F.size == 0:
        if mesh.V.size:
            raise RuntimeError("清理后只有孤立顶点、没有面，期望得到空网格")
        return

    min_idx = int(mesh.F.min())
    max_idx = int(mesh.F.max())

    if min_idx != 0:
        raise RuntimeError(f"面索引没有从 0 开始压缩，当前最小索引为 {min_idx}")

    if max_idx != mesh.V.shape[0] - 1:
        raise RuntimeError(
            f"面索引没有完全压缩: 最大面索引={max_idx}, 顶点数={mesh.V.shape[0]}"
        )


def _mesh_topology(mesh: Mesh) -> dict[str, object]:
    """对当前 Mesh 计算 topology diagnostics。"""
    return topology_summary(mesh.V, mesh.F)


def _build_diagnostics(
    *,
    input_topology: dict[str, object],
    pre_topology: dict[str, object],
    autorefine_topology: dict[str, object],
    post_topology: dict[str, object],
) -> dict[str, object]:
    """统一组织 report 中的 diagnostics 字段。"""
    return {
        "input": input_topology,
        "after_pre_cleanup": pre_topology,
        "after_autorefine": autorefine_topology,
        "after_post_cleanup": post_topology,
    }


def _status_from_readiness(
    *,
    self_intersect: bool,
    readiness: dict[str, object],
) -> str:
    """根据最终自交检测和 readiness 给出主流程状态。"""
    if self_intersect:
        return "self_intersection_remaining"

    if readiness.get("surface_manifold_ready") is True:
        return "surface_manifold_ready"

    return "self_intersection_free_but_not_manifold_ready"


def python_cleanup_only(
    mesh_in: MeshDict,
    *,
    eps_v: float = 1e-9,
    eps_mode: str = "relative_bbox",
) -> tuple[MeshDict, CleanupReport, float]:
    """只执行 Python 侧拓扑清理，不调用 CGAL。"""
    checked = _require_mesh_dict(mesh_in)
    mesh = Mesh(checked["V"].copy(), checked["F"].copy())

    eps_v_abs = normalize_eps(eps_v, mesh.V, mode=eps_mode)
    area_eps = area_threshold_from_mesh(mesh)

    report = cleanup_topology(mesh, eps_v=eps_v_abs, area_eps=area_eps)
    _validate_post_cleanup_mesh(mesh)

    return {"V": mesh.V.copy(), "F": mesh.F.copy()}, report, eps_v_abs


def _make_work_dir(work_dir: PathLike | None) -> tuple[Path, TemporaryDirectory[str] | None]:
    """准备工作目录；外部传入 work_dir 时保留中间 OBJ。"""
    if work_dir is not None:
        path = Path(work_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path, None

    tmp_mgr = TemporaryDirectory(prefix="assembly_mesh_repair_")
    return Path(tmp_mgr.name), tmp_mgr


def _save_stage_mesh(path: Path, mesh: Mesh) -> None:
    """保存当前阶段的 Mesh，供 CGAL 可执行程序读取。"""
    save_obj(path, mesh.V, mesh.F)


def _run_precheck_and_maybe_autorefine(
    *,
    mesh: Mesh,
    pre_cleanup_obj: Path,
    work_dir: Path,
    build_dir: PathLike,
    checker_timeout: int,
    refine_timeout: int,
    snap_grid_size: int,
    number_of_iterations: int,
    run_precheck: bool,
) -> tuple[Mesh, CheckResult | None, bool, dict[str, object]]:
    """执行 pre-check；若存在自交，则调用 CGAL autorefine 并读回修复结果。"""
    if not run_precheck:
        print("[PIPELINE] pre-check skipped by configuration")
        return mesh, None, False, _mesh_topology(mesh)

    pre_check = check_self_intersections(
        pre_cleanup_obj,
        build_dir=build_dir,
        timeout=checker_timeout,
    )
    print(
        f"[PIPELINE] pre-check: "
        f"self_intersect={int(pre_check.self_intersect)} count={pre_check.count}"
    )

    if not pre_check.self_intersect:
        return mesh, pre_check, False, _mesh_topology(mesh)

    refined_obj = work_dir / "refined.obj"
    autorefine_obj(
        pre_cleanup_obj,
        refined_obj,
        build_dir=build_dir,
        timeout=refine_timeout,
        snap_grid_size=snap_grid_size,
        number_of_iterations=number_of_iterations,
    )

    refined_V, refined_F = load_obj(refined_obj)
    refined_mesh = Mesh(refined_V, refined_F)

    print(
        f"[PIPELINE] CGAL autorefine produced "
        f"V={refined_mesh.num_vertices}, F={refined_mesh.num_faces}"
    )

    return refined_mesh, pre_check, True, _mesh_topology(refined_mesh)


def _post_cleanup_and_check(
    *,
    mesh: Mesh,
    eps_v_abs: float,
    work_dir: Path,
    build_dir: PathLike,
    checker_timeout: int,
) -> tuple[CleanupReport, CheckResult, dict[str, object]]:
    """执行后处理清理、保存 post OBJ，并做最终 CGAL 自交检测。"""
    post_cleanup = cleanup_topology(
        mesh,
        eps_v=eps_v_abs,
        area_eps=area_threshold_from_mesh(mesh),
    )
    _validate_post_cleanup_mesh(mesh)

    post_cleanup_obj = work_dir / "post_cleanup.obj"
    _save_stage_mesh(post_cleanup_obj, mesh)

    print(
        f"[PIPELINE] post-cleanup: "
        f"V {post_cleanup.V_before} -> {post_cleanup.V_after}, "
        f"F {post_cleanup.F_before} -> {post_cleanup.F_after}"
    )

    post_check = check_self_intersections(
        post_cleanup_obj,
        build_dir=build_dir,
        timeout=checker_timeout,
    )
    print(
        f"[PIPELINE] post-check: "
        f"self_intersect={int(post_check.self_intersect)} count={post_check.count}"
    )

    post_topology = _mesh_topology(mesh)
    return post_cleanup, post_check, post_topology


def repair_single_mesh(
    mesh_in: MeshDict,
    *,
    eps_v: float = 1e-9,
    eps_mode: str = "relative_bbox",
    build_dir: PathLike = "build/cgal",
    checker_timeout: int = 60,
    refine_timeout: int = 300,
    snap_grid_size: int = 23,
    number_of_iterations: int = 5,
    run_precheck: bool = True,
    run_postcheck: bool = True,
    work_dir: PathLike | None = None,
) -> tuple[MeshDict, RepairRunReport]:
    """修复单个 OBJ 网格，并返回修复后的 V/F 与主流程 report。"""
    if not run_postcheck:
        raise RuntimeError("生产路径必须开启 run_postcheck，不能跳过最终 CGAL 检测")

    input_mesh = _require_mesh_dict(mesh_in)
    input_topology = topology_summary(input_mesh["V"], input_mesh["F"])

    pre_mesh_dict, pre_cleanup, eps_v_abs = python_cleanup_only(
        input_mesh,
        eps_v=eps_v,
        eps_mode=eps_mode,
    )
    current_mesh = Mesh(pre_mesh_dict["V"], pre_mesh_dict["F"])
    pre_topology = _mesh_topology(current_mesh)

    work_dir_path, tmp_mgr = _make_work_dir(work_dir)

    try:
        pre_cleanup_obj = work_dir_path / "pre_cleanup.obj"
        _save_stage_mesh(pre_cleanup_obj, current_mesh)

        print(
            f"[PIPELINE] pre-cleanup: "
            f"V {pre_cleanup.V_before} -> {pre_cleanup.V_after}, "
            f"F {pre_cleanup.F_before} -> {pre_cleanup.F_after}, "
            f"eps_v_abs={eps_v_abs:.17g}"
        )

        (
            current_mesh,
            pre_check,
            autorefined,
            autorefine_topology,
        ) = _run_precheck_and_maybe_autorefine(
            mesh=current_mesh,
            pre_cleanup_obj=pre_cleanup_obj,
            work_dir=work_dir_path,
            build_dir=build_dir,
            checker_timeout=checker_timeout,
            refine_timeout=refine_timeout,
            snap_grid_size=snap_grid_size,
            number_of_iterations=number_of_iterations,
            run_precheck=run_precheck,
        )

        post_cleanup, post_check, post_topology = _post_cleanup_and_check(
            mesh=current_mesh,
            eps_v_abs=eps_v_abs,
            work_dir=work_dir_path,
            build_dir=build_dir,
            checker_timeout=checker_timeout,
        )

        if post_check.self_intersect:
            raise RuntimeError(
                f"最终 CGAL 检测失败，仍有 {post_check.count} 对自交三角面"
            )

        diagnostics = _build_diagnostics(
            input_topology=input_topology,
            pre_topology=pre_topology,
            autorefine_topology=autorefine_topology,
            post_topology=post_topology,
        )
        readiness = build_readiness(
            self_intersect=post_check.self_intersect,
            topology=post_topology,
        )
        status = _status_from_readiness(
            self_intersect=post_check.self_intersect,
            readiness=readiness,
        )

        report = RepairRunReport(
            eps_mode=eps_mode,
            eps_v_input=float(eps_v),
            eps_v_abs=float(eps_v_abs),
            pre_cleanup=pre_cleanup,
            pre_check=pre_check,
            autorefined=autorefined,
            post_cleanup=post_cleanup,
            post_check=post_check,
            final_vertices=current_mesh.num_vertices,
            final_faces=current_mesh.num_faces,
            diagnostics=diagnostics,
            readiness=readiness,
            status=status,
        )

        return {"V": current_mesh.V.copy(), "F": current_mesh.F.copy()}, report

    finally:
        if tmp_mgr is not None:
            tmp_mgr.cleanup()


def repair_assembly_mesh(
    meshes_in: Sequence[MeshDict],
    *,
    eps_v: float = 1e-9,
    eps_mode: str = "relative_bbox",
    build_dir: PathLike = "build/cgal",
    checker_timeout: int = 60,
    refine_timeout: int = 300,
    snap_grid_size: int = 23,
    number_of_iterations: int = 5,
    run_precheck: bool = True,
    run_postcheck: bool = True,
) -> list[MeshDict]:
    """逐个修复多个 OBJ 网格；当前版本不处理跨文件自交。"""
    repaired: list[MeshDict] = []

    for idx, mesh in enumerate(meshes_in):
        print(f"[PIPELINE] processing mesh #{idx}")

        out_mesh, _report = repair_single_mesh(
            mesh,
            eps_v=eps_v,
            eps_mode=eps_mode,
            build_dir=build_dir,
            checker_timeout=checker_timeout,
            refine_timeout=refine_timeout,
            snap_grid_size=snap_grid_size,
            number_of_iterations=number_of_iterations,
            run_precheck=run_precheck,
            run_postcheck=run_postcheck,
        )
        repaired.append(out_mesh)

    return repaired
