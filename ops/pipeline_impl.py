# -*- coding: utf-8 -*-
"""网格修复主流程：Python 清理 + CGAL 检测/Autorefine + 再清理 + 最终检测。

这是之前调试的demo
- Python 侧近似三角形相交检测
- Python 侧边拆分
- T-junction 传播
- 平滑修复
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence

import numpy as np

from mesh.io_obj import load_obj, save_obj
from mesh.mesh import Mesh
from ops.cgal_refine import CheckResult, autorefine_obj, check_self_intersections
from ops.stitch import (
    CleanupReport,
    area_threshold_from_mesh,
    cleanup_topology,
    mesh_has_degenerate_faces,
    mesh_has_duplicate_faces,
)


MeshDict = Dict[str, np.ndarray]
PathLike = str | Path


@dataclass
class RepairRunReport:
    eps_mode: str
    eps_v_input: float
    eps_v_abs: float
    pre_cleanup: CleanupReport
    pre_check: Optional[CheckResult]
    autorefined: bool
    post_cleanup: CleanupReport
    post_check: CheckResult
    final_vertices: int
    final_faces: int

    def as_dict(self) -> dict:
        """转换成方便写入 JSON 的普通字典。"""
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
        }


def _check_result_to_dict(result: Optional[CheckResult]) -> Optional[dict]:
    if result is None:
        return None

    return {
        "self_intersect": result.self_intersect,
        "count": result.count,
        "pairs": [[int(a), int(b)] for a, b in result.pairs],
    }


def _require_mesh_dict(mesh_in: MeshDict) -> MeshDict:
    """检查主流程需要的 V/F 字段是否存在。"""
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
    """把 eps_v 转成绝对尺度。"""
    eps = float(eps)

    if eps < 0:
        raise ValueError(f"eps_v 不能为负数，当前为 {eps}")

    if mode == "absolute":
        return eps

    if mode == "relative_bbox":
        return float(eps * bbox_diag(V))

    raise ValueError(f"未知 eps_mode={mode!r}，只能是 'absolute' 或 'relative_bbox'")


def _validate_post_cleanup_mesh(mesh: Mesh) -> None:
    """清理后必须满足的最小拓扑约束。"""
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
    """创建临时工作目录；外部传入 work_dir 时保留中间文件。"""
    if work_dir is not None:
        path = Path(work_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path, None

    tmp_mgr = TemporaryDirectory(prefix="assembly_mesh_repair_")
    return Path(tmp_mgr.name), tmp_mgr


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
    """修复单个 OBJ 网格。"""
    if not run_postcheck:
        raise RuntimeError("生产路径必须开启 run_postcheck，不能跳过最终 CGAL 检测")

    pre_mesh_dict, pre_cleanup, eps_v_abs = python_cleanup_only(
        mesh_in,
        eps_v=eps_v,
        eps_mode=eps_mode,
    )
    current_mesh = Mesh(pre_mesh_dict["V"], pre_mesh_dict["F"])

    work_dir_path, tmp_mgr = _make_work_dir(work_dir)

    try:
        pre_cleanup_obj = work_dir_path / "pre_cleanup.obj"
        save_obj(pre_cleanup_obj, current_mesh.V, current_mesh.F)

        print(
            f"[PIPELINE] pre-cleanup: "
            f"V {pre_cleanup.V_before} -> {pre_cleanup.V_after}, "
            f"F {pre_cleanup.F_before} -> {pre_cleanup.F_after}, "
            f"eps_v_abs={eps_v_abs:.17g}"
        )

        pre_check: Optional[CheckResult] = None
        autorefined = False

        if run_precheck:
            pre_check = check_self_intersections(
                pre_cleanup_obj,
                build_dir=build_dir,
                timeout=checker_timeout,
            )
            print(
                f"[PIPELINE] pre-check: "
                f"self_intersect={int(pre_check.self_intersect)} count={pre_check.count}"
            )

            if pre_check.self_intersect:
                refined_obj = work_dir_path / "refined.obj"
                autorefine_obj(
                    pre_cleanup_obj,
                    refined_obj,
                    build_dir=build_dir,
                    timeout=refine_timeout,
                    snap_grid_size=snap_grid_size,
                    number_of_iterations=number_of_iterations,
                )

                refined_V, refined_F = load_obj(refined_obj)
                current_mesh = Mesh(refined_V, refined_F)
                autorefined = True

                print(
                    f"[PIPELINE] CGAL autorefine produced "
                    f"V={current_mesh.num_vertices}, F={current_mesh.num_faces}"
                )
        else:
            print("[PIPELINE] pre-check skipped by configuration")

        post_cleanup = cleanup_topology(
            current_mesh,
            eps_v=eps_v_abs,
            area_eps=area_threshold_from_mesh(current_mesh),
        )
        _validate_post_cleanup_mesh(current_mesh)

        post_cleanup_obj = work_dir_path / "post_cleanup.obj"
        save_obj(post_cleanup_obj, current_mesh.V, current_mesh.F)

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

        if post_check.self_intersect:
            raise RuntimeError(
                f"最终 CGAL 检测失败，仍有 {post_check.count} 对自交三角面"
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
) -> List[MeshDict]:
    """逐个修复多个 OBJ 网格；当前版本不处理跨文件自交。"""
    repaired: List[MeshDict] = []

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