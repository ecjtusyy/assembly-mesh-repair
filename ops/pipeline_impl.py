# -*- coding: utf-8 -*-
"""New main pipeline: Python cleanup + CGAL checker/autorefine + cleanup + checker.

The old teaching route based on approximate triangle-triangle tests, Python-side edge
splitting, T-junction propagation, and smoothing is intentionally not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mesh.io_obj import load_obj, save_obj
from mesh.mesh import Mesh
from ops.cgal_refine import AutorefineResult, CGALBridgeError, CheckResult, autorefine_obj, check_self_intersections
from ops.stitch import (
    CleanupReport,
    area_threshold_from_mesh,
    cleanup_topology,
    mesh_has_degenerate_faces,
    mesh_has_duplicate_faces,
)


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
        return {
            "eps_mode": self.eps_mode,
            "eps_v_input": self.eps_v_input,
            "eps_v_abs": self.eps_v_abs,
            "pre_cleanup": self.pre_cleanup.as_dict(),
            "pre_check": None if self.pre_check is None else {
                "self_intersect": self.pre_check.self_intersect,
                "count": self.pre_check.count,
                "pairs": self.pre_check.pairs,
            },
            "autorefined": self.autorefined,
            "post_cleanup": self.post_cleanup.as_dict(),
            "post_check": {
                "self_intersect": self.post_check.self_intersect,
                "count": self.post_check.count,
                "pairs": self.post_check.pairs,
            },
            "final_vertices": self.final_vertices,
            "final_faces": self.final_faces,
        }



def bbox_diag(V: np.ndarray) -> float:
    if V.size == 0:
        return 0.0
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    return float(np.linalg.norm(hi - lo))



def normalize_eps(eps: float, V: np.ndarray, mode: str = "relative_bbox") -> float:
    if eps < 0:
        raise ValueError(f"epsilon must be non-negative, got {eps}")
    if mode == "absolute":
        return float(eps)
    if mode == "relative_bbox":
        diag = bbox_diag(V)
        return float(eps * diag)
    raise ValueError(f"Unknown eps_mode={mode!r}. Expected 'absolute' or 'relative_bbox'.")



def _validate_post_cleanup_mesh(mesh: Mesh) -> None:
    if mesh_has_degenerate_faces(mesh):
        raise RuntimeError("cleanup produced or retained degenerate faces")
    if mesh_has_duplicate_faces(mesh):
        raise RuntimeError("cleanup produced or retained duplicate faces")
    if mesh.F.size:
        min_idx = int(mesh.F.min())
        max_idx = int(mesh.F.max())
        if min_idx != 0:
            raise RuntimeError(f"face indices are not compacted from 0, got min={min_idx}")
        if max_idx != mesh.V.shape[0] - 1:
            raise RuntimeError(
                f"face indices are not fully compacted: max face index={max_idx}, vertex count={mesh.V.shape[0]}"
            )
    elif mesh.V.size:
        raise RuntimeError("mesh has vertices but no faces after cleanup; expected compact zero-vertex mesh")



def python_cleanup_only(
    mesh_in: Dict[str, np.ndarray],
    *,
    eps_v: float = 1e-9,
    eps_mode: str = "relative_bbox",
) -> tuple[Dict[str, np.ndarray], CleanupReport, float]:
    mesh = Mesh(np.asarray(mesh_in["V"], dtype=np.float64).copy(), np.asarray(mesh_in["F"], dtype=np.int64).copy())
    eps_v_abs = normalize_eps(eps_v, mesh.V, mode=eps_mode)
    report = cleanup_topology(mesh, eps_v=eps_v_abs, area_eps=area_threshold_from_mesh(mesh))
    _validate_post_cleanup_mesh(mesh)
    return {"V": mesh.V.copy(), "F": mesh.F.copy()}, report, eps_v_abs



def repair_single_mesh(
    mesh_in: Dict[str, np.ndarray],
    *,
    eps_v: float = 1e-9,
    eps_mode: str = "relative_bbox",
    build_dir: str | Path = "build/cgal",
    checker_timeout: int = 60,
    refine_timeout: int = 300,
    snap_grid_size: int = 23,
    number_of_iterations: int = 5,
    run_precheck: bool = True,
    run_postcheck: bool = True,
    work_dir: str | Path | None = None,
) -> tuple[Dict[str, np.ndarray], RepairRunReport]:
    pre_mesh_dict, pre_cleanup, eps_v_abs = python_cleanup_only(mesh_in, eps_v=eps_v, eps_mode=eps_mode)
    pre_mesh = Mesh(pre_mesh_dict["V"], pre_mesh_dict["F"])

    created_tmp_dir = False
    if work_dir is None:
        tmp_mgr = TemporaryDirectory(prefix="assembly_mesh_repair_")
        work_dir_path = Path(tmp_mgr.name)
        created_tmp_dir = True
    else:
        tmp_mgr = None
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        pre_cleanup_obj = work_dir_path / "pre_cleanup.obj"
        save_obj(pre_cleanup_obj, pre_mesh.V, pre_mesh.F)
        print(
            f"[PIPELINE] pre-cleanup: V {pre_cleanup.V_before} -> {pre_cleanup.V_after}, "
            f"F {pre_cleanup.F_before} -> {pre_cleanup.F_after}, eps_v_abs={eps_v_abs:.17g}"
        )

        pre_check: Optional[CheckResult] = None
        autorefined = False
        current_mesh = pre_mesh.copy()
        current_obj = pre_cleanup_obj

        if run_precheck:
            pre_check = check_self_intersections(pre_cleanup_obj, build_dir=build_dir, timeout=checker_timeout)
            print(f"[PIPELINE] pre-check: self_intersect={int(pre_check.self_intersect)} count={pre_check.count}")
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
                current_obj = refined_obj
                autorefined = True
                print(f"[PIPELINE] CGAL autorefine produced V={current_mesh.num_vertices}, F={current_mesh.num_faces}")
        else:
            print("[PIPELINE] pre-check skipped by configuration")

        post_cleanup = cleanup_topology(current_mesh, eps_v=eps_v_abs, area_eps=area_threshold_from_mesh(current_mesh))
        _validate_post_cleanup_mesh(current_mesh)
        post_cleanup_obj = work_dir_path / "post_cleanup.obj"
        save_obj(post_cleanup_obj, current_mesh.V, current_mesh.F)
        print(
            f"[PIPELINE] post-cleanup: V {post_cleanup.V_before} -> {post_cleanup.V_after}, "
            f"F {post_cleanup.F_before} -> {post_cleanup.F_after}"
        )

        if not run_postcheck:
            raise RuntimeError("run_postcheck=False is not allowed for the production path")
        post_check = check_self_intersections(post_cleanup_obj, build_dir=build_dir, timeout=checker_timeout)
        print(f"[PIPELINE] post-check: self_intersect={int(post_check.self_intersect)} count={post_check.count}")
        if post_check.self_intersect:
            raise RuntimeError(
                f"post-check failed: CGAL still reports {post_check.count} self-intersection pair(s)"
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
        if created_tmp_dir and tmp_mgr is not None:
            tmp_mgr.cleanup()



def repair_assembly_mesh(
    meshes_in: Sequence[Dict[str, np.ndarray]],
    *,
    eps_v: float = 1e-9,
    eps_mode: str = "relative_bbox",
    build_dir: str | Path = "build/cgal",
    checker_timeout: int = 60,
    refine_timeout: int = 300,
    snap_grid_size: int = 23,
    number_of_iterations: int = 5,
    run_precheck: bool = True,
    run_postcheck: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Repair multiple OBJ meshes independently.

    The first bridge prototype intentionally processes each input OBJ independently and
    does not attempt cross-file intersection repair.
    """
    repaired: List[Dict[str, np.ndarray]] = []
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
