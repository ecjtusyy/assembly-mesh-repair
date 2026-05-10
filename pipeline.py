#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBJ 修复命令行入口：Python 清理 + CGAL 修复 + 可选 Gmsh 网格加密。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from mesh.io_obj import load_obj, save_obj
from ops.pipeline_impl import repair_single_mesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OBJ 自相交修复流程：Python 顶点焊接/清理 + "
            "CGAL 检查 + CGAL 自动修复 + 后处理检查。"
        )
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="一个或多个输入 OBJ 文件。",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出目录，用于保存修复后的 OBJ 文件和中间结果。",
    )
    parser.add_argument(
        "--eps_v",
        type=float,
        default=1e-9,
        help="顶点焊接容差。具体含义由 --eps_mode 决定。",
    )
    parser.add_argument(
        "--eps_mode",
        choices=("absolute", "relative_bbox"),
        default="relative_bbox",
        help="顶点焊接容差模式：absolute 表示绝对长度；relative_bbox 表示相对 bbox 对角线。",
    )
    parser.add_argument(
        "--build_dir",
        default="build/cgal",
        help="CGAL 桥接程序所在目录。",
    )
    parser.add_argument(
        "--checker_timeout",
        type=int,
        default=60,
        help="CGAL 检查程序超时时间，单位为秒。",
    )
    parser.add_argument(
        "--refine_timeout",
        type=int,
        default=300,
        help="CGAL 自动修复程序超时时间，单位为秒。",
    )
    parser.add_argument(
        "--snap_grid_size",
        type=int,
        default=23,
        help="CGAL 迭代 snap rounding 的网格大小参数。",
    )
    parser.add_argument(
        "--number_of_iterations",
        type=int,
        default=5,
        help="CGAL 迭代 snap rounding 的最大迭代次数。",
    )

    # 修复完成后，是否继续对 repaired OBJ 做 Gmsh 三角网格加密。
    parser.add_argument(
        "--gmsh_refine",
        action="store_true",
        help="开启后，会在修复完成的 OBJ 上继续执行 Gmsh 网格加密。",
    )
    parser.add_argument(
        "--gmsh_refine_levels",
        type=int,
        default=1,
        help=(
            "手动指定 Gmsh 细分次数。"
            "只有没有设置 --gmsh_target_edge_length 和 --gmsh_target_edge_ratio 时才会使用。"
        ),
    )
    parser.add_argument(
        "--gmsh_target_edge_length",
        type=float,
        default=None,
        help=(
            "目标边长的绝对值。"
            "程序会根据修复后网格的最大边长，自动估算需要细分几次。"
        ),
    )
    parser.add_argument(
        "--gmsh_target_edge_ratio",
        type=float,
        default=None,
        help=(
            "目标边长相对 bbox 对角线的比例。"
            "例如 0.02 表示目标边长约为 bbox 对角线的 2%。"
            "当 --gmsh_target_edge_length 没有设置时才会使用。"
        ),
    )
    parser.add_argument(
        "--gmsh_max_refine_levels",
        type=int,
        default=5,
        help="自动估算细分次数时允许的最大值，防止面数意外爆炸。",
    )
    parser.add_argument(
        "--gmsh_keep_msh",
        action="store_true",
        help="保留 Gmsh 生成的中间 refined.msh 文件，方便调试。",
    )
    parser.add_argument(
        "--gmsh_terminal",
        type=int,
        choices=(0, 1),
        default=1,
        help="是否显示 Gmsh 终端日志。1 表示显示，0 表示隐藏。",
    )

    parser.add_argument(
        "--report_json",
        default=None,
        help="可选 JSON 报告路径，用于记录每个文件的修复和加密统计信息。",
    )

    return parser


def import_gmsh_deps():
    try:
        import gmsh  # type: ignore
        import meshio  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "使用 --gmsh_refine 需要额外依赖，请先安装：pip install gmsh meshio"
        ) from exc

    return gmsh, meshio


def collect_triangles(mesh) -> np.ndarray:
    triangles = []

    for cell in mesh.cells:
        if cell.type == "triangle":
            triangles.append(cell.data)

    if not triangles:
        raise RuntimeError("Gmsh 输出中没有找到 triangle 三角面片。")

    return np.vstack(triangles).astype(np.int64)


def mesh_max_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    points = np.asarray(V, dtype=np.float64)
    faces = np.asarray(F, dtype=np.int64)

    if points.size == 0 or faces.size == 0:
        return 0.0

    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]

    e01 = np.linalg.norm(p0 - p1, axis=1)
    e12 = np.linalg.norm(p1 - p2, axis=1)
    e20 = np.linalg.norm(p2 - p0, axis=1)

    return float(max(e01.max(), e12.max(), e20.max()))


def bbox_diagonal_length(V: np.ndarray) -> float:
    points = np.asarray(V, dtype=np.float64)

    if points.size == 0:
        return 0.0

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)

    return float(np.linalg.norm(bbox_max - bbox_min))


def choose_gmsh_refine_levels(
    V: np.ndarray,
    F: np.ndarray,
    *,
    fallback_levels: int,
    target_edge_length: float | None,
    target_edge_ratio: float | None,
    max_refine_levels: int,
) -> tuple[int, dict[str, object]]:
    if fallback_levels < 0:
        raise ValueError(f"--gmsh_refine_levels 不能是负数：{fallback_levels}")

    if max_refine_levels < 0:
        raise ValueError(f"--gmsh_max_refine_levels 不能是负数：{max_refine_levels}")

    current_max_edge = mesh_max_edge_length(V, F)
    bbox_diag = bbox_diagonal_length(V)

    if target_edge_length is not None:
        if target_edge_length <= 0:
            raise ValueError(f"--gmsh_target_edge_length 必须是正数：{target_edge_length}")

        target_h = float(target_edge_length)
        mode = "绝对目标边长"

    elif target_edge_ratio is not None:
        if target_edge_ratio <= 0:
            raise ValueError(f"--gmsh_target_edge_ratio 必须是正数：{target_edge_ratio}")

        if bbox_diag <= 0:
            raise ValueError("模型 bbox 对角线长度为 0，不能使用 --gmsh_target_edge_ratio。")

        target_h = float(bbox_diag * target_edge_ratio)
        mode = "相对 bbox 目标边长"

    else:
        levels = int(fallback_levels)
        estimated_max_edge_after = (
            current_max_edge / (2.0 ** levels)
            if current_max_edge > 0
            else 0.0
        )

        return levels, {
            "模式": "手动细分次数",
            "目标边长": None,
            "目标边长比例": None,
            "bbox对角线": bbox_diag,
            "当前最大边长": current_max_edge,
            "原始估算细分次数": levels,
            "实际使用细分次数": levels,
            "是否被最大细分次数截断": False,
            "估算加密后最大边长": estimated_max_edge_after,
        }

    if current_max_edge <= 0 or current_max_edge <= target_h:
        raw_levels = 0
    else:
        raw_levels = int(math.ceil(math.log2(current_max_edge / target_h)))

    used_levels = min(raw_levels, max_refine_levels)
    capped = used_levels < raw_levels

    estimated_max_edge_after = (
        current_max_edge / (2.0 ** used_levels)
        if current_max_edge > 0
        else 0.0
    )

    return used_levels, {
        "模式": mode,
        "目标边长": target_h,
        "目标边长比例": target_edge_ratio,
        "bbox对角线": bbox_diag,
        "当前最大边长": current_max_edge,
        "原始估算细分次数": raw_levels,
        "实际使用细分次数": used_levels,
        "最大允许细分次数": int(max_refine_levels),
        "是否被最大细分次数截断": capped,
        "估算加密后最大边长": estimated_max_edge_after,
    }


def write_tmp_stl(meshio, V: np.ndarray, F: np.ndarray, stl_path: Path) -> None:
    stl_path.parent.mkdir(parents=True, exist_ok=True)

    stl_mesh = meshio.Mesh(
        points=np.asarray(V[:, :3], dtype=np.float64),
        cells=[("triangle", np.asarray(F, dtype=np.int64))],
    )

    meshio.write(stl_path, stl_mesh)


def run_gmsh_refine(gmsh, stl_path: Path, msh_path: Path, levels: int, terminal: int) -> None:
    if levels < 0:
        raise ValueError(f"Gmsh 细分次数不能是负数：{levels}")

    msh_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()

    try:
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", int(terminal))

        gmsh.merge(str(stl_path))

        # 只细分已有三角网格，不重建几何。
        for i in range(levels):
            print(f"[Gmsh] 细分 {i + 1}/{levels}")
            gmsh.model.mesh.refine()

        gmsh.write(str(msh_path))

    finally:
        gmsh.finalize()


def read_refined_msh(meshio, msh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(msh_path)

    points = np.asarray(mesh.points[:, :3], dtype=np.float64)
    triangles = collect_triangles(mesh)

    # 只保留三角面真正用到的点，避免 OBJ 里出现孤立点。
    used = np.unique(triangles.reshape(-1))

    old_to_new = -np.ones(points.shape[0], dtype=np.int64)
    old_to_new[used] = np.arange(len(used), dtype=np.int64)

    new_points = points[used]
    new_triangles = old_to_new[triangles]

    return new_points.astype(np.float64), new_triangles.astype(np.int64)


def refine_repaired_mesh_with_gmsh(
    repaired_mesh: dict[str, np.ndarray],
    *,
    work_dir: Path,
    fallback_levels: int,
    target_edge_length: float | None,
    target_edge_ratio: float | None,
    max_refine_levels: int,
    terminal: int,
    keep_msh: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    gmsh, meshio = import_gmsh_deps()

    V = np.asarray(repaired_mesh["V"], dtype=np.float64)
    F = np.asarray(repaired_mesh["F"], dtype=np.int64)

    levels, level_report = choose_gmsh_refine_levels(
        V,
        F,
        fallback_levels=fallback_levels,
        target_edge_length=target_edge_length,
        target_edge_ratio=target_edge_ratio,
        max_refine_levels=max_refine_levels,
    )

    gmsh_dir = work_dir / "gmsh_refine"
    gmsh_dir.mkdir(parents=True, exist_ok=True)

    tmp_stl = gmsh_dir / "repaired_tmp.stl"
    tmp_msh = gmsh_dir / "refined.msh"

    print(f"[Gmsh] 加密前：顶点数={len(V)}，三角形数={len(F)}")
    print(
        "[Gmsh] 当前最大边长="
        f"{level_report['当前最大边长']}，"
        "目标边长="
        f"{level_report['目标边长']}，"
        "实际细分次数="
        f"{levels}"
    )

    if level_report.get("是否被最大细分次数截断"):
        print(
            "[Gmsh][警告] 估算细分次数超过限制，已被 "
            f"--gmsh_max_refine_levels={max_refine_levels} 截断。",
            file=sys.stderr,
        )

    write_tmp_stl(meshio, V, F, tmp_stl)
    run_gmsh_refine(gmsh, tmp_stl, tmp_msh, levels, terminal)

    refined_V, refined_F = read_refined_msh(meshio, tmp_msh)

    print(f"[Gmsh] 加密后：顶点数={len(refined_V)}，三角形数={len(refined_F)}")

    if not keep_msh and tmp_msh.exists():
        tmp_msh.unlink()

    report = {
        "是否开启": True,
        "加密前顶点数": int(len(V)),
        "加密前三角形数": int(len(F)),
        "加密后顶点数": int(len(refined_V)),
        "加密后三角形数": int(len(refined_F)),
        "工作目录": str(gmsh_dir),
        "临时STL": str(tmp_stl),
        "中间MSH": str(tmp_msh) if keep_msh else None,
        **level_report,
    }

    return {"V": refined_V, "F": refined_F}, report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    any_error = False

    for input_path_str in args.input:
        input_path = Path(input_path_str)
        base = input_path.stem

        output_path = out_dir / f"{base}_repaired.obj"
        refined_output_path = out_dir / f"{base}_repaired_refined.obj"

        work_dir = out_dir / f"{base}_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CLI] 读取输入文件：{input_path}")

        V, F = load_obj(input_path)

        if V.size == 0 or F.size == 0:
            print(
                f"[CLI][错误] {input_path} 不包含非空的 v/f 三角面片数据。",
                file=sys.stderr,
            )
            any_error = True
            continue

        try:
            repaired_mesh, report = repair_single_mesh(
                {"V": V.astype(np.float64), "F": F.astype(np.int64)},
                eps_v=args.eps_v,
                eps_mode=args.eps_mode,
                build_dir=args.build_dir,
                checker_timeout=args.checker_timeout,
                refine_timeout=args.refine_timeout,
                snap_grid_size=args.snap_grid_size,
                number_of_iterations=args.number_of_iterations,
                run_precheck=True,
                run_postcheck=True,
                work_dir=work_dir,
            )

            save_obj(output_path, repaired_mesh["V"], repaired_mesh["F"])
            print(f"[CLI] 已写出修复后 OBJ：{output_path}")

            gmsh_report: dict[str, object] = {"是否开启": False}

            if args.gmsh_refine:
                refined_mesh, gmsh_report = refine_repaired_mesh_with_gmsh(
                    repaired_mesh,
                    work_dir=work_dir,
                    fallback_levels=args.gmsh_refine_levels,
                    target_edge_length=args.gmsh_target_edge_length,
                    target_edge_ratio=args.gmsh_target_edge_ratio,
                    max_refine_levels=args.gmsh_max_refine_levels,
                    terminal=args.gmsh_terminal,
                    keep_msh=args.gmsh_keep_msh,
                )

                save_obj(refined_output_path, refined_mesh["V"], refined_mesh["F"])
                gmsh_report["输出OBJ"] = str(refined_output_path)

                print(f"[CLI] 已写出加密后 OBJ：{refined_output_path}")

            reports.append(
                {
                    "输入文件": str(input_path),
                    "修复输出OBJ": str(output_path),
                    "Gmsh加密": gmsh_report,
                    **report.as_dict(),
                }
            )

        except Exception as exc:  # CLI 入口需要保留完整错误上下文。
            any_error = True
            print(f"[CLI][错误] 处理 {input_path} 失败：{exc}", file=sys.stderr)

    if args.report_json is not None:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(reports, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[CLI] 已写出 JSON 报告：{report_path}")

    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())