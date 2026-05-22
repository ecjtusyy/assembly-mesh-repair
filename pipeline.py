#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBJ 修复命令行入口。

主流程只做调度：
1. 读取 OBJ 为 V/F；
2. 调用 repair_single_mesh 做 Python 清理 + CGAL 修复；
3. 保存 repaired OBJ；
4. 可选调用 Gmsh 对 repaired OBJ 继续细分；
5. 写出 JSON 统计报告。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from mesh.io_obj import load_obj, save_obj
from ops.pipeline_impl import repair_single_mesh

Mesh = dict[str, np.ndarray]
Report = dict[str, object]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    这里只定义用户能传什么参数，不做任何网格处理。
    真正的修复流程从 main() 开始。
    """
    parser = argparse.ArgumentParser(
        description=(
            "OBJ 自相交修复流程：Python 顶点焊接/清理 + "
            "CGAL 检查 + CGAL 自动修复 + 后处理检查。"
        )
    )

    parser.add_argument("--input", nargs="+", required=True, help="一个或多个输入 OBJ 文件。")
    parser.add_argument("--output_dir", required=True, help="输出目录。")

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
        help="absolute 为绝对长度；relative_bbox 为相对 bbox 对角线。",
    )

    parser.add_argument("--build_dir", default="build/cgal", help="CGAL 桥接程序所在目录。")
    parser.add_argument("--checker_timeout", type=int, default=60, help="CGAL 检查超时时间，单位秒。")
    parser.add_argument("--refine_timeout", type=int, default=300, help="CGAL 修复超时时间，单位秒。")
    parser.add_argument("--snap_grid_size", type=int, default=23, help="CGAL snap rounding 网格大小。")
    parser.add_argument("--number_of_iterations", type=int, default=5, help="CGAL snap rounding 最大迭代次数。")

    parser.add_argument("--gmsh_refine", action="store_true", help="修复后继续执行 Gmsh 三角网格细分。")
    parser.add_argument("--gmsh_refine_levels", type=int, default=1, help="手动指定 Gmsh 细分次数。")
    parser.add_argument("--gmsh_target_edge_length", type=float, default=None, help="目标边长绝对值。")
    parser.add_argument("--gmsh_target_edge_ratio", type=float, default=None, help="目标边长相对 bbox 对角线的比例。")
    parser.add_argument("--gmsh_max_refine_levels", type=int, default=5, help="自动估算细分次数的上限。")
    parser.add_argument("--gmsh_keep_msh", action="store_true", help="保留 Gmsh 生成的 refined.msh。")
    parser.add_argument("--gmsh_terminal", type=int, choices=(0, 1), default=1, help="是否显示 Gmsh 日志。")

    parser.add_argument("--report_json", default=None, help="可选 JSON 报告路径。")

    return parser


def import_gmsh_deps() -> tuple[object, object]:
    """导入 Gmsh 加密所需依赖。

    只有用户开启 --gmsh_refine 时才会调用，避免普通修复流程强依赖 gmsh/meshio。
    """
    try:
        import gmsh  # type: ignore
        import meshio  # type: ignore
    except ImportError as exc:
        raise RuntimeError("使用 --gmsh_refine 需要先安装：pip install gmsh meshio") from exc

    return gmsh, meshio


def collect_triangles(mesh) -> np.ndarray:
    """从 meshio 读出的网格中提取所有 triangle 单元。"""
    triangles = [cell.data for cell in mesh.cells if cell.type == "triangle"]
    if not triangles:
        raise RuntimeError("Gmsh 输出中没有找到 triangle 三角面片。")

    return np.vstack(triangles).astype(np.int64)


def mesh_max_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    """计算三角网格中的最大边长。

    F 中存的是顶点编号；先用编号回查 V 得到三角形三个点，再计算三条边长。
    """
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
    """计算点集 bbox 对角线长度，用来估计模型整体尺度。"""
    points = np.asarray(V, dtype=np.float64)
    if points.size == 0:
        return 0.0

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return float(np.linalg.norm(bbox_max - bbox_min))


def _manual_level_report(levels: int, current_max_edge: float, bbox_diag: float) -> Report:
    """生成手动细分次数模式下的报告。"""
    estimated_h = current_max_edge / (2.0**levels) if current_max_edge > 0 else 0.0

    return {
        "模式": "手动细分次数",
        "目标边长": None,
        "目标边长比例": None,
        "bbox对角线": bbox_diag,
        "当前最大边长": current_max_edge,
        "原始估算细分次数": levels,
        "实际使用细分次数": levels,
        "是否被最大细分次数截断": False,
        "估算加密后最大边长": estimated_h,
    }


def choose_gmsh_refine_levels(
    V: np.ndarray,
    F: np.ndarray,
    *,
    fallback_levels: int,
    target_edge_length: float | None,
    target_edge_ratio: float | None,
    max_refine_levels: int,
) -> tuple[int, Report]:
    """决定 Gmsh 需要细分几次。

    优先级：绝对目标边长 > 相对 bbox 目标边长 > 手动细分次数。
    估算依据是：Gmsh 每 refine 一次，最大边长大致减半。
    """
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
        return levels, _manual_level_report(levels, current_max_edge, bbox_diag)

    raw_levels = 0 if current_max_edge <= 0 or current_max_edge <= target_h else int(
        math.ceil(math.log2(current_max_edge / target_h))
    )
    used_levels = min(raw_levels, max_refine_levels)
    estimated_h = current_max_edge / (2.0**used_levels) if current_max_edge > 0 else 0.0

    return used_levels, {
        "模式": mode,
        "目标边长": target_h,
        "目标边长比例": target_edge_ratio,
        "bbox对角线": bbox_diag,
        "当前最大边长": current_max_edge,
        "原始估算细分次数": raw_levels,
        "实际使用细分次数": used_levels,
        "最大允许细分次数": int(max_refine_levels),
        "是否被最大细分次数截断": used_levels < raw_levels,
        "估算加密后最大边长": estimated_h,
    }


def write_tmp_stl(meshio, V: np.ndarray, F: np.ndarray, stl_path: Path) -> None:
    """把 V/F 三角网格临时写成 STL，供 Gmsh 读取。"""
    stl_path.parent.mkdir(parents=True, exist_ok=True)

    stl_mesh = meshio.Mesh(
        points=np.asarray(V[:, :3], dtype=np.float64),
        cells=[("triangle", np.asarray(F, dtype=np.int64))],
    )
    meshio.write(stl_path, stl_mesh)


def run_gmsh_refine(gmsh, stl_path: Path, msh_path: Path, levels: int, terminal: int) -> None:
    """调用 Gmsh 对已有三角网格做 levels 次细分。"""
    if levels < 0:
        raise ValueError(f"Gmsh 细分次数不能是负数：{levels}")

    msh_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", int(terminal))
        gmsh.merge(str(stl_path))

        # 只细分已有三角面，不重建 CAD 几何。
        for i in range(levels):
            print(f"[Gmsh] 细分 {i + 1}/{levels}")
            gmsh.model.mesh.refine()

        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()


def read_refined_msh(meshio, msh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 Gmsh 输出的 MSH，并转回干净的 V/F。

    只保留被三角面真正引用的点，然后用 old_to_new 重映射三角面索引。
    """
    mesh = meshio.read(msh_path)

    points = np.asarray(mesh.points[:, :3], dtype=np.float64)
    triangles = collect_triangles(mesh)

    used = np.unique(triangles.reshape(-1))
    old_to_new = -np.ones(points.shape[0], dtype=np.int64)
    old_to_new[used] = np.arange(len(used), dtype=np.int64)

    clean_V = points[used]
    clean_F = old_to_new[triangles]

    return clean_V.astype(np.float64), clean_F.astype(np.int64)


def refine_repaired_mesh_with_gmsh(
    repaired_mesh: Mesh,
    *,
    work_dir: Path,
    fallback_levels: int,
    target_edge_length: float | None,
    target_edge_ratio: float | None,
    max_refine_levels: int,
    terminal: int,
    keep_msh: bool,
) -> tuple[Mesh, Report]:
    """对 repaired mesh 做可选 Gmsh 加密，并返回加密后的 V/F 与统计报告。"""
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
    tmp_stl = gmsh_dir / "repaired_tmp.stl"
    tmp_msh = gmsh_dir / "refined.msh"

    print(f"[Gmsh] 加密前：顶点数={len(V)}，三角形数={len(F)}")
    print(
        f"[Gmsh] 当前最大边长={level_report['当前最大边长']}，"
        f"目标边长={level_report['目标边长']}，"
        f"实际细分次数={levels}"
    )

    if level_report.get("是否被最大细分次数截断"):
        print(
            f"[Gmsh][警告] 估算细分次数超过限制，已按 --gmsh_max_refine_levels={max_refine_levels} 截断。",
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


def repair_one_obj(input_path: Path, out_dir: Path, args: argparse.Namespace) -> Report:
    """处理单个 OBJ：读取、修复、可选加密、保存，并返回报告。"""
    base = input_path.stem
    work_dir = out_dir / f"{base}_work"
    repaired_path = out_dir / f"{base}_repaired.obj"
    refined_path = out_dir / f"{base}_repaired_refined.obj"

    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CLI] 读取输入文件：{input_path}")
    V, F = load_obj(input_path)

    if V.size == 0 or F.size == 0:
        raise RuntimeError(f"{input_path} 不包含非空的 v/f 三角面片数据。")

    repaired_mesh, repair_report = repair_single_mesh(
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

    save_obj(repaired_path, repaired_mesh["V"], repaired_mesh["F"])
    print(f"[CLI] 已写出修复后 OBJ：{repaired_path}")

    gmsh_report: Report = {"是否开启": False}
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
        save_obj(refined_path, refined_mesh["V"], refined_mesh["F"])
        gmsh_report["输出OBJ"] = str(refined_path)
        print(f"[CLI] 已写出加密后 OBJ：{refined_path}")

    return {
        "输入文件": str(input_path),
        "修复输出OBJ": str(repaired_path),
        "Gmsh加密": gmsh_report,
        **repair_report.as_dict(),
    }


def write_report_json(report_path: Path, reports: list[Report]) -> None:
    """把批处理结果写成 JSON，保留中文字段。"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[CLI] 已写出 JSON 报告：{report_path}")


def main(argv: list[str] | None = None) -> int:
    """命令行主入口：逐个处理输入 OBJ，并返回进程退出码。"""
    args = build_parser().parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[Report] = []
    any_error = False

    for input_name in args.input:
        try:
            reports.append(repair_one_obj(Path(input_name), out_dir, args))
        except Exception as exc:  # CLI 层保留错误，不让单个失败文件中断整批处理。
            any_error = True
            print(f"[CLI][错误] 处理 {input_name} 失败：{exc}", file=sys.stderr)

    if args.report_json is not None:
        write_report_json(Path(args.report_json), reports)

    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
