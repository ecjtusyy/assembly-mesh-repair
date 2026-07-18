"""OBJ 修复命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mesh.io_obj import load_obj_data, save_obj_data
from ops.pipeline_impl import repair_mesh_data
from volume.gmsh_tetra import TetrahedralizationOptions, tetrahedralize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="装配体三角网格修复")
    parser.add_argument("--input", nargs="+", required=True, help="一个或多个 OBJ 文件")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument(
        "--mode",
        choices=("assembly", "surface", "solid"),
        default="assembly",
        help="保留装配体、修复开放表面或合并为实体",
    )
    parser.add_argument("--eps_v", type=float, default=0.0, help="零件内部焊点容差")
    parser.add_argument(
        "--eps_mode",
        choices=("absolute", "relative_bbox"),
        default="relative_bbox",
    )
    parser.add_argument("--fill_holes", action="store_true", help="填补三角形和四边形小孔")
    parser.add_argument("--max_hole_edges", type=int, default=4, help="自动补洞的最大边数")
    parser.add_argument(
        "--approximate_rebuild",
        action="store_true",
        help="精确实体合并失败时允许近似重建",
    )
    parser.add_argument("--rebuild_resolution", type=int, default=50000)
    parser.add_argument(
        "--uniform_refine_levels",
        type=int,
        default=0,
        help="Gmsh 均匀细分级数，每一级把一个三角形拆成四个",
    )
    parser.add_argument(
        "--tetrahedralize",
        action="store_true",
        help="在 solid 表面验收后生成单区域四面体体网格",
    )
    parser.add_argument(
        "--target_size",
        type=float,
        default=0.0,
        help="目标单元尺寸；0 表示包围盒对角线的 1/8",
    )
    parser.add_argument(
        "--surface_angle",
        type=float,
        default=40.0,
        help="Gmsh 离散表面分类角度（度）",
    )
    parser.add_argument(
        "--min_tet_quality",
        type=float,
        default=0.05,
        help="四面体 mean-ratio 最低验收值，范围 0 到 1",
    )
    parser.add_argument(
        "--max_geometry_deviation_rel",
        type=float,
        default=1e-6,
        help="体网格边界相对包围盒的最大允许偏差",
    )
    parser.add_argument(
        "--max_volume_error_rel",
        type=float,
        default=1e-6,
        help="四面体总体积与修复表面体积的最大相对误差",
    )
    parser.add_argument(
        "--no_tet_optimize",
        action="store_true",
        help="关闭 Gmsh Netgen 四面体质量优化",
    )
    parser.add_argument("--report_json", default=None, help="汇总报告路径")
    return parser


def repair_one_obj(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    mesh = load_obj_data(input_path)
    if args.tetrahedralize and args.mode != "solid":
        raise ValueError("--tetrahedralize 只能与 --mode solid 一起使用")

    materials = sorted({name for name in mesh.face_material if name})
    if args.tetrahedralize and len(materials) > 1:
        raise ValueError(
            "第一阶段只支持单材料区域；检测到多个 usemtl，"
            "不能悄悄合并材料界面："
            + ", ".join(materials)
        )
    output, report = repair_mesh_data(
        mesh,
        mode=args.mode,
        eps_v=args.eps_v,
        eps_mode=args.eps_mode,
        fill_holes=args.fill_holes,
        max_hole_edges=args.max_hole_edges,
        approximate_rebuild=args.approximate_rebuild,
        rebuild_resolution=args.rebuild_resolution,
        uniform_refine_levels=args.uniform_refine_levels,
    )

    suffix = f"_gmsh_l{args.uniform_refine_levels}" if args.uniform_refine_levels else ""
    output_path = output_dir / f"{input_path.stem}_{args.mode}_repaired{suffix}.obj"
    save_obj_data(output_path, output)

    result = report.as_dict()
    result["input"] = str(input_path)
    result["output"] = str(output_path)
    result["input_materials"] = materials
    if args.tetrahedralize:
        base = output_dir / f"{input_path.stem}_{args.mode}"
        _, volume_report = tetrahedralize(
            output,
            base.with_name(base.name + "_volume.msh"),
            base.with_name(base.name + "_quality.vtk"),
            options=TetrahedralizationOptions(
                target_size=args.target_size,
                surface_angle=args.surface_angle,
                min_quality=args.min_tet_quality,
                max_relative_deviation=args.max_geometry_deviation_rel,
                max_relative_volume_error=args.max_volume_error_rel,
                optimize=not args.no_tet_optimize,
            ),
            domain_name=materials[0] if len(materials) == 1 else "domain",
        )
        result["volume_mesh"] = volume_report
        if not bool(volume_report["success"]):
            result["status"] = "failed"
            result["error"] = "四面体网格验收失败：" + ", ".join(
                str(value) for value in volume_report["errors"]
            )
    return result


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    failed = False

    for value in args.input:
        input_path = Path(value)
        try:
            report = repair_one_obj(input_path, output_dir, args)
            reports.append(report)
            if report.get("status") == "failed":
                failed = True
                print(f"[失败] {input_path.name}: {report['error']}")
            else:
                print(f"[完成] {input_path.name} -> {report['output']}")
        except Exception as exc:
            failed = True
            reports.append(
                {
                    "input": str(input_path),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[失败] {input_path.name}: {exc}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
