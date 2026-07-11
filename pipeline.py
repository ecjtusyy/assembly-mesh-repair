"""OBJ 修复命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mesh.io_obj import load_obj_data, save_obj_data
from ops.pipeline_impl import repair_mesh_data


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
    parser.add_argument("--report_json", default=None, help="汇总报告路径")
    return parser


def repair_one_obj(input_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    mesh = load_obj_data(input_path)
    output, report = repair_mesh_data(
        mesh,
        mode=args.mode,
        eps_v=args.eps_v,
        eps_mode=args.eps_mode,
        fill_holes=args.fill_holes,
        max_hole_edges=args.max_hole_edges,
        approximate_rebuild=args.approximate_rebuild,
        rebuild_resolution=args.rebuild_resolution,
    )

    output_path = output_dir / f"{input_path.stem}_{args.mode}_repaired.obj"
    save_obj_data(output_path, output)

    result = report.as_dict()
    result["input"] = str(input_path)
    result["output"] = str(output_path)
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
