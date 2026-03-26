#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entrypoint for the Python + CGAL bridge prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from mesh.io_obj import load_obj, save_obj
from ops.pipeline_impl import repair_single_mesh



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OBJ self-intersection repair prototype: Python vertex welding / cleanup + "
            "CGAL checker + CGAL autorefinement + cleanup + CGAL checker"
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="One or more input OBJ files")
    parser.add_argument("--output_dir", required=True, help="Directory for repaired OBJ files")
    parser.add_argument(
        "--eps_v",
        type=float,
        default=1e-9,
        help="Vertex welding tolerance. Interpretation depends on --eps_mode.",
    )
    parser.add_argument(
        "--eps_mode",
        choices=("absolute", "relative_bbox"),
        default="relative_bbox",
        help="Interpret --eps_v as either absolute model units or relative to bbox diagonal.",
    )
    parser.add_argument(
        "--build_dir",
        default="build/cgal",
        help="Directory containing the CGAL bridge executables",
    )
    parser.add_argument("--checker_timeout", type=int, default=60, help="Timeout in seconds for the CGAL checker")
    parser.add_argument("--refine_timeout", type=int, default=300, help="Timeout in seconds for CGAL autorefine")
    parser.add_argument("--snap_grid_size", type=int, default=23, help="CGAL iterative snap rounding grid size")
    parser.add_argument(
        "--number_of_iterations",
        type=int,
        default=5,
        help="Maximum CGAL iterative snap rounding iterations",
    )
    parser.add_argument(
        "--report_json",
        default=None,
        help="Optional path for a JSON report capturing per-file counts and checker results",
    )
    return parser



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
        work_dir = out_dir / f"{base}_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CLI] loading {input_path}")
        V, F = load_obj(input_path)
        if V.size == 0 or F.size == 0:
            print(f"[CLI][ERROR] {input_path} does not contain a non-empty v/f triangle soup", file=sys.stderr)
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
            print(f"[CLI] wrote {output_path}")
            reports.append(
                {
                    "input": str(input_path),
                    "output": str(output_path),
                    **report.as_dict(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - CLI must report full context.
            any_error = True
            print(f"[CLI][ERROR] processing failed for {input_path}: {exc}", file=sys.stderr)

    if args.report_json is not None:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[CLI] wrote report {report_path}")

    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
