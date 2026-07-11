"""生成四个真实模型的修复前后对比图和验收报告。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh.components import split_edge_components
from mesh.diagnostics import topology_summary
from mesh.io_obj import ObjMesh, load_obj_data, save_obj_data
from ops.pipeline_impl import repair_mesh_data


DATA = ROOT / "tests" / "data"
DEFAULT_OUTPUT = ROOT / "docs" / "validation"

MODELS = {
    "soil_overlap": "土块加底土（相互穿透、存在体积重叠）.obj",
    "pit_overlap": "基坑1.0（存在多部分贴合和局部重叠）.obj",
    "cell_overlap": "基坑单元格未合并（存在多部分贴合和局部重叠）.obj",
    "shared_surface": "整体元素土块底土（存在共享接触面）.obj",
}

COLORS = [
    "#3b82f6",
    "#ef4444",
    "#22c55e",
    "#f59e0b",
    "#a855f7",
    "#06b6d4",
    "#ec4899",
    "#84cc16",
]


def _mesh_volume(mesh: ObjMesh) -> float:
    tri = trimesh.Trimesh(vertices=mesh.V, faces=mesh.F, process=False)
    if not tri.is_watertight or not tri.is_winding_consistent:
        return 0.0
    return abs(float(tri.volume))


def _compact_topology(summary: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in summary.items()
        if key != "face_component_id"
    }


def _input_metrics(mesh: ObjMesh) -> dict[str, object]:
    parts = split_edge_components(mesh)
    part_volumes = [_mesh_volume(part) for part in parts]
    return {
        "vertices": int(len(mesh.V)),
        "faces": int(len(mesh.F)),
        "parts": int(len(parts)),
        "topology": _compact_topology(topology_summary(mesh.V, mesh.F)),
        "part_volume_sum": float(sum(part_volumes)),
        "closed_volume_parts": int(sum(volume > 0.0 for volume in part_volumes)),
    }


def _manifold_roundtrip(mesh: ObjMesh) -> bool:
    import manifold3d

    source = manifold3d.Mesh(
        vert_properties=np.ascontiguousarray(mesh.V, dtype=np.float32),
        tri_verts=np.ascontiguousarray(mesh.F, dtype=np.uint32),
    )
    return not manifold3d.Manifold(mesh=source).is_empty()


def build_case_report(source: ObjMesh, repaired: ObjMesh, run_report) -> dict[str, object]:
    before = _input_metrics(source)
    after = dict(run_report.output_validation)
    after["topology"] = _compact_topology(dict(after["topology"]))
    output_volume = float(after["volume"])
    volume_scale = max(float(before["part_volume_sum"]), output_volume, 1.0)
    overlap_volume = max(float(before["part_volume_sum"]) - output_volume, 0.0)
    overlap_tolerance = volume_scale * 1e-10

    checks = {
        "pipeline_success": run_report.status == "success",
        "exact_path_used": run_report.approximate_rebuild is False,
        "no_degenerate_faces": after["topology"]["degenerate_faces"] == 0,
        "no_duplicate_faces": after["topology"]["duplicate_faces"] == 0,
        "no_nonmanifold_edges": after["topology"]["nonmanifold_edges"] == 0,
        "no_nonmanifold_vertices": after["topology"]["nonmanifold_vertices"] == 0,
        "watertight": after["is_watertight"] is True,
        "winding_consistent": after["is_winding_consistent"] is True,
        "positive_volume": after["is_volume"] is True,
        "single_output_component": after["topology"]["edge_component_count"] == 1,
        "manifold3d_roundtrip": _manifold_roundtrip(repaired),
        "union_volume_not_larger_than_parts": (
            output_volume <= float(before["part_volume_sum"]) + overlap_tolerance
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"验收失败：{', '.join(failed)}")

    return {
        "before": before,
        "after": after,
        "overlap": {
            "removed_volume": overlap_volume,
            "detected": overlap_volume > overlap_tolerance,
            "relative_to_part_volume_sum": (
                overlap_volume / float(before["part_volume_sum"])
                if float(before["part_volume_sum"]) > 0.0
                else 0.0
            ),
        },
        "checks": checks,
        "all_checks_passed": True,
    }


def _set_equal_axes(axis, bounds: np.ndarray) -> None:
    center = bounds.mean(axis=0)
    radius = max(float(np.ptp(bounds, axis=0).max()) * 0.55, 1e-9)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_axis_off()


def _draw_mesh(axis, mesh: ObjMesh, color: str, alpha: float) -> None:
    triangles = mesh.V[mesh.F]
    collection = Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor="#1f2937",
        linewidth=0.18,
        alpha=alpha,
    )
    axis.add_collection3d(collection)


def _draw_before(axis, mesh: ObjMesh) -> None:
    for part_id, part in enumerate(split_edge_components(mesh)):
        _draw_mesh(axis, part, COLORS[part_id % len(COLORS)], 0.68)


def render_comparison(
    source: ObjMesh,
    repaired: ObjMesh,
    report: dict[str, object],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 7), dpi=180)
    bounds = np.vstack([source.V, repaired.V])
    views = [(24, -58), (90, -90)]

    for column, (elevation, azimuth) in enumerate(views):
        before_axis = fig.add_subplot(2, 2, column + 1, projection="3d")
        _draw_before(before_axis, source)
        _set_equal_axes(before_axis, bounds)
        before_axis.view_init(elev=elevation, azim=azimuth)
        before_axis.set_title(
            f"Before | {report['before']['parts']} parts | "
            f"{report['before']['faces']} faces",
            fontsize=11,
        )

        after_axis = fig.add_subplot(2, 2, column + 3, projection="3d")
        _draw_mesh(after_axis, repaired, "#14b8a6", 0.96)
        _set_equal_axes(after_axis, bounds)
        after_axis.view_init(elev=elevation, azim=azimuth)
        after_axis.set_title(
            f"After | 1 manifold solid | {report['after']['faces']} faces",
            fontsize=11,
        )

    overlap = report["overlap"]
    fig.suptitle(
        "Exact solid union | "
        f"removed overlap volume = {overlap['removed_volume']:.8g} | "
        "all checks passed",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _markdown_table(reports: dict[str, dict[str, object]]) -> str:
    lines = [
        "# 四个真实模型修复验证",
        "",
        "所有模型均使用 `solid` 精确路径，没有启用近似重建。",
        "",
        "| 模型 | 输入零件 | 输入面 | 输出面 | 删除重叠体积 | 闭合 | 绕序 | 正体积 | 非流形边/点 |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for slug, report in reports.items():
        before = report["before"]
        after = report["after"]
        topology = after["topology"]
        overlap = report["overlap"]
        lines.append(
            f"| {slug} | {before['parts']} | {before['faces']} | {after['faces']} | "
            f"{overlap['removed_volume']:.8g} | {after['is_watertight']} | "
            f"{after['is_winding_consistent']} | {after['is_volume']} | "
            f"{topology['nonmanifold_edges']}/{topology['nonmanifold_vertices']} |"
        )

    lines.extend(
        [
            "",
            "## 修复前后对比",
            "",
            "![soil overlap](soil_overlap.png)",
            "",
            "![pit overlap](pit_overlap.png)",
            "",
            "![cell overlap](cell_overlap.png)",
            "",
            "![shared surface](shared_surface.png)",
            "",
            "## 复现",
            "",
            "```bash",
            "python -m pip install -r requirements-visual.txt",
            "python tools/render_validation.py",
            "```",
            "",
            "脚本任一检查失败都会以非零状态退出，不会继续生成“成功”报告。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_dir = output_dir / "repaired"
    reports: dict[str, dict[str, object]] = {}

    for slug, filename in MODELS.items():
        source = load_obj_data(DATA / filename)
        repaired, run_report = repair_mesh_data(source, mode="solid")
        report = build_case_report(source, repaired, run_report)
        report["source"] = filename

        save_obj_data(repaired_dir / f"{slug}.obj", repaired)
        render_comparison(source, repaired, report, output_dir / f"{slug}.png")
        reports[slug] = report

    payload = {
        "models": reports,
        "all_models_passed": all(
            report["all_checks_passed"] for report in reports.values()
        ),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _markdown_table(reports),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
