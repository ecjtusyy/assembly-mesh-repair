"""测试可视化验收报告中的硬性检查。"""

from pathlib import Path

from mesh.io_obj import load_obj_data
from ops.pipeline_impl import repair_mesh_data
from tools.render_validation import build_case_report


DATA = Path(__file__).resolve().parent / "data"


def test_real_overlap_case_passes_all_evidence_checks():
    source = load_obj_data(DATA / "土块加底土（相互穿透、存在体积重叠）.obj")
    repaired, run_report = repair_mesh_data(source, mode="solid")
    report = build_case_report(source, repaired, run_report)

    assert report["all_checks_passed"] is True
    assert report["overlap"]["detected"] is True
    assert all(report["checks"].values())

