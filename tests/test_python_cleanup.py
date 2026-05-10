from __future__ import annotations

import unittest
from pathlib import Path

from mesh.io_obj import load_obj
from mesh.mesh import Mesh
from ops.pipeline_impl import python_cleanup_only
from ops.stitch import mesh_has_degenerate_faces, mesh_has_duplicate_faces


# 项目根目录和测试数据目录
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


class PythonCleanupTests(unittest.TestCase):
    """测试 Python 预清理流程，主要检查重复点、重复面和退化面。"""

    def test_clean_tri_remains_clean(self) -> None:
        """干净三角形经过清理后，顶点数和面数都不应该变化。"""

        # 读取一个本来就是干净的三角形
        V, F = load_obj(DATA / "clean_tri.obj")

        # 只运行 Python 预清理，不调用 CGAL
        mesh_dict, report, _eps_abs = python_cleanup_only(
            {"V": V, "F": F},
            eps_v=1e-9,
            eps_mode="absolute",
        )

        # 干净数据清理前后数量应该一致
        self.assertEqual(report.V_before, 3)
        self.assertEqual(report.V_after, 3)
        self.assertEqual(report.F_before, 1)
        self.assertEqual(report.F_after, 1)

        # 再检查结果里面不能出现退化面和重复面
        mesh = Mesh(mesh_dict["V"], mesh_dict["F"])
        self.assertFalse(mesh_has_degenerate_faces(mesh))
        self.assertFalse(mesh_has_duplicate_faces(mesh))

    def test_dup_vertex_collapses_to_single_triangle(self) -> None:
        """重复顶点应该被合并，合并后产生的重复三角面也应该被删除。"""

        # 这个数据里面有重复顶点，所以两个面最后会变成同一个面
        V, F = load_obj(DATA / "dup_vertex.obj")

        # eps_v 用 absolute，方便直接按给定容差合并顶点
        mesh_dict, report, _eps_abs = python_cleanup_only(
            {"V": V, "F": F},
            eps_v=1e-12,
            eps_mode="absolute",
        )

        # 顶点从 4 个合并到 3 个
        self.assertEqual(report.V_before, 4)
        self.assertEqual(report.V_after, 3)

        # 两个重复三角形最后只保留一个
        self.assertEqual(report.F_before, 2)
        self.assertEqual(report.F_after, 1)

        # 报告里面也要记录合并和删除的数量
        self.assertEqual(report.merged_vertices, 1)
        self.assertEqual(report.duplicate_removed, 1)

        # 最后再确认没有留下坏面
        mesh = Mesh(mesh_dict["V"], mesh_dict["F"])
        self.assertFalse(mesh_has_degenerate_faces(mesh))
        self.assertFalse(mesh_has_duplicate_faces(mesh))

    def test_mixed_case_precleanup_only_removes_duplicate_triangle(self) -> None:
        """混合测试数据里有重复点和重复面，清理后结果应该是合法的。"""

        # mixed_case 用来测试稍微复杂一点的预清理情况
        V, F = load_obj(DATA / "mixed_case.obj")

        # 这里只检查 Python cleanup 的效果，不检查 CGAL 自相交修复
        mesh_dict, report, _eps_abs = python_cleanup_only(
            {"V": V, "F": F},
            eps_v=1e-12,
            eps_mode="absolute",
        )

        # 检查清理前后的顶点数和面数
        self.assertEqual(report.V_before, 7)
        self.assertEqual(report.V_after, 6)
        self.assertEqual(report.F_before, 3)
        self.assertEqual(report.F_after, 2)

        # 清理后的 mesh 不能再有退化面和重复面
        mesh = Mesh(mesh_dict["V"], mesh_dict["F"])
        self.assertFalse(mesh_has_degenerate_faces(mesh))
        self.assertFalse(mesh_has_duplicate_faces(mesh))


if __name__ == "__main__":
    unittest.main()