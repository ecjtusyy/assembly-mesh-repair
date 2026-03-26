from __future__ import annotations

import unittest
from pathlib import Path

from mesh.io_obj import load_obj
from mesh.mesh import Mesh
from ops.pipeline_impl import python_cleanup_only
from ops.stitch import mesh_has_degenerate_faces, mesh_has_duplicate_faces


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


class PythonCleanupTests(unittest.TestCase):
    def test_clean_tri_remains_clean(self) -> None:
        V, F = load_obj(DATA / "clean_tri.obj")
        mesh_dict, report, _eps_abs = python_cleanup_only({"V": V, "F": F}, eps_v=1e-9, eps_mode="absolute")
        self.assertEqual(report.V_before, 3)
        self.assertEqual(report.V_after, 3)
        self.assertEqual(report.F_before, 1)
        self.assertEqual(report.F_after, 1)
        mesh = Mesh(mesh_dict["V"], mesh_dict["F"])
        self.assertFalse(mesh_has_degenerate_faces(mesh))
        self.assertFalse(mesh_has_duplicate_faces(mesh))

    def test_dup_vertex_collapses_to_single_triangle(self) -> None:
        V, F = load_obj(DATA / "dup_vertex.obj")
        mesh_dict, report, _eps_abs = python_cleanup_only({"V": V, "F": F}, eps_v=1e-12, eps_mode="absolute")
        self.assertEqual(report.V_before, 4)
        self.assertEqual(report.V_after, 3)
        self.assertEqual(report.F_before, 2)
        self.assertEqual(report.F_after, 1)
        self.assertEqual(report.merged_vertices, 1)
        self.assertEqual(report.duplicate_removed, 1)
        mesh = Mesh(mesh_dict["V"], mesh_dict["F"])
        self.assertFalse(mesh_has_degenerate_faces(mesh))
        self.assertFalse(mesh_has_duplicate_faces(mesh))

    def test_mixed_case_precleanup_only_removes_duplicate_triangle(self) -> None:
        V, F = load_obj(DATA / "mixed_case.obj")
        mesh_dict, report, _eps_abs = python_cleanup_only({"V": V, "F": F}, eps_v=1e-12, eps_mode="absolute")
        self.assertEqual(report.V_before, 7)
        self.assertEqual(report.V_after, 6)
        self.assertEqual(report.F_before, 3)
        self.assertEqual(report.F_after, 2)
        mesh = Mesh(mesh_dict["V"], mesh_dict["F"])
        self.assertFalse(mesh_has_degenerate_faces(mesh))
        self.assertFalse(mesh_has_duplicate_faces(mesh))


if __name__ == "__main__":
    unittest.main()
