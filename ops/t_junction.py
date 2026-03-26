# -*- coding: utf-8 -*-
"""Deprecated teaching-only T-junction fixer.

T-junction propagation is not used as the main 3D self-intersection repair route in this
branch.
"""

from __future__ import annotations



def fix_t_junctions(*args, **kwargs):
    raise RuntimeError(
        "ops.t_junction.fix_t_junctions() is deprecated in the production path. "
        "Use the CGAL bridge path from ops.pipeline_impl instead."
    )
