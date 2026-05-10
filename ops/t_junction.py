# -*- coding: utf-8 -*-
"""
Python 侧 T-junction 传播修复
"""

from __future__ import annotations



def fix_t_junctions(*args, **kwargs):
    raise RuntimeError(
        "ops.t_junction.fix_t_junctions() is deprecated in the production path. "
        "Use the CGAL bridge path from ops.pipeline_impl instead."
    )
