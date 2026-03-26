# -*- coding: utf-8 -*-
"""Deprecated teaching-only narrow-phase helpers.

The production path in this branch no longer uses approximate triangle-triangle tests or
edge splitting from Python. Use the CGAL bridge instead.
"""

from __future__ import annotations


def tri_tri_intersection(*args, **kwargs):
    raise RuntimeError(
        "geom.intersection.tri_tri_intersection() is retained only as a deprecated teaching stub. "
        "Use cgal_bridge/check_self_intersections and cgal_bridge/autorefine_obj instead."
    )



def split_edge_if_needed(*args, **kwargs):
    raise RuntimeError(
        "geom.intersection.split_edge_if_needed() is retained only as a deprecated teaching stub. "
        "Edge insertion for self-intersection repair is now delegated to CGAL autorefinement."
    )
