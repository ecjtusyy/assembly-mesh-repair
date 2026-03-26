# -*- coding: utf-8 -*-
"""Deprecated teaching-only polygon retriangulation helpers."""

from __future__ import annotations



def retriangulate_polygon(*args, **kwargs):
    raise RuntimeError(
        "geom.retriangle.retriangulate_polygon() is not part of the production repair path anymore. "
        "Triangle-soup splitting is handled by CGAL autorefine_triangle_soup()."
    )
