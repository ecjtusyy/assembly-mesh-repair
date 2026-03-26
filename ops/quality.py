# -*- coding: utf-8 -*-
"""Geometry-moving post-processors are intentionally disabled by default."""

from __future__ import annotations



def laplacian_smooth(*args, **kwargs):
    raise RuntimeError(
        "Laplacian smoothing is disabled by default in this branch because it moves geometry."
    )
