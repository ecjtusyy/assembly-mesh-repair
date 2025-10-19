# -*- coding: utf-8 -*-
"""
AABB（轴对齐包围盒）的小工具。
"""

import numpy as np


def tri_aabb(V, tri):
    """给一个三角形（索引 tri），计算它的 AABB。"""
    pts = V[tri]
    vmin = pts.min(axis=0)
    vmax = pts.max(axis=0)
    return vmin, vmax


def overlap(a_min, a_max, b_min, b_max, eps=0.0):
    """判断两个 AABB 是否相交（带一个微小容差）。"""
    return not (
        a_max[0] < b_min[0] - eps or a_min[0] > b_max[0] + eps or
        a_max[1] < b_min[1] - eps or a_min[1] > b_max[1] + eps or
        a_max[2] < b_min[2] - eps or a_min[2] > b_max[2] + eps
    )
