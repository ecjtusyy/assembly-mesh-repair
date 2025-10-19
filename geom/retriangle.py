# -*- coding: utf-8 -*-
"""
局部“重三角化”小工具（极简）：扇形三角化。
复杂的质量改进/受约束重三角化不在这里实现。
"""

import numpy as np
from typing import List


def retriangulate_polygon(poly: np.ndarray) -> np.ndarray:
    """
    输入：按顺序给出的多边形顶点（索引或坐标以外，这里只返回面顶点在 poly 内的相对索引）
    输出：若干个三角形（扇形连接 0 号点）
    说明：为了简单，假定多边形“近似凸”，仅作演示。
    """
    tris = []
    for i in range(1, len(poly) - 1):
        tris.append([0, i, i + 1])
    return np.array(tris, dtype=np.int32)
