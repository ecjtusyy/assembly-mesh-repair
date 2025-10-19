# -*- coding: utf-8 -*-
"""
公差处理与简单“相对尺度”换算：把相对 eps 换成绝对长度（基于包围盒对角线）。
"""

import numpy as np


def normalize_eps(eps, V_list):
    """
    把“相对公差 eps”转成“绝对公差”（以所有顶点的全局包围盒对角线为参考）。
    这样不同尺寸/单位的模型就能用同一组相对参数。
    """
    mins = []
    maxs = []
    for V in V_list:
        if V.size == 0:
            continue
        mins.append(V.min(axis=0))
        maxs.append(V.max(axis=0))
    if not mins:
        return eps
    gmin = np.min(np.stack(mins), axis=0)
    gmax = np.max(np.stack(maxs), axis=0)
    diag = np.linalg.norm(gmax - gmin) + 1e-18
    return eps * diag
