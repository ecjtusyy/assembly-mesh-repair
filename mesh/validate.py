# -*- coding: utf-8 -*-
"""
一些简单的网格合法性检查
"""

import numpy as np
from mesh.mesh import Mesh


def edge_degrees(mesh: Mesh):
    """返回一个 dict：无向边(tuple) -> 关联三角形个数。"""
    return {k: len(v) for k, v in mesh.edge2faces.items()}


def has_nonmanifold_edges(mesh: Mesh, max_degree=2) -> bool:
    """若存在 incident 面数 > max_degree 的边，则认为含有非流形边。"""
    degs = edge_degrees(mesh)
    return any(d > max_degree for d in degs.values())


def has_duplicate_vertices(V: np.ndarray, eps=1e-12) -> bool:
    """检测“非常接近”的重复顶点（O(N^2) 简单实现，适合小模型）。"""
    N = V.shape[0]
    for i in range(N):
        d = np.linalg.norm(V[i + 1:] - V[i], axis=1)
        if (d < eps).any():
            return True
    return False
