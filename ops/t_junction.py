# -*- coding: utf-8 -*-
"""
T 口（挂点）检测与修复（带简单传播）。
"""

import numpy as np
from geom.intersection import split_edge_if_needed


def detect_hanging_points(mesh, eps_e=1e-9):
    """
    检测“顶点落在边上但不是该边端点”的情况。
    这里为了直观，采用简单做法：把每个顶点拿去和所有边做投影检测（O(NF*NV)）。
    大模型建议用空间哈希优化（本项目 utils/spatial_hash.py 可借鉴）。
    返回：列表 [(edge(tuple), point_pos(np.array)), ...]
    """
    V, F = mesh.V, mesh.F
    hangs = []

    def key(a, b): return (a, b) if a < b else (b, a)

    # 枚举所有边
    edges = set()
    for a, b, c in F:
        edges.add(key(a, b))
        edges.add(key(b, c))
        edges.add(key(c, a))
    edges = list(edges)

    # 把每个顶点与每条边做检查
    for vid, p in enumerate(V):
        for (u, v) in edges:
            if vid in (u, v):
                continue
            # 把点 p 投影到边 (u,v) 上，看看是不是“落在边段内部且距离很近”
            ab = V[v] - V[u]
            t = np.dot(p - V[u], ab) / (np.dot(ab, ab) + 1e-18)
            t = np.clip(t, 0.0, 1.0)
            proj = V[u] + t * ab
            if np.linalg.norm(proj - p) < eps_e:
                hangs.append(((u, v), p.copy()))
    return hangs


def fix_t_junctions(mesh, eps_e=1e-9):
    """
    修复 T 口：把挂点所在的“宿主边”拆开（在边上插入点），再对受影响的面做重连。
    简单做法：循环检测 -> 有变动就继续，最多迭代几次。
    """
    changed = True
    it = 0
    while changed and it < 5:  # 限制最多迭代 5 次，避免死循环
        it += 1
        changed = False
        hangs = detect_hanging_points(mesh, eps_e)
        for e, p in hangs:
            vid = split_edge_if_needed(mesh, e, p, eps=eps_e*0.1)
            if vid is not None:
                changed = True
    return changed
