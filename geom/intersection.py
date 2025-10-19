# -*- coding: utf-8 -*-
"""
三角形-三角形相交（基于线段最近点近似）以及局部拆边的小工具。
"""

import numpy as np


def seg_seg_3d(p0, p1, q0, q1, eps=1e-12):
    """
    3D 线段-线段最近点计算，并用“距离很小”判断是否相交/相切。
    返回：(dist, cp_on_p, cp_on_q)，其中 cp_* 是最近点。
    """
    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    denom = a * c - b * b
    if abs(denom) < eps:
        # 几乎平行：直接把 s=0，t=clamp(e/c)
        s = 0.0
        t = e / (c + eps)
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    cp_p = p0 + s * u
    cp_q = q0 + t * v
    dist = np.linalg.norm(cp_p - cp_q)
    return dist, cp_p, cp_q


def tri_edges(V, tri):
    """把三角面 tri 的三条边以 (v0,v1) 的形式返回。"""
    a, b, c = tri
    return [(a, b), (b, c), (c, a)]


def tri_tri_intersection(Va, A, Vb, B, eps=1e-12):
    """
    检测两个三角形是否相交，返回“候选交点”的列表（去重后）。
    这里采用“边-边最近点”近似：任意一条边与另一三角形的任一条边足够接近，就认为有交。
    注意：缺失对“严格共面重叠”的处理，仅作演示。
    """
    Pa = Va[A]
    Pb = Vb[B]
    P = []

    # 边-边最近点距离很小 -> 收集最近点
    for (i0, i1) in tri_edges(Va, A):
        p0, p1 = Va[i0], Va[i1]
        for (j0, j1) in tri_edges(Vb, B):
            q0, q1 = Vb[j0], Vb[j1]
            dist, cp_p, cp_q = seg_seg_3d(p0, p1, q0, q1, eps=eps)
            if dist < eps * 5:
                # 认为交了，收集一个代表点（取平均更稳一点）
                P.append(0.5 * (cp_p + cp_q))

    # 对收集到的点做简单去重
    Q = []
    for p in P:
        if len(Q) == 0:
            Q.append(p)
            continue
        if min(np.linalg.norm(p - q) for q in Q) > eps * 5:
            Q.append(p)
    return Q


def point_on_edge(V, e, p, eps=1e-12):
    """
    把点 p 投影到边 e=(a,b) 上，返回 (t, proj, inside_flag)
    其中 t 在 [0,1]，inside_flag 表示投影是否落在边段内部（含端点）。
    """
    a, b = e
    va, vb = V[a], V[b]
    ab = vb - va
    t = np.dot(p - va, ab) / (np.dot(ab, ab) + eps)
    t = np.clip(t, 0.0, 1.0)
    proj = va + t * ab
    inside = (np.linalg.norm(proj - p) <= eps * 5)
    return t, proj, inside


def split_edge_if_needed(mesh, e, p, eps=1e-12):
    """
    尝试在边 e=(a,b) 上插入点 p：
    - 如果 p 在边上（误差内），则把边拆成两半；
    - 需要把受影响的面重建（这里采取“简单重连”的方式，不做完整重三角）。
    返回：新增顶点的索引；若未拆分则返回 None。
    """
    a, b = e
    V, F = mesh.V, mesh.F
    t, proj, inside = point_on_edge(V, (a, b), p, eps=eps)
    if not inside:
        return None

    # 在几何上太靠近端点就不必插入（避免产生重复点）
    if np.linalg.norm(proj - V[a]) < eps * 5 or np.linalg.norm(proj - V[b]) < eps * 5:
        return None

    # 1) 新增顶点
    vid = mesh.add_vertex(proj)

    # 2) 找到所有包含边 (a,b) 的三角形，把这些三角形重连为两条边
    new_faces = []
    removed = []
    for fi, (x, y, z) in enumerate(F):
        tri = [x, y, z]
        edges = [(x, y), (y, z), (z, x)]
        # 边匹配需要无向，所以排序后比较
        def key(u, v): return (u, v) if u < v else (v, u)
        if any(key(u, v) == key(a, b) for (u, v) in edges):
            # 这张面要被替换：把 (a,b) 替换成 (a,vid) 和 (vid,b)，再与第三个点组成两张新三角
            removed.append(fi)
            c = ({x, y, z} - {a, b}).pop()
            new_faces.append([a, vid, c])
            new_faces.append([vid, b, c])

    # 3) 用新三角替换旧三角
    if removed:
        mesh.replace_faces(removed, new_faces)
    return vid
