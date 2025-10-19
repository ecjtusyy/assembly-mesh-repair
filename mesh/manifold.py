# -*- coding: utf-8 -*-
"""
流形化（Manifoldization）：
- 如果一条边被超过两个三角形共享，就把这些三角形按法向做聚类，然后“复制边端点”，把多余的面分组重连；
- 如果一个顶点处有多于一个“扇形环”，就复制顶点，把不同的环分开。
"""

import numpy as np
from collections import defaultdict
from mesh.mesh import Mesh


def cluster_faces_by_normal(mesh: Mesh, faces: list, angle_thresh_deg=45.0):
    """
    按法向相似度对一组面做分簇（非常简化）。
    做法：算每个三角形的法向，再用阈值做“顺序分组”。
    返回：若干簇（每个簇是 face 索引列表）。
    """
    V, F = mesh.V, mesh.F
    normals = []
    for fi in faces:
        a, b, c = F[fi]
        n = np.cross(V[b] - V[a], V[c] - V[a])
        nlen = np.linalg.norm(n) + 1e-18
        n = n / nlen
        normals.append(n)
    normals = np.array(normals)

    clusters = []
    used = np.zeros(len(faces), dtype=bool)
    cos_thresh = np.cos(np.deg2rad(angle_thresh_deg))

    for i in range(len(faces)):
        if used[i]:
            continue
        group = [faces[i]]
        used[i] = True
        for j in range(i + 1, len(faces)):
            if used[j]:
                continue
            if np.dot(normals[i], normals[j]) >= cos_thresh:
                group.append(faces[j])
                used[j] = True
        clusters.append(group)
    return clusters


def fix_nonmanifold_edges(mesh: Mesh, angle_thresh_deg=45.0):
    """
    处理“边的邻接面数量 > 2”的情况：
    - 先按法向把 incident faces 分组；
    - 第一组保留原顶点，其他组复制边端点并改写三角形；
    """
    mesh.rebuild()
    V, F = mesh.V, mesh.F
    e2f = mesh.edge2faces
    new_faces = []
    removed = set()

    def key(a, b): return (a, b) if a < b else (b, a)

    for e, inc in list(e2f.items()):
        if len(inc) <= 2:
            continue
        # 对这条边上的面做分组
        clusters = cluster_faces_by_normal(mesh, inc, angle_thresh_deg)
        # 保留簇 0 在原边；其他簇复制边端点，改写这些面的顶点索引
        v0, v1 = e
        for cid, group in enumerate(clusters):
            if cid == 0:
                continue
            v0_new = mesh.add_vertex(V[v0].copy())
            v1_new = mesh.add_vertex(V[v1].copy())
            for fi in group:
                a, b, c = F[fi]
                tri = [a, b, c]
                # 找到包含该边的有向边位置，并把其端点替换成新复制的顶点
                for kk in range(3):
                    u, v = tri[kk], tri[(kk+1) % 3]
                    if key(u, v) == e:
                        tri[kk] = v0_new if u in e else tri[kk]
                        tri[(kk+1) % 3] = v1_new if v in e else tri[(kk+1) % 3]
                new_faces.append(tri)
                removed.add(fi)

    if removed:
        # 删除被替换的旧面，加入新面
        mesh.replace_faces(list(removed), new_faces)
    mesh.mark_dirty()


def fix_nonmanifold_vertices(mesh: Mesh):
    """
    处理“一个顶点上有多个不连通的扇形环”的情况：
    - 对该顶点的 incident faces 构建局部邻接；
    - 做连通分量划分；
    - 对每个额外的连通分量复制该顶点，并改写面顶点。
    """
    mesh.rebuild()
    V, F = mesh.V, mesh.F
    # 先统计每个顶点对应的 incident faces
    vert2faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(F):
        vert2faces[a].append(fi)
        vert2faces[b].append(fi)
        vert2faces[c].append(fi)

    visited = set()
    for v, inc in vert2faces.items():
        # 构建局部“面-面邻接”（只要两个面共享经过 v 的边，就算邻接）
        local_adj = defaultdict(set)
        for fi in inc:
            a, b, c = F[fi]
            face_vs = [a, b, c]
            for fj in inc:
                if fi == fj:
                    continue
                aa, bb, cc = F[fj]
                other_vs = [aa, bb, cc]
                common = set(face_vs) & set(other_vs)
                # 如果共享的顶点包含 v 且共享数量 >=2，说明共享了通过 v 的边
                if v in common and len(common) >= 2:
                    local_adj[fi].add(fj)

        # 在 incident faces 子图中做连通分量划分
        components = []
        visited = set()
        for fi in inc:
            if fi in visited:
                continue
            comp = []
            stack = [fi]
            visited.add(fi)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in local_adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            components.append(comp)

        if len(components) <= 1:
            continue

        # 如果有多个连通分量：为每个额外分量复制一个顶点，并把该分量内所有面里的 v 改成新顶点
        for comp in components[1:]:
            v_new = mesh.add_vertex(V[v].copy())
            for fi in comp:
                a, b, c = F[fi]
                tri = [a, b, c]
                for k in range(3):
                    if tri[k] == v:
                        tri[k] = v_new
                F[fi] = tri
        mesh.mark_dirty()
