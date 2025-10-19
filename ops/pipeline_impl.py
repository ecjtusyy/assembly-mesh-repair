# -*- coding: utf-8 -*-
"""
管线调度模块：把所有阶段按顺序跑起来
顺序：预处理 -> 广义碰撞(Octree) -> 窄相(三角-三角) + 局部切割 -> T 口修复 -> 焊接/拼缝 -> 流形化 -> 质量清理 -> 校验
"""

import numpy as np

from mesh.mesh import Mesh
from mesh.validate import has_nonmanifold_edges
from mesh.manifold import fix_nonmanifold_edges, fix_nonmanifold_vertices
from ops.t_junction import fix_t_junctions
from ops.stitch import vertex_weld, weld_across_parts, edge_align_placeholder
from ops.quality import laplacian_smooth
from geom.octree import build_octree_for_meshes
from geom.intersection import tri_tri_intersection, split_edge_if_needed
from geom.predicates import normalize_eps


def preprocess_meshes(meshes):
    """把输入的 dict(V,F) 变成 Mesh 类，方便后续做拓扑与邻接操作。"""
    out = []
    for m in meshes:
        out.append(Mesh(m["V"], m["F"]))
    return out


def broad_phase_octree(meshes):
    """
    广义阶段：用 Octree 对所有三角形的 AABB 做快速对碰，得到候选的“可能相交”的三角形对。
    返回：列表 [((partA, triIdA), (partB, triIdB)), ...]
    """
    tree = build_octree_for_meshes(meshes)
    pairs = tree.query_pairs()
    return pairs


def narrow_phase_and_cut(meshes, candidates, eps_rel):
    """
    窄相阶段：对每一个候选三角形对，做三角-三角检测。
    如果发现确实相交，就在两边相关的边上插入交点（也就是拆边/加点），从而完成“局部切割”。
    注意：这里的几何谓词是近似的，面对退化/共面时不完全鲁棒。
    """
    # 按整体尺度把相对公差换算成绝对公差
    allV = [m["V"] for m in meshes]
    eps = normalize_eps(eps_rel, allV)

    # 遍历每一个候选三角形对
    for (pa, ta), (pb, tb) in candidates:
        Va, Fa = meshes[pa]["V"], meshes[pa]["F"]
        Vb, Fb = meshes[pb]["V"], meshes[pb]["F"]
        A = Fa[ta]
        B = Fb[tb]
        P = tri_tri_intersection(Va, A, Vb, B, eps=eps)  # 返回交点（去重后）
        if not P:
            continue

        # 在两个三角形的三条边上分别尝试插入交点
        ma = Mesh(Va, Fa)
        mb = Mesh(Vb, Fb)
        for p in P:
            for e in [(A[0], A[1]), (A[1], A[2]), (A[2], A[0])]:
                split_edge_if_needed(ma, e, p, eps=eps)
            for e in [(B[0], B[1]), (B[1], B[2]), (B[2], B[0])]:
                split_edge_if_needed(mb, e, p, eps=eps)
        meshes[pa]["V"], meshes[pa]["F"] = ma.V, ma.F
        meshes[pb]["V"], meshes[pb]["F"] = mb.V, mb.F


def repair_assembly_mesh(meshes_in, eps_v=1e-6, eps_e=1e-6, eps_p=1e-8):
    """
    装配网格完整修复流程（新手友好解释版）：
    - 输入：若干零件，每个是 dict{V,F}
    - 输出：相同结构的列表，每个零件都被修复到“更接近流形”的状态
    - eps_v：顶点焊接容差（相对）
    - eps_e：边对齐/T 口容差（相对）
    - eps_p：几何相交/共面判定容差（相对）
    """
    # 0) 复制输入，避免原地修改
    meshes = [{"V": m["V"].copy(), "F": m["F"].copy()} for m in meshes_in]

    # 1) 预处理为 Mesh 类（方便拓扑操作）
    ms = preprocess_meshes(meshes)

    # 2) 顶点焊接（各自网格内部）
    for mm in ms:
        vertex_weld(mm, eps_v=eps_v)

    # 3) 广义阶段（Octree），拿到候选三角形对
    #    注意：此处 tree 是基于合并后的顶点布局做的，真实工程里通常更细化
    meshes_for_broad = [{"V": mm.V, "F": mm.F} for mm in ms]
    tree = build_octree_for_meshes(meshes_for_broad)
    candidates = tree.query_pairs()

    # 4) 窄相 + 局部切割
    narrow_phase_and_cut(meshes_for_broad, candidates, eps_rel=eps_p)

    # 更新 ms 为切割后的网格
    ms = [Mesh(m["V"], m["F"]) for m in meshes_for_broad]

    # 5) T 口修复（可能需要多轮传播）
    for mm in ms:
        fix_t_junctions(mm, eps_e=eps_e)

    # 6) 跨零件焊接（演示版本是“全局合并顶点+再拆回面”）
    #    注意：这里会把各零件共享同一份顶点数组（演示方便）；如果你要严格保持零件边界，需要在 weld_across_parts 里做精细的索引管理。
    meshes = [{"V": mm.V, "F": mm.F} for mm in ms]
    weld_across_parts(meshes, eps_v=eps_v)

    # 把 dict 重新转回 Mesh，方便做流形化
    ms = [Mesh(m["V"], m["F"]) for m in meshes]

    # 7) 流形化（边与点两类非流形情况）
    for mm in ms:
        fix_nonmanifold_edges(mm, angle_thresh_deg=45.0)
        fix_nonmanifold_vertices(mm)

    # 8) 质量清理（轻量版的 Laplacian 平滑）
    for mm in ms:
        laplacian_smooth(mm, iterations=2, lam=0.25)

    # 9) 简单校验提示（示意）
    for mm in ms:
        mm.rebuild()
        if has_nonmanifold_edges(mm):
            print("[Warn] 仍然存在非流形边（这是演示实现，复杂数据需要更鲁棒的算法）。")

    # 10) 返回字典形式
    return [{"V": mm.V, "F": mm.F} for mm in ms]
