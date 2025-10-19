# -*- coding: utf-8 -*-
"""
超简版 Octree：用于三角形 AABB 的广义阶段加速。
"""

import numpy as np
from geom.aabb import overlap


class OctNode:
    def __init__(self, center, half, depth=0, max_depth=12, max_items=64):
        # 节点的 AABB 由中心点 center 和半尺寸 half 定义
        self.center = center
        self.half = half
        self.depth = depth
        self.max_depth = max_depth
        self.max_items = max_items
        self.items = []      # 存放 (bb_min, bb_max, payload)
        self.children = None # 八个子节点

    def aabb(self):
        """返回该节点的 AABB（min, max）。"""
        c, h = self.center, self.half
        return c - h, c + h

    def subdivide(self):
        """将当前节点一分为八（八叉树），并创建子节点。"""
        if self.children is not None:
            return
        c, h = self.center, self.half
        hh = h / 2.0
        offs = np.array([[sx, sy, sz] for sx in (-hh[0], hh[0]) for sy in (-hh[1], hh[1]) for sz in (-hh[2], hh[2])])
        self.children = []
        for o in offs:
            self.children.append(OctNode(c + o, hh, self.depth + 1, self.max_depth, self.max_items))

    def insert(self, bb_min, bb_max, payload):
        """
        向树里插入一个 AABB 与其负载（payload，一般是 (part_id, tri_id)）。
        规则：节点未满或达到最大深度时放在当前节点，否则分裂给子节点。
        """
        if self.children is None and (len(self.items) < self.max_items or self.depth >= self.max_depth):
            self.items.append((bb_min, bb_max, payload))
            return
        if self.children is None:
            self.subdivide()
        inserted = False
        for ch in self.children:
            cmin, cmax = ch.aabb()
            if overlap(bb_min, bb_max, cmin, cmax, 0.0):
                ch.insert(bb_min, bb_max, payload)
                inserted = True
        if not inserted:
            # 如果没有完全落入任何子节点，就保留在当前节点
            self.items.append((bb_min, bb_max, payload))

    def query_pairs(self):
        """
        在每个叶节点里做一个简单的 n^2 配对，收集所有“可能相交”的 payload 对。
        注意：这里没有跨节点配对，简单但够用。
        """
        pairs = []
        L = len(self.items)
        for i in range(L):
            amin, amax, ap = self.items[i]
            for j in range(i + 1, L):
                bmin, bmax, bp = self.items[j]
                if overlap(amin, amax, bmin, bmax, 0.0):
                    pairs.append((ap, bp))
        # 递归子节点
        if self.children is not None:
            for ch in self.children:
                pairs.extend(ch.query_pairs())
        return pairs


def build_octree_for_meshes(meshes, max_depth=12, max_items=64):
    """
    为整个装配（多个网格）构建一个全局 Octree。
    简化处理：把所有三角形的 AABB 都放到同一棵树里。
    """
    # 计算全局包围盒，作为根节点 AABB 的范围
    mins = []
    maxs = []
    for m in meshes:
        vmin = m["V"].min(axis=0)
        vmax = m["V"].max(axis=0)
        mins.append(vmin)
        maxs.append(vmax)
    gmin = np.min(np.stack(mins), axis=0)
    gmax = np.max(np.stack(maxs), axis=0)
    center = 0.5 * (gmin + gmax)
    half = 0.5 * (gmax - gmin) + 1e-9  # 轻微膨胀，避免边界问题

    root = OctNode(center, half, 0, max_depth, max_items)

    # 把所有三角形插入到树里
    for pid, m in enumerate(meshes):
        V, F = m["V"], m["F"]
        for tid, tri in enumerate(F):
            bb_min = np.min(V[tri], axis=0)
            bb_max = np.max(V[tri], axis=0)
            root.insert(bb_min, bb_max, (pid, tid))
    return root
