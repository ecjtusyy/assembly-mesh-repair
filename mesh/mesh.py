# -*- coding: utf-8 -*-
"""
一个轻量的三角网格容器，提供按需构建的邻接关系（类似 half-edge 思想但非常简化）。
"""

import numpy as np


class Mesh:
    """
    三角网格：
      - V: (N, 3) float32/float64 顶点坐标
      - F: (M, 3) int32 三角形顶点索引
    提供若干邻接表：边到三角形、点到三角形、点到点、面到面等（按需构建）。
    """
    def __init__(self, V: np.ndarray, F: np.ndarray):
        self.V = V
        self.F = F
        self._edge2faces = None   # dict[(i,j)->[fi,...]]
        self._vert2faces = None   # list[np.ndarray]
        self._vert2verts = None   # list[np.ndarray]
        self._face2faces = None   # 目前未使用
        self._dirty = True        # 网格是否被修改，需要重建邻接

    def mark_dirty(self):
        """标记网格已被修改，下一次访问邻接表时会自动重建。"""
        self._dirty = True

    def _build_edge_table(self):
        """构建：边 -> 关联三角形索引列表；以及 顶点 -> 关联三角形索引列表。"""
        V, F = self.V, self.F
        M = F.shape[0]
        edge2faces = {}
        vert2faces = [[] for _ in range(V.shape[0])]

        def key(a, b):
            # 用无向边（i<j）作为键
            return (a, b) if a < b else (b, a)

        for fi in range(M):
            a, b, c = F[fi]
            # 记录三条边的面邻接
            for e in [(a, b), (b, c), (c, a)]:
                k = key(e[0], e[1])
                edge2faces.setdefault(k, []).append(fi)
            # 记录点到面的邻接
            vert2faces[a].append(fi)
            vert2faces[b].append(fi)
            vert2faces[c].append(fi)

        self._edge2faces = edge2faces
        # 转成 numpy，便于后续切片
        self._vert2faces = [np.array(lst, dtype=np.int32) for lst in vert2faces]

    def _build_vert_graph(self):
        """构建：顶点 -> 1-ring 邻接顶点（通过三角形推导）。"""
        V, F = self.V, self.F
        N = V.shape[0]
        nbrs = [set() for _ in range(N)]
        for a, b, c in F:
            nbrs[a].update([b, c])
            nbrs[b].update([a, c])
            nbrs[c].update([a, b])
        self._vert2verts = [np.array(sorted(list(s)), dtype=np.int32) for s in nbrs]

    def rebuild(self):
        """统一重建所有简单邻接结构。"""
        self._build_edge_table()
        self._build_vert_graph()
        self._dirty = False

    @property
    def edge2faces(self):
        if self._dirty or self._edge2faces is None:
            self._build_edge_table()
        return self._edge2faces

    @property
    def vert2faces(self):
        if self._dirty or self._vert2faces is None:
            self._build_edge_table()
        return self._vert2faces

    @property
    def vert2verts(self):
        if self._dirty or self._vert2verts is None:
            self._build_vert_graph()
        return self._vert2verts

    def bbox(self):
        """返回包围盒 (min, max)。"""
        return self.V.min(axis=0), self.V.max(axis=0)

    # ---- 一些简单的编辑/清理操作 ----

    def remove_degenerate_faces(self, eps=1e-18):
        """删除退化三角形（面积接近 0 或边长为 0）。"""
        V, F = self.V, self.F
        keep = []
        for a, b, c in F:
            ab = V[b] - V[a]
            ac = V[c] - V[a]
            area2 = np.linalg.norm(np.cross(ab, ac))
            if area2 > eps:
                keep.append([a, b, c])
        self.F = np.array(keep, dtype=np.int32) if keep else np.zeros((0, 3), dtype=np.int32)
        self.mark_dirty()

    def add_vertex(self, p: np.ndarray) -> int:
        """在末尾新增一个顶点，返回新顶点的索引。"""
        self.V = np.vstack([self.V, p.reshape(1, 3)])
        self._vert2faces = None
        self._vert2verts = None
        self._dirty = True
        return self.V.shape[0] - 1

    def replace_faces(self, face_ids, new_faces):
        """
        用 new_faces 替换掉 face_ids 指定的面集（常用于局部重三角化后重写）。
        新三角形顶点编号应基于当前 self.V。
        """
        mask = np.ones(self.F.shape[0], dtype=bool)
        mask[face_ids] = False
        F_keep = self.F[mask]
        if len(new_faces) > 0:
            F_new = np.array(new_faces, dtype=np.int32)
            self.F = np.vstack([F_keep, F_new])
        else:
            self.F = F_keep
        self.mark_dirty()
