# -*- coding: utf-8 -*-
"""
轻量级网格质量改进：固定边界的拉普拉斯平滑（Laplacian Smoothing）。

说明
----
- 仅对“内部顶点”做平滑，尽量避免改变边界形状（边界点保持不动）。
- 边界判定采用简单启发式：统计顶点在三角面与相邻边中的出现次数来近似判断。
- 工程化项目可替换为更鲁棒的边界检测与加权平滑（如 cotangent 权重）。
"""

from typing import List
import numpy as np


def is_boundary_vertex(F: np.ndarray, vid: int) -> bool:
    """
    判断一个顶点是否是“边界顶点”的近似方法。

    参数
    ----
    F : (M, 3) 的整型数组
        三角面索引，每行代表一个三角形的三个顶点编号。
    vid : int
        需要判断的顶点编号。

    返回
    ----
    bool
        True 表示该顶点被视为边界顶点；False 表示内部顶点。

    实现思路（启发式，非严格）：
    - 遍历所有三角面，统计该顶点被包含的次数（deg）。
    - 同时统计与该顶点相关的“边”的出现次数（deg2）。
    - 若 deg 很小（如 ≤2）或 deg2 过小，说明顶点邻接不充分，倾向认为它在边界上。
    """
    # deg 记录该顶点出现在多少个三角面中
    deg = 0
    # deg2 粗略记录与该顶点相关的边的出现次数
    deg2 = 0

    # 遍历所有三角面，做简单计数
    for a, b, c in F:
        # 如果该面包含 vid，则 deg + 1
        if vid in (a, b, c):
            deg += 1

            # 该面的三条边
            edges = [(a, b), (b, c), (c, a)]

            # 如果这条边与 vid 相邻，则 deg2 + 1
            for e in edges:
                if vid in e:
                    deg2 += 1

    # 这是一个“很粗”的边界判断：
    # - 出现于很少面片中的点（deg 小）通常处在网格边缘；
    # - 与该点相关的边计数太少（deg2 小）也提示它可能在边界。
    # 注意：此规则不适用于所有情况，仅作为教学示例。
    return (deg <= 2) or (deg2 < 6)


def laplacian_smooth(mesh, iterations: int = 3, lam: float = 0.3) -> None:
    """
    固定边界的拉普拉斯平滑：仅对内部顶点做邻域均值拉回。

    参数
    ----
    mesh : Mesh
        需要提供 .V (N x 3 浮点顶点坐标)、.F (M x 3 三角面索引) 以及 mesh.mark_dirty() 方法。
    iterations : int, 默认 3
        平滑迭代次数。次数越多，表面越“软”，但可能过度收缩。
    lam : float, 默认 0.3
        拉回系数（学习率）。范围通常在 (0, 1)。
        - 越大：形变更剧烈，收缩更明显；
        - 越小：变化更平滑，收敛更慢。

    实现步骤
    --------
    1) 预构建每个顶点的“一环邻域”列表（去重）。
    2) 进行多次迭代：对每个“内部顶点”，把它往邻居的平均位置移动一点点。
    3) 边界点保持不动（以减少轮廓漂移）。
    """
    # 复制一份顶点坐标，避免直接覆盖导致邻接在同一轮中相互污染
    V = mesh.V.copy()
    F = mesh.F
    N = V.shape[0]

    # ---------- 构建邻接表：nbrs[i] = 第 i 个点的一环邻居列表 ----------
    neighbors: List[List[int]] = [[] for _ in range(N)]
    for a, b, c in F:
        # 三角形 (a, b, c) 互为邻居
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])

    # 去重，避免重复邻居影响均值
    neighbors = [list(set(one_ring)) for one_ring in neighbors]

    # ---------- 迭代平滑 ----------
    for _ in range(iterations):
        # V_new 用于存放本轮的更新结果
        V_new = V.copy()

        # 逐点更新
        for i in range(N):
            # 如果是边界点，或者没有邻居（异常情况），就跳过
            if is_boundary_vertex(F, i) or len(neighbors[i]) == 0:
                continue

            # 计算邻居顶点的坐标均值
            mean_pos = np.mean(V[neighbors[i]], axis=0)

            # 用简单的 (1 - lam) * 自己 + lam * 邻居均值 进行拉回
            V_new[i] = (1.0 - lam) * V[i] + lam * mean_pos

        # 本轮结束，更新 V
        V = V_new

    # ---------- 写回网格并标记失效 ----------
    mesh.V = V
    mesh.mark_dirty()
