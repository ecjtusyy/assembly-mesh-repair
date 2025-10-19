# -*- coding: utf-8 -*-
"""
一个用于验证修复管线是否正确运行的极简演示脚本
----------------------------------------------------------
功能说明：
1) 自动生成两个存在非流形问题的小测试网格（包含 T 形连接和轻微重叠）；
2) 将它们保存为 OBJ 文件；
3) 调用 pipeline.py 进行修复；
4) 加载修复后的结果并进行简单的拓扑检查。

运行方式：
    python demo_verify_framework.py
"""

import sys
import os
import subprocess
import numpy as np

# ------------------------------------------------------------
# 让 Python 能找到项目内的模块（mesh、geom、ops 等）
# ------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# ------------------------------------------------------------
# 导入项目中的 Mesh 类与 OBJ 文件读写函数
# ------------------------------------------------------------
from mesh.mesh import Mesh
from mesh.io_obj import write_obj, read_obj  # 若函数名不同，可在此修改


# ------------------------------------------------------------
# 构造两个测试网格
# ------------------------------------------------------------
def make_part_A():
    """
    构造第一个测试零件 Part A：
    - 是一个单位方形（XY 平面上的 1x1 正方形）；
    - 上边中点多出一个顶点，形成潜在的 T-junction。
    """
    # 顶点坐标（共 5 个）
    V = np.array([
        [0.0, 0.0, 0.0],  # 顶点0：左下角
        [1.0, 0.0, 0.0],  # 顶点1：右下角
        [1.0, 1.0, 0.0],  # 顶点2：右上角
        [0.0, 1.0, 0.0],  # 顶点3：左上角
        [0.5, 1.0, 0.0],  # 顶点4：上边中点（制造T点）
    ], dtype=float)

    # 面（三角形）索引
    F = np.array([
        [0, 1, 3],  # 下方三角
        [1, 2, 3],  # 右方三角
        [3, 4, 2],  # 额外小三角（制造 T-junction）
    ], dtype=int)

    return Mesh(V, F)


def make_part_B():
    """
    构造第二个测试零件 Part B：
    - 位于 Part A 上方；
    - 与 A 的上边有轻微重叠；
    - 该重叠会导致“跨件拼接”问题。
    """
    V = np.array([
        [0.25, 1.0, 0.0],  # 顶点0：与A上边部分重叠
        [1.25, 1.0, 0.0],  # 顶点1：稍微超出A右侧
        [1.25, 1.5, 0.0],  # 顶点2：上层右角
        [0.25, 1.5, 0.0],  # 顶点3：上层左角
    ], dtype=float)

    F = np.array([
        [0, 1, 3],  # 下方三角
        [1, 2, 3],  # 上方三角
    ], dtype=int)

    return Mesh(V, F)


# ------------------------------------------------------------
# 一些基础检查函数（判断非流形边和重复顶点）
# ------------------------------------------------------------
def count_nonmanifold_edges(F):
    """
    统计网格中“非流形边”的数量。
    原理：
    - 每条边若被超过两个面共享，则是非流形边。
    """
    from collections import defaultdict
    edge_count = defaultdict(int)

    # 规范化边（确保顺序一致）
    def norm_edge(a, b):
        return (a, b) if a < b else (b, a)

    # 遍历所有三角面，统计每条边的出现次数
    for a, b, c in F:
        edge_count[norm_edge(a, b)] += 1
        edge_count[norm_edge(b, c)] += 1
        edge_count[norm_edge(c, a)] += 1

    # 找出被超过两面共享的边
    nonmanifold = [e for e, cnt in edge_count.items() if cnt > 2]
    return len(nonmanifold), edge_count


def count_close_duplicates(V, eps=1e-9):
    """
    统计非常接近的重复顶点对（欧氏距离小于 eps）。
    O(N^2) 实现，仅适合小模型。
    """
    n = V.shape[0]
    dup = 0
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(V[i] - V[j]) <= eps:
                dup += 1
    return dup


def print_stats(tag, mesh_or_tuple):
    """
    打印网格的基本统计信息：
    - 顶点数
    - 面数
    - 非流形边数量
    - 接近重复的顶点对数
    """
    if isinstance(mesh_or_tuple, Mesh):
        V, F = mesh_or_tuple.V, mesh_or_tuple.F
    else:
        V, F = mesh_or_tuple
    nm_cnt, _ = count_nonmanifold_edges(F)
    dup_cnt = count_close_duplicates(V, eps=1e-12)
    print(f"[{tag}] 顶点数={V.shape[0]}, 面数={F.shape[0]}, 非流形边={nm_cnt}, 近重复顶点对={dup_cnt}")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    # 创建输出目录
    os.makedirs("demo_out", exist_ok=True)

    # 定义输入输出路径
    inA = "demo_out/partA_demo.obj"
    inB = "demo_out/partB_demo.obj"
    out_dir = "demo_out"

    # 1) 生成两个测试网格
    A = make_part_A()
    B = make_part_B()

    # 2) 保存为 OBJ 文件
    write_obj(inA, A.V, A.F)
    write_obj(inB, B.V, B.F)

    # 输出修复前统计信息
    print_stats("修复前 PartA", A)
    print_stats("修复前 PartB", B)

    # 3) 调用 pipeline.py 执行修复
    cmd = [
        "python", "pipeline.py",
        "--input", inA, inB,
        "--output_dir", out_dir,
        "--eps_v", "1e-6",
        "--eps_e", "1e-6",
        "--eps_p", "1e-7",
    ]

    print("\n[运行命令] ", " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=True, text=True)

    # 打印 pipeline 输出
    print("[pipeline 标准输出]")
    print(ret.stdout)
    print("[pipeline 错误输出]")
    print(ret.stderr)

    # 如果返回码非 0，说明运行出错
    if ret.returncode != 0:
        print("❌ pipeline 执行失败，请查看错误信息。")
        return

    # 4) 读取修复后的结果（默认命名 *_repaired.obj）
    outA = "demo_out/partA_demo_repaired.obj"
    outB = "demo_out/partB_demo_repaired.obj"

    # 读取 OBJ 文件（只含顶点与三角形面）
    V_A, F_A = read_obj(outA)
    V_B, F_B = read_obj(outB)

    # 输出修复后统计信息
    print_stats("修复后 PartA", (V_A, F_A))
    print_stats("修复后 PartB", (V_B, F_B))

    # 5) 简单检查结果正确性
    nmA, _ = count_nonmanifold_edges(F_A)
    nmB, _ = count_nonmanifold_edges(F_B)

    # 检查是否还有非流形边
    if nmA == 0 and nmB == 0:
        print("\n✅ 修复后没有检测到非流形边。")
    else:
        print(f"\n⚠️ 修复后仍存在非流形边 (A:{nmA}, B:{nmB})")

    # 检查是否还有非常接近的重复顶点
    dupA = count_close_duplicates(V_A, eps=1e-12)
    dupB = count_close_duplicates(V_B, eps=1e-12)
    if dupA == 0 and dupB == 0:
        print("✅ 焊接后没有极其接近的重复顶点。")
    else:
        print(f"⚠️ 仍有重复顶点 (A:{dupA}, B:{dupB})，可能需要进一步焊接清理。")



# ------------------------------------------------------------
# 程序入口
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
