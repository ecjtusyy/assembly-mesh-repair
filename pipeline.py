#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np

from mesh.io_obj import load_obj, save_obj
from ops.pipeline_impl import repair_assembly_mesh


def main():
    # 构造命令行参数解析器
    parser = argparse.ArgumentParser(
        description="装配网格非流形修复（Octree + 三角-三角 + T修复 + 拼接 + 流形化）"
    )
    # 支持输入多个 OBJ 文件（装配中的各个零件）
    parser.add_argument("--input", nargs="+", required=True, help="一个或多个 OBJ 文件路径（装配的各个零件）")
    # 输出目录
    parser.add_argument("--output_dir", default="out", help="输出目录，用于保存修复后的 OBJ")
    # 下面三个容差是“相对包围盒对角线”的比例（见 pipeline_impl.normalize_eps）
    parser.add_argument("--eps_v", type=float, default=1e-6, help="顶点焊接的容差（相对 bbox 对角线）")
    parser.add_argument("--eps_e", type=float, default=1e-6, help="边对齐 / T 形连接修复容差")
    parser.add_argument("--eps_p", type=float, default=1e-8, help="相交/共面近似判定容差")
    args = parser.parse_args()

    # 1) 读取所有零件的网格
    meshes = []
    names = []
    for p in args.input:
        V, F = load_obj(p)
        if V.size == 0 or F.size == 0:
            raise RuntimeError(f"文件 {p} 中没有有效几何数据（顶点或三角形为空）")
        meshes.append({"V": V.astype(np.float64), "F": F.astype(np.int32)})
        names.append(os.path.splitext(os.path.basename(p))[0])

    # 2) 调用修复管线（返回与输入对应的列表）
    repaired = repair_assembly_mesh(meshes, eps_v=args.eps_v, eps_e=args.eps_e, eps_p=args.eps_p)

    # 3) 保存结果（逐个零件保存 *_repaired.obj）
    os.makedirs(args.output_dir, exist_ok=True)
    for i, m in enumerate(repaired):
        out_path = os.path.join(args.output_dir, f"{names[i]}_repaired.obj")
        save_obj(out_path, m["V"], m["F"])
        print(f"[OK] 已保存: {out_path}")

    print("[OK] 全部零件处理完成。")


if __name__ == "__main__":
    main()
