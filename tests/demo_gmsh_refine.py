# -*- coding: utf-8 -*-
"""
最小 demo：用 gmsh / meshio 对已有 OBJ 三角面片做网格加密。

路线：

    OBJ
      -> meshio 写成 STL
      -> gmsh 直接 refine 已有三角网格
      -> 写 refined.msh
      -> meshio 抽取 triangle
      -> 写 refined.obj
"""

from pathlib import Path

import gmsh
import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_OBJ = ROOT / "tests/data/advanced_assembly_case.obj"

OUT_DIR = ROOT / "tests/out/gmsh_refine_demo"
TMP_STL = OUT_DIR / "input_tmp.stl"
OUT_MSH = OUT_DIR / "refined.msh"
OUT_OBJ = OUT_DIR / "refined.obj"

# 细分次数：1 次约等于每个三角形变 4 个；2 次约等于变 16 个
REFINE_LEVELS = 2


def get_triangles(mesh):
    triangles = []

    for cell in mesh.cells:
        if cell.type == "triangle":
            triangles.append(cell.data)

    if not triangles:
        raise RuntimeError("没有找到 triangle 三角面片")

    return np.vstack(triangles).astype(np.int64)


def obj_to_stl(obj_path: Path, stl_path: Path):
    mesh = meshio.read(obj_path)
    triangles = get_triangles(mesh)

    stl_mesh = meshio.Mesh(
        points=np.asarray(mesh.points[:, :3], dtype=np.float64),
        cells=[("triangle", triangles)],
    )

    meshio.write(stl_path, stl_mesh)

    print("    input vertices:", len(mesh.points))
    print("    input triangles:", len(triangles))


def gmsh_refine(stl_path: Path, msh_path: Path):
    gmsh.initialize()

    try:
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.merge(str(stl_path))

        # 关键：这里只做已有三角网格的细分。
        # 不做 classifySurfaces / createGeometry / generate。
        for i in range(REFINE_LEVELS):
            print(f"    refine level {i + 1}/{REFINE_LEVELS}")
            gmsh.model.mesh.refine()

        gmsh.write(str(msh_path))

    finally:
        gmsh.finalize()


def msh_to_obj(msh_path: Path, obj_path: Path):
    mesh = meshio.read(msh_path)
    triangles = get_triangles(mesh)

    points = np.asarray(mesh.points[:, :3], dtype=np.float64)

    # 只保留 triangle 真正用到的点，避免 OBJ 里带多余点
    used = np.unique(triangles.reshape(-1))

    old_to_new = -np.ones(points.shape[0], dtype=np.int64)
    old_to_new[used] = np.arange(len(used), dtype=np.int64)

    new_points = points[used]
    new_triangles = old_to_new[triangles]

    obj_mesh = meshio.Mesh(
        points=new_points,
        cells=[("triangle", new_triangles)],
    )

    meshio.write(obj_path, obj_mesh)

    print("    refined vertices:", len(new_points))
    print("    refined triangles:", len(new_triangles))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_OBJ.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_OBJ}")

    print("[1] OBJ -> STL")
    obj_to_stl(INPUT_OBJ, TMP_STL)

    print("[2] Gmsh refine existing triangles")
    gmsh_refine(TMP_STL, OUT_MSH)

    print("[3] MSH -> OBJ")
    msh_to_obj(OUT_MSH, OUT_OBJ)

    print()
    print("完成")
    print("输入 OBJ:", INPUT_OBJ)
    print("输出 MSH:", OUT_MSH)
    print("输出 OBJ:", OUT_OBJ)


if __name__ == "__main__":
    main()
