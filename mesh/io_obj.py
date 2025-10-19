# -*- coding: utf-8 -*-
"""
极简 OBJ 读写（只处理三角形面 'f'，忽略纹理/法线等其他元素）。
若读到多边形面，会自动做“扇形三角化”。
"""

import numpy as np


def load_obj(path: str):
    """读取一个（已经三角化或可扇形三角化的）OBJ 文件，返回 (V, F)。"""
    vs = []
    faces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()
            if tok[0] == "v" and len(tok) >= 4:
                vs.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == "f":
                # 面的每个顶点可能是 "idx" 或 "idx/..." 形式，这里只取第一个
                ids = []
                for it in tok[1:]:
                    a = it.split("/")[0]
                    ids.append(int(a) - 1)  # OBJ 索引从 1 开始，这里改为 0 基
                if len(ids) == 3:
                    faces.append(ids)
                else:
                    # 多边形 -> 扇形三角化
                    for i in range(1, len(ids) - 1):
                        faces.append([ids[0], ids[i], ids[i + 1]])
    V = np.array(vs, dtype=np.float64) if vs else np.zeros((0, 3))
    F = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    return V, F


def save_obj(path: str, V: np.ndarray, F: np.ndarray):
    """写出 OBJ（顶点 v、三角形 f）。"""
    with open(path, "w", encoding="utf-8") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for a, b, c in F:
            f.write(f"f {a+1} {b+1} {c+1}\n")  # 写回 1 基索引

def read_obj(path: str):
    return load_obj(path)

def write_obj(path: str, V: np.ndarray, F: np.ndarray):
    return save_obj(path, V, F)
