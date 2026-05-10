# -*- coding: utf-8 -*-
"""OBJ 文件读写工具。

实现的功能：
- 读取顶点 v
- 读取面 f
- 支持 OBJ 常见索引格式：i、i/j、i/j/k、i//k
- 支持负索引
- 多边形面用扇形方式拆成三角形
- 写出时只保存 v/f
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


def _empty_vertices() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float64)


def _empty_faces() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.int64)


def _loc(path: Path, line_no: int) -> str:
    return f"{path}:{line_no}"


def _strip_comment(line: str) -> str:
    """去掉 OBJ 行内注释。"""
    return line.split("#", 1)[0].strip()


def _parse_face_index(token: str, vertex_count: int, *, path: Path, line_no: int) -> int:
    """把 OBJ 面索引转换成 Python 的 0-based 顶点索引。"""
    head = token.split("/", 1)[0]
    if not head:
        raise ValueError(f"面索引为空: {_loc(path, line_no)}: {token!r}")

    try:
        raw_idx = int(head)
    except ValueError as exc:
        raise ValueError(f"面索引不是整数: {_loc(path, line_no)}: {token!r}") from exc

    if raw_idx == 0:
        raise ValueError(f"OBJ 不允许使用顶点索引 0: {_loc(path, line_no)}: {token!r}")

    # OBJ 正索引从 1 开始；负索引表示从当前已有顶点末尾倒数。
    idx = raw_idx - 1 if raw_idx > 0 else vertex_count + raw_idx

    if idx < 0 or idx >= vertex_count:
        raise ValueError(
            f"面索引越界: {_loc(path, line_no)}: "
            f"{token!r}, 当前顶点数={vertex_count}"
        )

    return idx


def _normalize_vertices(V: np.ndarray) -> np.ndarray:
    """检查并统一顶点数组形状。"""
    V = np.asarray(V, dtype=np.float64)

    if V.size == 0:
        return _empty_vertices()

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V 必须是形状为 (N, 3) 的数组，当前是 {V.shape}")

    if not np.all(np.isfinite(V)):
        raise ValueError("V 中存在 NaN 或 Inf")

    return V


def _normalize_faces(F: np.ndarray, vertex_count: int) -> np.ndarray:
    """检查并统一三角面数组形状。"""
    F = np.asarray(F, dtype=np.int64)

    if F.size == 0:
        return _empty_faces()

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F 必须是形状为 (M, 3) 的数组，当前是 {F.shape}")

    min_idx = int(F.min())
    max_idx = int(F.max())

    if min_idx < 0 or max_idx >= vertex_count:
        raise ValueError(
            f"F 中存在越界顶点索引: "
            f"min={min_idx}, max={max_idx}, 顶点数={vertex_count}"
        )

    return F


def load_obj(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """读取 OBJ 文件，返回顶点数组 V 和三角面数组 F。

    V 的形状是 (N, 3)，类型为 float64。
    F 的形状是 (M, 3)，类型为 int64，索引从 0 开始。
    """
    path = Path(path)

    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = _strip_comment(raw_line)
            if not line:
                continue

            parts = line.split()
            record = parts[0]

            if record == "v":
                if len(parts) < 4:
                    raise ValueError(f"顶点行不完整: {_loc(path, line_no)}: {raw_line.rstrip()}")

                try:
                    vertices.append([
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ])
                except ValueError as exc:
                    raise ValueError(
                        f"顶点坐标不是有效数字: {_loc(path, line_no)}: {raw_line.rstrip()}"
                    ) from exc

            elif record == "f":
                if len(parts) < 4:
                    raise ValueError(f"面的顶点数少于 3 个: {_loc(path, line_no)}")

                ids = [
                    _parse_face_index(tok, len(vertices), path=path, line_no=line_no)
                    for tok in parts[1:]
                ]

                # 多边形面用简单扇形剖分拆成三角形。
                for i in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[i], ids[i + 1]])

            else:
                # 当前修复流程只需要 v/f，其它 OBJ 记录暂时忽略。
                continue

    V = _empty_vertices() if not vertices else np.asarray(vertices, dtype=np.float64)
    F = _empty_faces() if not faces else np.asarray(faces, dtype=np.int64)

    return V, F


def save_obj(path: PathLike, V: np.ndarray, F: np.ndarray) -> None:
    """保存最小 OBJ 文件，只写出 v 和三角形 f。"""
    path = Path(path)

    V = _normalize_vertices(V)
    F = _normalize_faces(F, vertex_count=V.shape[0])

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for x, y, z in V:
            f.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")

        for a, b, c in F:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def read_obj(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """load_obj 的兼容别名。"""
    return load_obj(path)


def write_obj(path: PathLike, V: np.ndarray, F: np.ndarray) -> None:
    """save_obj 的兼容别名。"""
    save_obj(path, V, F)