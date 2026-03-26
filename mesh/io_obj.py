# -*- coding: utf-8 -*-
"""OBJ I/O for the first bridge prototype.

Scope of the first version:
- reads only vertex positions (v) and faces (f)
- ignores normals / UVs / materials / groups
- triangulates polygon faces using simple fan triangulation
- writes only v/f OBJ
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _parse_face_index(token: str, vertex_count: int) -> int:
    """Parse an OBJ face vertex token and return a zero-based vertex index.

    Supports forms such as ``i``, ``i/j``, ``i/j/k``, ``i//k`` and negative indices.
    """
    head = token.split("/")[0]
    idx = int(head)
    if idx > 0:
        return idx - 1
    if idx < 0:
        return vertex_count + idx
    raise ValueError(f"OBJ vertex index 0 is invalid: {token!r}")



def load_obj(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load an OBJ file and return ``(V, F)``.

    ``V`` is ``(N, 3)`` float64 and ``F`` is ``(M, 3)`` int64.
    """
    vs: List[List[float]] = []
    faces: List[List[int]] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            head = parts[0]
            if head == "v":
                if len(parts) < 4:
                    raise ValueError(f"Invalid vertex line at {path}:{line_no}: {line.rstrip()}" )
                vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif head == "f":
                if len(parts) < 4:
                    raise ValueError(f"Face with fewer than 3 vertices at {path}:{line_no}")
                ids = [_parse_face_index(tok, len(vs)) for tok in parts[1:]]
                if len(ids) == 3:
                    faces.append(ids)
                else:
                    for i in range(1, len(ids) - 1):
                        faces.append([ids[0], ids[i], ids[i + 1]])
            else:
                # First version intentionally ignores all other record types.
                continue

    V = np.asarray(vs, dtype=np.float64)
    if V.size == 0:
        V = np.zeros((0, 3), dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.size == 0:
        F = np.zeros((0, 3), dtype=np.int64)
    return V, F



def save_obj(path: str | Path, V: np.ndarray, F: np.ndarray) -> None:
    """Write an OBJ file containing only vertex positions and triangle faces."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in np.asarray(V, dtype=np.float64):
            f.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for tri in np.asarray(F, dtype=np.int64):
            a, b, c = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
            f.write(f"f {a} {b} {c}\n")



def read_obj(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    return load_obj(path)



def write_obj(path: str | Path, V: np.ndarray, F: np.ndarray) -> None:
    save_obj(path, V, F)
