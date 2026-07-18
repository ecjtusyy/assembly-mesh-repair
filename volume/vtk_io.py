"""无需额外依赖的 VTK legacy 四面体输出。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from volume.tetra_mesh import TetraMesh


def write_quality_vtk(path: str | Path, mesh: TetraMesh, quality: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# vtk DataFile Version 3.0",
        "assembly-mesh-repair tetra quality",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {len(mesh.V)} double",
    ]
    lines.extend("{:.17g} {:.17g} {:.17g}".format(*point) for point in mesh.V)
    lines.append(f"CELLS {len(mesh.T)} {len(mesh.T) * 5}")
    lines.extend("4 {} {} {} {}".format(*tet) for tet in mesh.T)
    lines.append(f"CELL_TYPES {len(mesh.T)}")
    lines.extend(["10"] * len(mesh.T))
    lines.extend(
        [
            f"CELL_DATA {len(mesh.T)}",
            "SCALARS mean_ratio double 1",
            "LOOKUP_TABLE default",
        ]
    )
    lines.extend("{:.17g}".format(float(value)) for value in quality)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
