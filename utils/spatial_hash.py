# -*- coding: utf-8 -*-
"""
Simple spatial hash for near-neighbor queries (used by vertex weld).
"""

import numpy as np
from collections import defaultdict


class SpatialHash:
    def __init__(self, cell: float):
        self.cell = max(float(cell), 1e-18)
        self.buckets = defaultdict(list)

    def _key(self, p):
        q = np.floor(p / self.cell).astype(int)
        return (q[0], q[1], q[2])

    def insert(self, idx: int, p):
        self.buckets[self._key(p)].append(idx)

    def query_ball(self, p, radius: float):
        """Return indices near p within radius (checks 3x3x3 neighborhoods)."""
        r = int(max(1, np.ceil(radius / self.cell)))
        q = np.floor(p / self.cell).astype(int)
        out = []
        for i in range(q[0] - r, q[0] + r + 1):
            for j in range(q[1] - r, q[1] + r + 1):
                for k in range(q[2] - r, q[2] + r + 1):
                    out.extend(self.buckets.get((i, j, k), []))
        return out
