from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """Triangle mesh backed by a float32 array shaped [triangle, vertex, xyz]."""

    triangles: np.ndarray

    def __post_init__(self) -> None:
        triangles = np.asarray(self.triangles, dtype=np.float32)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("mesh triangles must have shape [n, 3, 3]")
        object.__setattr__(self, "triangles", triangles)

    @property
    def z_min(self) -> float:
        return float(np.min(self.triangles[:, :, 2]))

    @property
    def z_max(self) -> float:
        return float(np.max(self.triangles[:, :, 2]))


def load_stl(path: str | Path) -> Mesh:
    """Load binary or ASCII STL without requiring third-party mesh packages."""

    stl_path = Path(path)
    if not stl_path.exists():
        raise FileNotFoundError(stl_path)

    data = stl_path.read_bytes()
    if _looks_like_binary_stl(data):
        return _load_binary_stl(data)
    return _load_ascii_stl(data.decode("utf-8", errors="ignore"))


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    return expected_size == len(data)


def _load_binary_stl(data: bytes) -> Mesh:
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    triangles = np.empty((triangle_count, 3, 3), dtype=np.float32)
    offset = 84
    for index in range(triangle_count):
        # normal: 3 float32, vertices: 9 float32, attr byte count: uint16
        values = struct.unpack_from("<12fH", data, offset)
        triangles[index] = np.array(values[3:12], dtype=np.float32).reshape(3, 3)
        offset += 50
    return Mesh(triangles)


def _load_ascii_stl(text: str) -> Mesh:
    vertices: list[list[float]] = []
    triangles: list[list[list[float]]] = []
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(vertices) == 3:
                triangles.append(vertices)
                vertices = []

    if not triangles:
        raise ValueError("STL file contains no triangles")
    return Mesh(np.asarray(triangles, dtype=np.float32))

