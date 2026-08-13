from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def _read_stl(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            vertices: list[tuple[float, float, float]] = []
            normals: list[tuple[float, float, float]] = []
            for index in range(triangle_count):
                values = struct.unpack_from("<12fH", data, 84 + index * 50)
                normal = tuple(float(value) for value in values[:3])
                normals.extend([normal, normal, normal])
                for vertex_index in range(3):
                    start = 3 + vertex_index * 3
                    vertices.append(tuple(float(value) for value in values[start : start + 3]))
            return vertices, normals

    vertices = []
    face_normals: list[tuple[float, float, float]] = []
    active_normal = (0.0, 0.0, 1.0)
    for raw_line in data.decode("ascii", errors="ignore").splitlines():
        parts = raw_line.split()
        if len(parts) == 5 and parts[:2] == ["facet", "normal"]:
            active_normal = tuple(float(value) for value in parts[2:5])
        elif len(parts) == 4 and parts[0] == "vertex":
            vertices.append(tuple(float(value) for value in parts[1:4]))
            face_normals.append(active_normal)
    if not vertices or len(vertices) % 3:
        raise ValueError(f"invalid STL triangle stream: {path}")
    return vertices, face_normals


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector) if length > 1e-12 else (0.0, 0.0, 1.0)


def _cad_to_tool(
    point: tuple[float, float, float], tcp: tuple[float, float, float]
) -> tuple[float, float, float]:
    # The nozzle points along CAD -Y.  The project convention uses +X_TOOL
    # as the work/nozzle axis, with +Y_TOOL across CAD +Z and +Z_TOOL across
    # CAD -X.  This is a right-handed frame whose origin is the nozzle outlet.
    dx, dy, dz = (point[index] - tcp[index] for index in range(3))
    return (-dy, dz, -dx)


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((-len(data)) % 4)


def _write_glb(
    path: Path,
    positions_mm: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]],
    metadata: dict[str, object],
) -> None:
    positions_m = [tuple(value / 1000.0 for value in point) for point in positions_mm]
    position_bytes = b"".join(struct.pack("<3f", *point) for point in positions_m)
    normal_bytes = b"".join(struct.pack("<3f", *normal) for normal in normals)
    binary = _pad4(position_bytes) + _pad4(normal_bytes)
    position_min = [min(point[axis] for point in positions_m) for axis in range(3)]
    position_max = [max(point[axis] for point in positions_m) for axis in range(3)]
    document = {
        "asset": {"version": "2.0", "generator": "kuka_slicer build_printhead_asset.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "printhead_tcp_frame"}],
        "meshes": [
            {
                "name": "printhead_interference_check",
                "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "material": 0}],
                "extras": metadata,
            }
        ],
        "materials": [
            {
                "name": "machined_aluminum",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.54, 0.61, 0.66, 1.0],
                    "metallicFactor": 0.72,
                    "roughnessFactor": 0.36,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {
                "buffer": 0,
                "byteOffset": len(_pad4(position_bytes)),
                "byteLength": len(normal_bytes),
                "target": 34962,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions_m),
                "type": "VEC3",
                "min": position_min,
                "max": position_max,
            },
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
        ],
    }
    json_chunk = _pad4(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), b" ")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    glb = b"".join(
        [
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary), b"BIN\x00"),
            binary,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb)


def _mesh_components(
    positions: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], list[list[int]], list[list[int]]]:
    """Return deduplicated points, triangles, and connected triangle groups.

    The group triangle indexes refer to the exported preview mesh.  They retain
    the real CAD-derived triangle surface for offline collision checks; bounds
    are merely descriptive and are never collision proxies.
    """
    unique_positions: list[tuple[float, float, float]] = []
    position_indices: dict[tuple[float, float, float], int] = {}
    triangle_indices: list[int] = []
    for point in positions:
        key = tuple(round(value, 7) for value in point)
        index = position_indices.get(key)
        if index is None:
            index = len(unique_positions)
            position_indices[key] = index
            unique_positions.append(key)
        triangle_indices.append(index)

    parent = list(range(len(unique_positions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index in range(0, len(triangle_indices), 3):
        first, second, third = triangle_indices[index : index + 3]
        union(first, second)
        union(second, third)

    groups: dict[int, list[int]] = {}
    for triangle_index in range(len(triangle_indices) // 3):
        root = find(triangle_indices[triangle_index * 3])
        groups.setdefault(root, []).append(triangle_index)
    return (
        unique_positions,
        [triangle_indices[index : index + 3] for index in range(0, len(triangle_indices), 3)],
        list(groups.values()),
    )


def _component_bounds(
    points: list[tuple[float, float, float]],
) -> tuple[list[float], list[float]]:
    return (
        [min(point[axis] for point in points) for axis in range(3)],
        [max(point[axis] for point in points) for axis in range(3)],
    )


def _model_components(
    positions: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], list[list[int]], list[dict[str, object]]]:
    unique_positions, triangles, groups = _mesh_components(positions)
    components: list[dict[str, object]] = []
    for triangle_indexes in groups:
        point_indexes = {index for triangle_index in triangle_indexes for index in triangles[triangle_index]}
        points = [unique_positions[index] for index in point_indexes]
        minimum, maximum = _component_bounds(points)
        components.append({"minimum": minimum, "maximum": maximum, "triangle_indices": triangle_indexes})

    # The upper housing is represented by adjacent STL solids.  Merge it only
    # as a display component; the heater block remains a distinct exact mesh.
    merged: list[dict[str, object]] = []
    for component in sorted(components, key=lambda item: item["minimum"][0]):
        minimum = component["minimum"]
        maximum = component["maximum"]
        for existing in merged:
            merged_minimum = existing["minimum"]
            merged_maximum = existing["maximum"]
            same_xy = all(
                abs(left - right) <= 1e-5
                for left, right in zip(
                    (minimum[0], maximum[0], minimum[1], maximum[1]),
                    (
                        merged_minimum[0],
                        merged_maximum[0],
                        merged_minimum[1],
                        merged_maximum[1],
                    ),
                )
            )
            touches_z = minimum[2] <= merged_maximum[2] + 1e-5 and maximum[2] >= merged_minimum[2] - 1e-5
            if same_xy and touches_z:
                merged_minimum[2] = min(merged_minimum[2], minimum[2])
                merged_maximum[2] = max(merged_maximum[2], maximum[2])
                existing["triangle_indices"].extend(component["triangle_indices"])
                break
        else:
            merged.append({
                "minimum": minimum[:], "maximum": maximum[:],
                "triangle_indices": list(component["triangle_indices"]),
            })

    if len(merged) != 3:
        raise ValueError(f"expected upper housing, heater block, and nozzle; found {len(merged)} components")
    # Export order in an STL is not a physical-component contract.  In the
    # replacement assembly the long nozzle/heat-break sorts ahead of the
    # compact heater block along X, so identify the components by their
    # physical transverse extent instead.  The housing is the broadest
    # cross-section; the heater is the remaining component with the broadest
    # minimum transverse span; the third component is the nozzle/heat-break.
    def transverse_spans(component: dict[str, object]) -> tuple[float, float]:
        minimum = component["minimum"]
        maximum = component["maximum"]
        return float(maximum[1] - minimum[1]), float(maximum[2] - minimum[2])

    upper_housing = max(merged, key=lambda component: math.prod(transverse_spans(component)))
    remaining = [component for component in merged if component is not upper_housing]
    heater_block = max(remaining, key=lambda component: min(transverse_spans(component)))
    nozzle = next(component for component in remaining if component is not heater_block)
    named_components = (
        ("upper_housing", upper_housing),
        ("heater_block", heater_block),
        ("nozzle", nozzle),
    )
    exported: list[dict[str, object]] = []
    for name, component in named_components:
        exported.append({
            "name": name,
            "local_bounds_mm": {
                "minimum_mm": [round(value, 7) for value in component["minimum"]],
                "maximum_mm": [round(value, 7) for value in component["maximum"]],
            },
            "triangle_indices": component["triangle_indices"],
            "triangle_count": len(component["triangle_indices"]),
        })
    return unique_positions, triangles, exported


def build_asset(stl_path: Path, output_dir: Path, source_cad: Path | None) -> dict[str, object]:
    cad_vertices, cad_normals = _read_stl(stl_path)
    minimum = [min(point[axis] for point in cad_vertices) for axis in range(3)]
    maximum = [max(point[axis] for point in cad_vertices) for axis in range(3)]
    tolerance = max(maximum[1] - minimum[1], 1.0) * 1e-5
    outlet_vertices = {
        point for point in cad_vertices if abs(point[1] - minimum[1]) <= tolerance
    }
    if not outlet_vertices:
        raise ValueError("cannot locate the minimum-Y nozzle outlet plane")
    tcp = (
        sum(point[0] for point in outlet_vertices) / len(outlet_vertices),
        minimum[1],
        sum(point[2] for point in outlet_vertices) / len(outlet_vertices),
    )
    tool_positions = [_cad_to_tool(point, tcp) for point in cad_vertices]
    tool_normals = [_normalize((-normal[1], normal[2], -normal[0])) for normal in cad_normals]
    box_min = [min(point[axis] for point in tool_positions) for axis in range(3)]
    box_max = [max(point[axis] for point in tool_positions) for axis in range(3)]
    box_size = [box_max[axis] - box_min[axis] for axis in range(3)]
    source_hash = hashlib.sha256(stl_path.read_bytes()).hexdigest()
    cad_hash = hashlib.sha256(source_cad.read_bytes()).hexdigest() if source_cad and source_cad.is_file() else None
    metadata: dict[str, object] = {
        "name": "喷头模块-干涉检查",
        "source_stl_sha256": source_hash,
        "source_cad_sha256": cad_hash,
        "source_units": "millimeter",
        "glb_units": "meter",
        "tcp_cad_mm": [round(value, 7) for value in tcp],
        "tcp_definition": "center of the minimum-Y nozzle outlet plane",
        "tool_axis": "+X_TOOL = CAD -Y (nozzle points downward in the calibrated flat pose)",
        "cad_to_tool_axes": {"x_tool": "-y_cad", "y_tool": "+z_cad", "z_tool": "-x_cad"},
        "model_bounds": {
            "type": "display_extent_only",
            "minimum_mm": [round(value, 7) for value in box_min],
            "maximum_mm": [round(value, 7) for value in box_max],
            "size_mm": [round(value, 7) for value in box_size],
        },
        "triangle_count": len(tool_positions) // 3,
    }
    # Keep this source line ASCII-only so Windows code-page handling cannot
    # corrupt the Chinese asset label.
    metadata["name"] = "\u55b7\u5934\u6a21\u5757-\u5e72\u6d89\u68c0\u67e5"
    unique_positions, triangle_indices, model_components = _model_components(tool_positions)
    metadata["model_components"] = model_components

    glb_path = output_dir / "printhead_interference_check.glb"
    _write_glb(glb_path, tool_positions, tool_normals, metadata)

    preview = {
        "format": "kuka_printhead_preview_v3",
        "units": "millimeter",
        "positions": unique_positions,
        "triangles": triangle_indices,
        "model_bounds": metadata["model_bounds"],
        "model_components": model_components,
        "metadata": metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "printhead_interference_check.preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (output_dir / "printhead_interference_check.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TCP-relative printhead GLB and preview mesh")
    parser.add_argument("stl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-cad", type=Path)
    args = parser.parse_args()
    metadata = build_asset(args.stl, args.output_dir, args.source_cad)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
