from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Sequence

import numpy as np


class PathType(str, Enum):
    FIBER_PRINT = "fiber_print"
    RESIN_PRINT = "resin_print"
    TRAVEL = "travel"
    TOOL_CHANGE_EVENT = "tool_change_event"
    EVENT = "event"


Point3 = tuple[float, float, float]
Pose6 = tuple[float, float, float, float, float, float]


_ROW_CONTINUITY_GAP_LIMIT = 1000


@dataclass
class PreviewPath:
    layer: int
    order_index: int
    path_type: PathType
    tool_id: int
    points: tuple[Point3, ...]
    poses: tuple[Pose6, ...]
    start: Point3
    end: Point3
    start_abc: Point3
    end_abc: Point3
    src_line_start: str
    src_line_end: str
    path_id: int = 0
    event_type: str = ""
    payload: str = ""


def list_preview_layers(npz_root: str | Path) -> list[int]:
    root = Path(npz_root).expanduser()
    layers = set()
    saw_modern_layer_field = False
    for path in _candidate_npz_files(root, layer=None):
        inferred_layer = _infer_layer_from_path(path)
        if inferred_layer is not None:
            layers.add(inferred_layer)
        try:
            with np.load(str(path)) as data:
                if "preview_layer_index" in data:
                    saw_modern_layer_field = True
                    layers.update(
                        int(v)
                        for v in np.unique(data["preview_layer_index"])
                    )
                elif "layer_index" in data:
                    saw_modern_layer_field = True
                    layers.update(
                        int(v)
                        for v in np.unique(data["layer_index"])
                    )
        except Exception:
            continue
    if saw_modern_layer_field:
        return sorted(layers)

    preview_layers = _flat_preview_layers(root)
    if preview_layers:
        return preview_layers

    physical_layers = _legacy_physical_layer_z_map(root)
    if physical_layers:
        return sorted(physical_layers)

    return sorted(layers)


def extract_layer_preview_paths(
    npz_root: str | Path,
    layer: int,
    max_paths: int | None = None,
    max_rows: int | None = None,
) -> list[PreviewPath]:
    root = Path(npz_root).expanduser()
    layer_z_map = _legacy_physical_layer_z_map(root)
    files = _candidate_npz_files(
        root,
        layer=layer,
        include_legacy_cross_layer=bool(layer_z_map),
    )
    if max_paths is not None or layer_z_map:
        paths: list[PreviewPath] = []
        for path in files:
            rows = _rows_from_npz(path, layer, root, layer_z_map, max_rows=max_rows)
            rows.sort(key=lambda row: row["seq"])
            remaining = None if max_paths is None else max_paths - len(paths)
            paths.extend(
                _extract_paths_from_rows(
                    rows,
                    int(layer),
                    max_paths=remaining,
                )
            )
            if max_paths is not None and len(paths) >= max_paths:
                break
        for index, preview_path in enumerate(paths):
            preview_path.order_index = index
        return paths

    rows = []
    for path in files:
        rows.extend(_rows_from_npz(path, layer, root, layer_z_map, max_rows=max_rows))
    rows.sort(key=lambda row: row["seq"])
    return _extract_paths_from_rows(rows, int(layer), max_paths=max_paths)


def _candidate_npz_files(
    root: Path,
    layer: int | None,
    include_legacy_cross_layer: bool = False,
) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".npz":
        return [root]

    if layer is not None:
        layer_dir = root / f"layer_{int(layer):04d}"
        if layer_dir.is_dir():
            if include_legacy_cross_layer:
                return sorted(root.glob("layer_*/*.npz"))
            return sorted(layer_dir.glob("*.npz"))

    if not root.is_dir():
        return []

    direct = sorted(root.glob("*.npz"))
    if direct:
        return direct

    return sorted(root.glob("layer_*/*.npz"))


def _infer_layer_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.search(r"layer_(-?\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _legacy_physical_layer_z_map(root: Path) -> dict[int, float]:
    if root.is_file() or not root.is_dir():
        return {}
    if list(root.glob("*.npz")):
        return {}

    layer_heights = []
    for path in sorted(root.glob("layer_*/*.npz")):
        try:
            with np.load(str(path)) as data:
                if "layer_index" in data or "z" not in data:
                    continue
                if "move_type" not in data:
                    continue
                z_arr = np.asarray(data["z"], dtype=np.float64)
                move_type_arr = np.asarray(data["move_type"])
                move_vocab = _vocab(data, "move_type")
                print_mask = np.array([
                    move_vocab.get(int(value), str(int(value)))
                    in ("PRINT", "PRINT_FIT")
                    for value in move_type_arr
                ])
                if not np.any(print_mask):
                    continue
                if "e" in data:
                    e_arr = np.asarray(data["e"], dtype=np.float64)
                    de = np.diff(e_arr, prepend=e_arr[0])
                    deposit_mask = print_mask & (de > 1e-6)
                    if np.any(deposit_mask):
                        print_mask = deposit_mask
                layer_heights.append(float(np.median(z_arr[print_mask])))
        except Exception:
            continue

    heights = _cluster_z_heights(layer_heights)
    if len(heights) < 2:
        return {}
    return {index: height for index, height in enumerate(heights)}


def _cluster_z_heights(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(float(value) for value in values)
    clusters: list[list[float]] = []
    for value in sorted_values:
        if not clusters or abs(value - np.median(clusters[-1])) > 1e-4:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.median(cluster)) for cluster in clusters]


def _preview_image_layers(root: Path) -> list[int]:
    preview_dir = root / "layer_previews"
    if not preview_dir.is_dir():
        return []
    layers = []
    for path in preview_dir.glob("layer_*.png"):
        match = re.match(r"layer_(-?\d+)\.png$", path.name)
        if match:
            layers.append(int(match.group(1)))
    return sorted(set(layers))


def _flat_preview_layers(root: Path) -> list[int]:
    if root.is_file():
        preview_layers = _preview_image_layers(root.parent)
    elif root.is_dir():
        preview_layers = _preview_image_layers(root)
    else:
        preview_layers = []
    return preview_layers


def _legacy_flat_layer0_z_map(data, preview_layers: list[int]):
    layer0_display_layers = [layer for layer in preview_layers if layer <= 0]
    if len(layer0_display_layers) <= 1:
        return {}
    if not {"layer_index", "z", "e", "move_type"}.issubset(data.files):
        return {}

    layer_index = np.asarray(data["layer_index"], dtype=np.int64)
    z_arr = np.asarray(data["z"], dtype=np.float64)
    e_arr = np.asarray(data["e"], dtype=np.float64)
    move_type_arr = np.asarray(data["move_type"])
    de = np.diff(e_arr, prepend=e_arr[0])
    deposit_mask = (
        (layer_index == 0)
        & ((move_type_arr == 1) | (move_type_arr == 3))
        & (de > 1e-6)
    )
    z_values = z_arr[deposit_mask]
    if len(z_values) == 0:
        return {}

    hist, edges = np.histogram(z_values, bins=160)
    peak_indices = []
    min_count = max(10, int(hist.max() * 0.05)) if len(hist) else 10
    for idx, count in enumerate(hist):
        left = hist[idx - 1] if idx > 0 else -1
        right = hist[idx + 1] if idx + 1 < len(hist) else -1
        if count >= min_count and count >= left and count >= right:
            peak_indices.append(idx)
    heights = [0.5 * (float(edges[idx]) + float(edges[idx + 1]))
               for idx in peak_indices]
    heights = _cluster_z_heights(heights)
    if len(heights) < len(layer0_display_layers):
        quantiles = np.linspace(0.0, 1.0, len(layer0_display_layers) + 2)[1:-1]
        heights = [float(np.quantile(z_values, q)) for q in quantiles]
    heights = sorted(heights)[:len(layer0_display_layers)]
    z_map = dict(zip(layer0_display_layers, heights))

    for preview_layer in preview_layers:
        if preview_layer <= 0:
            continue
        layer_mask = (
            (layer_index == int(preview_layer))
            & ((move_type_arr == 1) | (move_type_arr == 3))
            & (de > 1e-6)
        )
        if np.any(layer_mask):
            z_map[int(preview_layer)] = float(np.median(z_arr[layer_mask]))
    return z_map


def _legacy_row_mask_for_physical_layer(
    z_arr,
    layer: int,
    layer_z_map: dict[int, float],
):
    if int(layer) not in layer_z_map:
        return np.zeros(len(z_arr), dtype=bool)
    z_values = np.asarray(z_arr, dtype=np.float64)
    items = sorted(layer_z_map.items(), key=lambda item: item[1])
    index = [item[0] for item in items].index(int(layer))
    target_z = float(items[index][1])
    if index == 0:
        lower = -np.inf
    else:
        lower = 0.5 * (float(items[index - 1][1]) + target_z)
    if index == len(items) - 1:
        upper = np.inf
    else:
        upper = 0.5 * (target_z + float(items[index + 1][1]))
    return (z_values >= lower) & (z_values < upper)


def _legacy_layer_z_tolerance(layer_z_map: dict[int, float]) -> float:
    zs = sorted(set(float(value) for value in layer_z_map.values()))
    gaps = [b - a for a, b in zip(zs, zs[1:]) if b - a > 1e-6]
    if not gaps:
        return 0.25
    return max(0.05, min(gaps) * 0.45)


def _rows_from_npz(
    path: Path,
    layer: int,
    root: Path | None = None,
    layer_z_map: dict[int, float] | None = None,
    max_rows: int | None = None,
) -> list[dict]:
    try:
        with np.load(str(path)) as data:
            required = ("x", "y", "z", "tool_id", "move_type")
            if any(name not in data for name in required):
                return []

            count = len(data["x"])
            move_vocab = _vocab(data, "move_type")
            event_vocab = _vocab(data, "event_type")

            x_arr = data["x"]
            y_arr = data["y"]
            z_arr = data["z"]
            a_arr = data["a"] if "a" in data.files else np.zeros(count)
            b_arr = data["b"] if "b" in data.files else np.zeros(count)
            c_arr = data["c"] if "c" in data.files else np.zeros(count)
            e_arr = data["e"] if "e" in data.files else np.zeros(count)
            tool_id_arr = data["tool_id"]
            move_type_arr = data["move_type"]

            if "preview_layer_index" in data:
                preview_layer = data["preview_layer_index"]
                mask = np.asarray(preview_layer, dtype=np.int64) == int(layer)
            elif "layer_index" in data:
                layer_index = data["layer_index"]
                mask = (
                    np.asarray(layer_index, dtype=np.int64)
                    == int(layer)
                )
            elif layer_z_map:
                mask = _legacy_row_mask_for_physical_layer(
                    z_arr,
                    int(layer),
                    layer_z_map,
                )
            else:
                inferred_layer = _infer_layer_from_path(path)
                if inferred_layer is not None and inferred_layer != int(layer):
                    return []
                mask = np.ones(count, dtype=bool)
            has_seq = "seq" in data.files
            x_arr = np.asarray(x_arr)
            y_arr = np.asarray(y_arr)
            z_arr = np.asarray(z_arr)
            a_arr = np.asarray(a_arr)
            b_arr = np.asarray(b_arr)
            c_arr = np.asarray(c_arr)
            e_arr = np.asarray(e_arr)
            tool_id_arr = np.asarray(tool_id_arr)
            move_type_arr = np.asarray(move_type_arr)
            src_line = _optional_array(data, "src_line", count, "")
            event_flag = _optional_array(data, "event_flag", count, 0)
            event_type = _optional_array(data, "event_type", count, 0)
            payload = _optional_array(data, "payload", count, "")
            seq = _optional_array(data, "seq", count, 0)
            path_id = _optional_array(data, "path_id", count, 0)
            path_end_flag = _optional_array(data, "path_end_flag", count, 0)

            row_indices = np.nonzero(mask)[0]
            if max_rows is not None and len(row_indices) > int(max_rows):
                sample_positions = np.linspace(
                    0,
                    len(row_indices) - 1,
                    max(2, int(max_rows)),
                    dtype=np.int64,
                )
                row_indices = row_indices[np.unique(sample_positions)]

            rows = []
            for idx in row_indices:
                move_type_value = int(move_type_arr[idx])
                event_type_value = int(event_type[idx])
                rows.append(
                    {
                        "seq": int(seq[idx]) if has_seq else idx,
                        "x": float(x_arr[idx]),
                        "y": float(y_arr[idx]),
                        "z": float(z_arr[idx]),
                        "a": float(a_arr[idx]),
                        "b": float(b_arr[idx]),
                        "c": float(c_arr[idx]),
                        "e": float(e_arr[idx]),
                        "tool_id": int(tool_id_arr[idx]),
                        "move_type": move_vocab.get(
                            move_type_value,
                            str(move_type_value),
                        ),
                        "src_line": _decode_value(src_line[idx]),
                        "event_flag": int(event_flag[idx]),
                        "event_type": event_vocab.get(
                            event_type_value,
                            _decode_value(event_type[idx]),
                        ),
                        "payload": _decode_value(payload[idx]),
                        "path_id": int(path_id[idx]),
                        "path_end_flag": int(path_end_flag[idx]),
                    }
                )
            return rows
    except Exception:
        return []


def _path_id_classification(rows: Sequence[dict]):
    for row in rows:
        if int(row.get("event_flag", 0)) == 1 or row.get("move_type") == "EVENT":
            event_type = row.get("event_type", "")
            path_type = (
                PathType.TOOL_CHANGE_EVENT
                if event_type in ("tool_change_cf", "tool_change_resin")
                else PathType.EVENT
            )
            return path_type, int(row.get("tool_id", 0)), event_type
    for row in rows:
        if row.get("move_type") in ("PRINT", "PRINT_FIT"):
            tool_id = int(row.get("tool_id", 0))
            if tool_id == 1:
                return PathType.FIBER_PRINT, tool_id, ""
            return PathType.RESIN_PRINT, tool_id, ""
    tool_id = int(rows[0].get("tool_id", 0)) if rows else 0
    return PathType.TRAVEL, tool_id, ""


def _extract_paths_from_rows(
    rows: Sequence[dict],
    layer: int,
    max_paths: int | None = None,
) -> list[PreviewPath]:
    paths: list[PreviewPath] = []
    current_key = None
    current_rows: list[dict] = []

    def flush():
        nonlocal current_key, current_rows
        if current_key is None or not current_rows:
            current_key = None
            current_rows = []
            return
        if current_key and current_key[0] == "PATH_ID":
            path_type, tool_id, event_type = _path_id_classification(current_rows)
            path_id = int(current_key[1])
        else:
            path_type, tool_id, event_type = current_key[:3]
            path_id = 0
        points = tuple((_point(row)) for row in current_rows)
        poses = tuple((_pose(row)) for row in current_rows)
        if _is_degenerate_print_path(path_type, points):
            current_key = None
            current_rows = []
            return
        if max_paths is not None and len(paths) >= max_paths:
            current_key = None
            current_rows = []
            return
        paths.append(
            PreviewPath(
                layer=layer,
                order_index=len(paths),
                path_type=path_type,
                tool_id=tool_id,
                points=points,
                poses=poses,
                start=points[0],
                end=points[-1],
                start_abc=poses[0][3:6],
                end_abc=poses[-1][3:6],
                src_line_start=current_rows[0]["src_line"],
                src_line_end=current_rows[-1]["src_line"],
                path_id=path_id,
                event_type=event_type,
                payload=current_rows[-1]["payload"],
            )
        )
        current_key = None
        current_rows = []

    prev_e = None
    prev_seq = None
    for index, row in enumerate(rows):
        current_seq = row.get("seq")
        if (
            prev_seq is not None
            and current_seq is not None
            and int(current_seq) - int(prev_seq) > _ROW_CONTINUITY_GAP_LIMIT
        ):
            flush()
            prev_e = None

        next_e = rows[index + 1].get("e") if index + 1 < len(rows) else None
        key = _classification_key(row, prev_e, next_e)
        prev_e = row.get("e", prev_e)
        prev_seq = current_seq
        if key is None:
            flush()
            continue
        if max_paths is not None and len(paths) >= max_paths:
            break
        grouping_key = key
        path_id = int(row.get("path_id", 0) or 0)
        if path_id > 0 and key[0] not in (PathType.TOOL_CHANGE_EVENT, PathType.EVENT):
            grouping_key = ("PATH_ID", path_id)
        if key[0] in (PathType.TOOL_CHANGE_EVENT, PathType.EVENT):
            flush()
            current_key = key
            current_rows = [row]
            flush()
            continue
        if current_key != grouping_key:
            flush()
            current_key = grouping_key
        current_rows.append(row)
        if int(row.get("path_end_flag", 0) or 0) == 1 and path_id > 0:
            flush()
    flush()
    return paths


def _classification_key(row: dict, prev_e=None, next_e=None):
    move_type = row["move_type"]
    tool_id = row["tool_id"]
    if int(row["event_flag"]) == 1 or move_type == "EVENT":
        event_type = row["event_type"]
        path_type = (
            PathType.TOOL_CHANGE_EVENT
            if event_type in ("tool_change_cf", "tool_change_resin")
            else PathType.EVENT
        )
        return path_type, tool_id, event_type
    if move_type in ("PRINT", "PRINT_FIT"):
        current_e = row.get("e", prev_e)
        delta_prev = current_e - prev_e if prev_e is not None else 0.0
        delta_next = next_e - current_e if next_e is not None else 0.0
        if delta_prev <= 1e-6 and delta_next <= 1e-6:
            return PathType.TRAVEL, tool_id, ""
        if tool_id == 1:
            return PathType.FIBER_PRINT, tool_id, ""
        if tool_id == 2:
            return PathType.RESIN_PRINT, tool_id, ""
        return PathType.RESIN_PRINT, tool_id, ""
    if move_type in ("TRAVEL", "TRAVEL_FIT"):
        return PathType.TRAVEL, tool_id, ""
    return None


def _is_degenerate_print_path(path_type: PathType, points: tuple[Point3, ...]):
    if path_type not in (PathType.FIBER_PRINT, PathType.RESIN_PRINT):
        return False
    if len(points) < 2:
        return True
    total_length = 0.0
    sx, sy, sz = points[0]
    for ex, ey, ez in points[1:]:
        dx = ex - sx
        dy = ey - sy
        dz = ez - sz
        total_length += float(np.sqrt(dx * dx + dy * dy + dz * dz))
        sx, sy, sz = ex, ey, ez
    return total_length <= 1e-3


def _point(row: dict) -> Point3:
    return (row["x"], row["y"], row["z"])


def _pose(row: dict) -> Pose6:
    return (
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
        float(row.get("a", 0.0)),
        float(row.get("b", 0.0)),
        float(row.get("c", 0.0)),
    )


def _vocab(data, prefix: str) -> dict[int, str]:
    keys_name = f"{prefix}_vocab_keys"
    vals_name = f"{prefix}_vocab_vals"
    if keys_name not in data or vals_name not in data:
        return {}
    return {
        int(val): _decode_value(key)
        for key, val in zip(data[keys_name], data[vals_name])
    }


def _optional_array(data, name: str, count: int, default):
    if name in data:
        return data[name]
    return np.array([default] * count)


def _decode_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if hasattr(value, "item"):
        return _decode_value(value.item())
    return str(value)
