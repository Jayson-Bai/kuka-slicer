#!/usr/bin/env python3
# Plot XY path from NPZ files (supports _partXXXX.npz sequences).

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def _resolve_npz_files(path: Path) -> List[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.npz"))
        return files

    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    if path.exists():
        return [path]

    # Try part files: <base>_part*.npz
    dir_path = path.parent if path.parent.as_posix() else Path(".")
    stem = path.stem
    prefix = f"{stem}_part"
    files = sorted([p for p in dir_path.glob("*.npz") if p.stem.startswith(prefix)])
    return files


def _decode_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if hasattr(value, "item"):
        return _decode_value(value.item())
    return str(value)


def _move_type_vocab(data) -> dict[int, str]:
    if "move_type_vocab_keys" not in data or "move_type_vocab_vals" not in data:
        return {}
    return {
        int(value): _decode_value(key)
        for key, value in zip(
            data["move_type_vocab_keys"],
            data["move_type_vocab_vals"],
        )
    }


def _append_positive_extrusion_xy(xs, ys, data, prev_point, prev_e):
    if not {"x", "y", "e", "move_type"}.issubset(data.files):
        return prev_point, prev_e

    x_arr = data["x"].astype(np.float32, copy=False)
    y_arr = data["y"].astype(np.float32, copy=False)
    e_arr = data["e"].astype(np.float32, copy=False)
    move_type_arr = data["move_type"]
    move_vocab = _move_type_vocab(data)
    in_segment = False

    for idx in range(len(x_arr)):
        current_point = (x_arr[idx], y_arr[idx])
        current_e = e_arr[idx]
        if prev_point is None or prev_e is None:
            prev_point = current_point
            prev_e = current_e
            continue

        move_type_value = int(move_type_arr[idx])
        move_type = move_vocab.get(move_type_value, str(move_type_value))
        is_deposit = (
            move_type in ("PRINT", "PRINT_FIT")
            and (current_e - prev_e) > 1e-6
        )
        if is_deposit:
            if not in_segment:
                if xs:
                    xs.append(np.float32(np.nan))
                    ys.append(np.float32(np.nan))
                xs.append(np.float32(prev_point[0]))
                ys.append(np.float32(prev_point[1]))
                in_segment = True
            xs.append(np.float32(current_point[0]))
            ys.append(np.float32(current_point[1]))
        else:
            in_segment = False

        prev_point = current_point
        prev_e = current_e

    return prev_point, prev_e


def load_xy(files: List[Path], include_travel: bool = False) -> np.ndarray:
    xs = []
    ys = []
    prev_point = None
    prev_e = None
    for f in files:
        with np.load(str(f)) as data:
            if "x" not in data or "y" not in data:
                continue
            if include_travel:
                xs.append(data["x"].astype(np.float32, copy=False))
                ys.append(data["y"].astype(np.float32, copy=False))
                continue
            prev_point, prev_e = _append_positive_extrusion_xy(
                xs,
                ys,
                data,
                prev_point,
                prev_e,
            )
    if not xs:
        return np.empty((0, 2), dtype=np.float32)
    if include_travel:
        x = np.concatenate(xs)
        y = np.concatenate(ys)
    else:
        x = np.array(xs, dtype=np.float32)
        y = np.array(ys, dtype=np.float32)
    return np.stack([x, y], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot XY trajectory from NPZ files.")
    parser.add_argument("--npz", required=True, help="NPZ file path, base path, or directory")
    parser.add_argument("--out", default="xy_path.png", help="Output image path (png)")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    parser.add_argument("--stride", type=int, default=1, help="Plot every Nth point to speed up")
    parser.add_argument(
        "--include-travel",
        action="store_true",
        help="Plot all XY points, including travel and non-extruding moves",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    files = _resolve_npz_files(npz_path)
    if not files:
        raise SystemExit(f"[error] no npz files found for: {npz_path}")

    xy = load_xy(files, include_travel=args.include_travel)
    if xy.size == 0:
        raise SystemExit("[error] no x/y data found in npz files")

    if args.stride > 1:
        xy = xy[:: args.stride]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)
    ax.plot(xy[:, 0], xy[:, 1], linewidth=0.6, color="#2b2b2b")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("XY Path")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"[info] saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
