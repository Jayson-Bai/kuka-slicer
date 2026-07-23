#!/usr/bin/env python3
# Plot per-layer XY paths using manifest order.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def resolve_npz_files(base_path: Path) -> List[Path]:
    if base_path.suffix != ".npz":
        base_path = base_path.with_suffix(".npz")
    if base_path.exists():
        return [base_path]
    dir_path = base_path.parent
    prefix = f"{base_path.stem}_part"
    return sorted([p for p in dir_path.glob("*.npz") if p.stem.startswith(prefix)])


def load_xy_from_base(base_path: Path, stride: int, only_deposit: bool) -> Tuple[np.ndarray, np.ndarray]:
    files = resolve_npz_files(base_path)
    xs = []
    ys = []
    for f in files:
        z = np.load(str(f))
        if "x" not in z or "y" not in z:
            continue
        x = z["x"]
        y = z["y"]
        if only_deposit:
            if "e" not in z or "move_type" not in z:
                continue
            e = z["e"]
            mt = z["move_type"]
            # move_type: 1=PRINT, 3=PRINT_FIT
            is_print = (mt == 1) | (mt == 3)
            # deposit only when E increases
            de = np.diff(e, prepend=e[0])
            is_deposit = is_print & (de > 1e-6)
            # break lines on non-deposit by inserting NaN
            x = x.astype(np.float32, copy=False)
            y = y.astype(np.float32, copy=False)
            x = np.where(is_deposit, x, np.nan)
            y = np.where(is_deposit, y, np.nan)
        if stride > 1:
            x = x[::stride]
            y = y[::stride]
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(xs), np.concatenate(ys)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot per-layer XY paths from manifest.")
    parser.add_argument("--manifest", required=True, help="manifest.json path")
    parser.add_argument("--out-dir", required=True, help="output directory for layer images")
    parser.add_argument("--stride", type=int, default=1, help="plot every Nth point")
    parser.add_argument("--dpi", type=int, default=150, help="image DPI")
    parser.add_argument("--only-print", action="store_true", help="only plot PRINT segments")
    parser.add_argument("--only-deposit", action="store_true", help="only plot points with positive E (real deposition)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    by_layer: Dict[int, List[dict]] = {}
    for it in items:
        if args.only_print:
            t = str(it.get("type", "")).upper()
            if t == "TRAVEL":
                continue
        layer = int(it.get("layer", 0))
        by_layer.setdefault(layer, []).append(it)

    for layer, segs in sorted(by_layer.items()):
        xs_all = []
        ys_all = []
        first_seg = True
        for seg in segs:
            base_path = Path(seg["base_path"]).expanduser().resolve()
            xs, ys = load_xy_from_base(base_path, args.stride, args.only_deposit)
            if xs.size == 0:
                continue
            if not first_seg:
                xs_all.append(np.array([np.nan], dtype=np.float32))
                ys_all.append(np.array([np.nan], dtype=np.float32))
            first_seg = False
            xs_all.append(xs)
            ys_all.append(ys)
        if not xs_all:
            continue
        x = np.concatenate(xs_all)
        y = np.concatenate(ys_all)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)
        ax.plot(x, y, linewidth=0.6, color="#2b2b2b")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Layer {layer:04d} XY Path")
        ax.grid(True, linewidth=0.3, alpha=0.5)

        out_path = out_dir / f"layer_{layer:04d}.png"
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        print(f"[info] saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
