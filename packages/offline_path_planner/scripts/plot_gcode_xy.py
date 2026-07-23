#!/usr/bin/env python3
# Plot XY path from a G-code file (uses G0/G1 moves, keeps last X/Y).

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_xy_from_gcode(lines):
    x = None
    y = None
    xs = []
    ys = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        # strip inline comments
        if ";" in line:
            line = line.split(";", 1)[0].strip()
        if not line:
            continue
        if not (line.startswith("G0") or line.startswith("G1")):
            continue
        parts = line.split()
        for p in parts[1:]:
            if p.startswith("X"):
                try:
                    x = float(p[1:])
                except ValueError:
                    pass
            elif p.startswith("Y"):
                try:
                    y = float(p[1:])
                except ValueError:
                    pass
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description="Plot XY trajectory from G-code.")
    parser.add_argument("--gcode", required=True, help="G-code file path")
    parser.add_argument("--out", default="gcode_xy.png", help="Output image path")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    parser.add_argument("--stride", type=int, default=1, help="Plot every Nth point")
    args = parser.parse_args()

    gcode_path = Path(args.gcode).expanduser().resolve()
    lines = gcode_path.read_text(encoding="utf-8", errors="replace").splitlines()
    xs, ys = parse_xy_from_gcode(lines)
    if not xs:
        raise SystemExit("[error] no XY moves found in G-code")

    if args.stride > 1:
        xs = xs[:: args.stride]
        ys = ys[:: args.stride]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)
    ax.plot(xs, ys, linewidth=0.6, color="#1f77b4")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("G-code XY Path")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"[info] saved: {out_path}")


if __name__ == "__main__":
    main()
