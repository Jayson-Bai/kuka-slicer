#!/usr/bin/env python3
# Split a G-code file into per-layer files using ";LAYER_CHANGE" markers.

from __future__ import annotations

import argparse
from pathlib import Path


def split_gcode_by_layer(input_path: Path, output_dir: Path, prefix: str) -> int:
    lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    header: list[str] = []
    layer_blocks: list[list[str]] = []
    current: list[str] | None = None

    for line in lines:
        if line.lstrip().startswith(";LAYER_CHANGE"):
            if current is not None:
                layer_blocks.append(current)
            current = [line]
            continue

        if current is None:
            header.append(line)
        else:
            current.append(line)

    if current is not None:
        layer_blocks.append(current)

    if not layer_blocks:
        # Fallback: no layer marker, write single file with full content.
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{prefix}_layer_0001.gcode"
        out_path.write_text("".join(lines), encoding="utf-8")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, block in enumerate(layer_blocks, start=1):
        out_path = output_dir / f"{prefix}_layer_{idx:04d}.gcode"
        out_path.write_text("".join(header + block), encoding="utf-8")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Split G-code into per-layer files.")
    parser.add_argument("--input", required=True, help="Input .gcode file path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--prefix",
        default="split",
        help="Output filename prefix (default: split)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prefix = args.prefix

    count = split_gcode_by_layer(input_path, output_dir, prefix)
    print(f"[info] Wrote {count} layer files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
