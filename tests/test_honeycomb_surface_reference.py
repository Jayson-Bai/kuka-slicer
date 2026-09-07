from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from kuka_slicer.surface_peak_collision import resolve_surface_collision_inputs


DEFAULT_REFERENCE_ROOT = Path(
    r"C:\Users\caRRot\Desktop\print_test\曲面测试\蜂窝-150x100x10mm-08-11"
)
REFERENCE_ROOT_ENV = "KUKA_SLICER_HONEYCOMB_REFERENCE_ROOT"
STL_RELATIVE_PATH = Path("01_模型") / "蜂窝-150x100x10mm-08-11-边长5mm-模型.stl"
SURFACE_RELATIVE_PATH = (
    Path("02_曲面参数与预览") / "蜂窝-150x100x10mm-08-11-相位45度-全边界曲面参数.json"
)
NPZ_RELATIVE_PATH = (
    Path("02_曲面参数与预览")
    / "honeycomb_minimum_no_u_turn_partitions_surface_preview.npz"
)
EXPECTED_STL_SHA256 = "4d8750a3f9759fcaced60b9d7ecf8c55cdeffa12dea57a47579ed4e3aebd66c9"
EXPECTED_SURFACE_SHA256 = "6ced5675fe56194adb07c47db637f829aa4ae955a863a688ed393c510c1a2491"


def _reference_root() -> Path:
    configured = os.environ.get(REFERENCE_ROOT_ENV)
    return Path(configured) if configured else DEFAULT_REFERENCE_ROOT


def _reference_files() -> tuple[Path, Path, Path]:
    root = _reference_root()
    files = (
        root / STL_RELATIVE_PATH,
        root / SURFACE_RELATIVE_PATH,
        root / NPZ_RELATIVE_PATH,
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        pytest.skip(
            "external honeycomb surface reference is unavailable; "
            f"set {REFERENCE_ROOT_ENV} to its root (missing: {missing})"
        )
    return files


def test_phase_45_minimum_partition_reference_matches_production_contract() -> None:
    """Lock the real 150 x 100 x 10 mm sample to the production planner contract."""

    stl_path, surface_path, npz_path = _reference_files()
    stl_sha256 = hashlib.sha256(stl_path.read_bytes()).hexdigest()
    surface = json.loads(surface_path.read_text(encoding="utf-8-sig"))
    canonical_surface = json.dumps(
        surface,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    surface_sha256 = hashlib.sha256(canonical_surface).hexdigest()

    assert stl_sha256 == EXPECTED_STL_SHA256
    assert surface_sha256 == EXPECTED_SURFACE_SHA256
    assert surface["domain"]["source"]["sha256"] == EXPECTED_STL_SHA256
    assert surface["domain"]["source"]["xy_bounds_mm"] == [0.0, 0.0, 150.0, 100.0]
    assert surface["surface"]["phase_x_rad"] == pytest.approx(math.pi / 4.0)
    assert surface["surface"]["phase_y_rad"] == pytest.approx(math.pi / 4.0)

    with np.load(npz_path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
        planner = meta["honeycomb_centerline_pathing"]
        mapping = meta["surface_mapping"]
        progression = mapping["progression"]

        assert planner["format"] == "honeycomb_macro_partition_zero_e_v1"
        assert planner["topology"] == "stl_honeycomb_macro_partition_zero_e"
        assert planner["topology_change"].startswith("none;")
        assert planner["macro_partition_count"] == 6
        assert planner["layer_path_count"] == 7
        assert planner["travel_count"] == 5
        assert planner["wall_edge_count"] == 693
        assert planner["minimum_trail_count"] == 251
        assert planner["repeated_wall_edge_count"] == 0
        assert planner["repeated_wall_length_mm_per_layer"] == 0.0
        assert planner["intra_partition_zero_e_max_turn_degrees"] <= 90.0

        assert mapping["target_source_sha256"] == EXPECTED_STL_SHA256
        assert mapping["target_surface_sha256"] == EXPECTED_SURFACE_SHA256
        assert mapping["sampling"]["max_segment_length_mm"] == 1.5
        assert mapping["extrusion"] == "arc_length_ratio_compensated"
        assert mapping["orientation"]["mode"] == "surface_normal_kuka_zyx"
        assert progression["surface_start_layer"] == 3
        assert progression["surface_return_layer"] == 16
        assert progression["peak_layers"] == [9, 10]

        z_values: list[np.ndarray] = []
        for layer in range(20):
            resin = data[f"layer_{layer:04d}_R"]
            extrusion = data[f"layer_{layer:04d}_R_E"]
            travel = data[f"layer_{layer:04d}_T"]
            assert resin.shape[0] == 7
            assert travel.shape[0] == 5
            assert extrusion.shape == resin.shape[:2]
            assert meta["path_roles"]["R"][str(layer)] == [
                "outer_contour",
                *("honeycomb_wall" for _ in range(6)),
            ]
            assert meta["motion_order"][str(layer)] == [
                {"kind": "deposit", "index": 0},
                {"kind": "deposit", "index": 1},
                {"kind": "travel", "index": 0},
                {"kind": "deposit", "index": 2},
                {"kind": "travel", "index": 1},
                {"kind": "deposit", "index": 3},
                {"kind": "travel", "index": 2},
                {"kind": "deposit", "index": 4},
                {"kind": "travel", "index": 3},
                {"kind": "deposit", "index": 5},
                {"kind": "travel", "index": 4},
                {"kind": "deposit", "index": 6},
            ]
            valid = np.isfinite(resin[..., 0])
            z_values.append(resin[..., 2][valid])
            assert any(
                np.any(
                    (np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > 1e-8)
                    & np.isclose(np.diff(values), 0.0, atol=1e-9)
                )
                for path, values in zip(resin[1:], extrusion[1:])
                if np.isfinite(path[:, 0]).sum() >= 2
            )

        all_z = np.concatenate(z_values)
        assert float(np.min(all_z)) == pytest.approx(0.5)
        assert float(np.max(all_z)) == pytest.approx(10.0)


def test_peak_collision_preflight_resolves_the_reference_by_hash() -> None:
    stl_path, surface_path, npz_path = _reference_files()

    source, target, resolved_surface, resolved_stl = resolve_surface_collision_inputs(npz_path)

    assert resolved_surface == surface_path
    assert resolved_stl == stl_path
    assert target.raw_config["surface"]["phase_x_rad"] == pytest.approx(math.pi / 4.0)
    assert source.meta["surface_mapping"]["progression"]["peak_layers"] == [9, 10]
