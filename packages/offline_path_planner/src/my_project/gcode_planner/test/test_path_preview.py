from pathlib import Path

import numpy as np
import pytest


def _write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        seq=np.array(
            arrays.get("seq", np.arange(len(arrays["x"]))),
            dtype=np.uint32,
        ),
        x=np.array(arrays["x"], dtype=np.float32),
        y=np.array(arrays["y"], dtype=np.float32),
        z=np.array(arrays["z"], dtype=np.float32),
        a=np.array(
            arrays.get("a", [0.0] * len(arrays["x"])),
            dtype=np.float32,
        ),
        b=np.array(
            arrays.get("b", [0.0] * len(arrays["x"])),
            dtype=np.float32,
        ),
        c=np.array(
            arrays.get("c", [0.0] * len(arrays["x"])),
            dtype=np.float32,
        ),
        e=np.array(arrays["e"], dtype=np.float32),
        tool_id=np.array(arrays["tool_id"], dtype=np.uint8),
        move_type=np.array(arrays["move_type"], dtype=np.uint8),
        src_line=np.array(arrays["src_line"], dtype="S32"),
        event_flag=np.array(
            arrays.get("event_flag", [0] * len(arrays["x"])),
            dtype=np.uint8,
        ),
        event_type=np.array(
            arrays.get("event_type", [0] * len(arrays["x"])),
            dtype=np.uint8,
        ),
        payload=np.array(
            arrays.get("payload", [""] * len(arrays["x"])),
            dtype="S32",
        ),
        layer_index=np.array(arrays["layer_index"], dtype=np.uint32),
        preview_layer_index=np.array(
            arrays.get("preview_layer_index", arrays["layer_index"]),
            dtype=np.int32,
        ),
        path_id=np.array(
            arrays.get("path_id", [0] * len(arrays["x"])),
            dtype=np.uint32,
        ),
        path_end_flag=np.array(
            arrays.get("path_end_flag", [0] * len(arrays["x"])),
            dtype=np.uint8,
        ),
        total_layers=np.array([2] * len(arrays["x"]), dtype=np.uint32),
        move_type_vocab_keys=np.array(
            [b"TRAVEL", b"PRINT", b"TRAVEL_FIT", b"PRINT_FIT", b"EVENT"],
            dtype="S32",
        ),
        move_type_vocab_vals=np.array([0, 1, 2, 3, 4], dtype=np.uint8),
        event_type_vocab_keys=np.array(
            [
                b"",
                b"heat_cf",
                b"heat_resin",
                b"fan_cf",
                b"fan_resin",
                b"extrude_reset",
                b"tool_change_cf",
                b"tool_change_resin",
            ],
            dtype="S32",
        ),
        event_type_vocab_vals=np.array(
            [0, 1, 2, 3, 4, 5, 6, 7],
            dtype=np.uint8,
        ),
    )


def test_extract_preview_paths_classifies_process_paths_from_flat_npz(
    tmp_path,
):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
    )

    root = tmp_path / "job"
    _write_npz(
        root / "job.npz",
        x=[0, 1, 2, 2, 3, 4, 4, 5],
        y=[0, 0, 0, 1, 1, 1, 2, 2],
        z=[0.2] * 8,
        e=[0, 0, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9],
        tool_id=[1, 1, 1, 2, 2, 2, 2, 1],
        move_type=[0, 1, 1, 1, 1, 1, 4, 0],
        event_flag=[0, 0, 0, 0, 0, 0, 1, 0],
        event_type=[0, 0, 0, 0, 0, 0, 6, 0],
        src_line=["1", "2", "3", "4", "5", "6", "7", "8"],
        payload=["", "", "", "", "", "", "1", ""],
        layer_index=[0] * 8,
    )

    paths = extract_layer_preview_paths(root, 0)

    assert [p.path_type for p in paths] == [
        PathType.TRAVEL,
        PathType.FIBER_PRINT,
        PathType.TRAVEL,
        PathType.RESIN_PRINT,
        PathType.TOOL_CHANGE_EVENT,
        PathType.TRAVEL,
    ]
    assert paths[1].tool_id == 1
    assert paths[1].start == pytest.approx((1.0, 0.0, 0.2))
    assert paths[1].end == pytest.approx((2.0, 0.0, 0.2))
    assert paths[3].tool_id == 2
    assert paths[4].points[0] == pytest.approx((4.0, 2.0, 0.2))
    assert [p.order_index for p in paths] == list(range(len(paths)))


def test_extract_preview_paths_prefers_path_id_boundaries(tmp_path):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
    )

    root = tmp_path / "job"
    _write_npz(
        root / "job.npz",
        x=[0, 1, 2, 3],
        y=[0, 0, 0, 0],
        z=[0.2, 0.2, 0.2, 0.2],
        e=[0.1, 0.5, 0.6, 1.0],
        tool_id=[2, 2, 2, 2],
        move_type=[1, 1, 1, 1],
        path_id=[11, 11, 12, 12],
        path_end_flag=[0, 1, 0, 1],
        src_line=["10", "11", "12", "13"],
        layer_index=[0, 0, 0, 0],
    )

    paths = extract_layer_preview_paths(root, 0)

    assert [p.path_type for p in paths] == [
        PathType.RESIN_PRINT,
        PathType.RESIN_PRINT,
    ]
    assert [p.path_id for p in paths] == [11, 12]
    assert paths[0].start == pytest.approx((0.0, 0.0, 0.2))
    assert paths[0].end == pytest.approx((1.0, 0.0, 0.2))
    assert paths[1].start == pytest.approx((2.0, 0.0, 0.2))
    assert paths[1].end == pytest.approx((3.0, 0.0, 0.2))


def test_extract_preview_paths_does_not_split_one_path_id_on_local_e_plateaus(tmp_path):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
    )

    root = tmp_path / "job"
    _write_npz(
        root / "job.npz",
        x=[0, 1, 2, 3, 4],
        y=[0, 0, 0, 0, 0],
        z=[0.2] * 5,
        e=[0.1, 0.1, 0.5, 0.5, 1.0],
        tool_id=[2, 2, 2, 2, 2],
        move_type=[1, 1, 1, 1, 1],
        path_id=[42, 42, 42, 42, 42],
        path_end_flag=[0, 0, 0, 0, 1],
        src_line=["10", "11", "12", "13", "14"],
        layer_index=[0] * 5,
    )

    paths = extract_layer_preview_paths(root, 0)

    assert len(paths) == 1
    assert paths[0].path_type == PathType.RESIN_PRINT
    assert paths[0].path_id == 42
    assert paths[0].start == pytest.approx((0.0, 0.0, 0.2))
    assert paths[0].end == pytest.approx((4.0, 0.0, 0.2))


def test_extract_preview_paths_uses_split_layer_directory(tmp_path):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
        list_preview_layers,
    )

    root = tmp_path / "job"
    _write_npz(
        root / "layer_0001" / "job_layer_0001_type_PRINT_occ_0001.npz",
        x=[10, 11],
        y=[0, 0],
        z=[0.4, 0.4],
        e=[0.1, 0.5],
        tool_id=[1, 1],
        move_type=[1, 1],
        src_line=["20", "21"],
        layer_index=[1, 1],
    )

    assert list_preview_layers(root) == [1]
    paths = extract_layer_preview_paths(root, 1)

    assert len(paths) == 1
    assert paths[0].path_type == PathType.FIBER_PRINT
    assert paths[0].start == pytest.approx((10.0, 0.0, 0.4))
    assert paths[0].end == pytest.approx((11.0, 0.0, 0.4))


def test_extract_preview_paths_preserves_xyzabc_pose_for_tool_preview(
    tmp_path,
):
    from gcode_planner.path_preview import extract_layer_preview_paths

    root = tmp_path / "job"
    _write_npz(
        root / "job.npz",
        x=[0, 1, 2],
        y=[0, 0, 1],
        z=[0.2, 0.3, 0.4],
        a=[10.0, 20.0, 30.0],
        b=[1.0, 2.0, 3.0],
        c=[-5.0, -6.0, -7.0],
        e=[0.0, 0.2, 0.4],
        tool_id=[2, 2, 2],
        move_type=[1, 1, 1],
        src_line=["1", "2", "3"],
        layer_index=[0, 0, 0],
    )

    paths = extract_layer_preview_paths(root, 0)

    assert paths[0].poses[0] == pytest.approx(
        (0.0, 0.0, 0.2, 10.0, 1.0, -5.0)
    )
    assert paths[0].start_abc == pytest.approx((10.0, 1.0, -5.0))
    assert paths[0].end_abc == pytest.approx((30.0, 3.0, -7.0))


def test_extract_preview_paths_supports_legacy_split_npz_without_layer_index(
    tmp_path,
):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
        list_preview_layers,
    )

    root = tmp_path / "legacy_job"
    path = (
        root
        / "layer_0003"
        / "legacy_job_layer_0003_type_PRINT_occ_0001.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        seq=np.array([0, 1], dtype=np.uint32),
        x=np.array([1.0, 2.0], dtype=np.float32),
        y=np.array([3.0, 3.0], dtype=np.float32),
        z=np.array([0.6, 0.6], dtype=np.float32),
        e=np.array([0.2, 0.8], dtype=np.float32),
        tool_id=np.array([2, 2], dtype=np.uint8),
        move_type=np.array([1, 1], dtype=np.uint8),
        src_line=np.array(["10", "11"], dtype="S32"),
        event_flag=np.array([0, 0], dtype=np.uint8),
        event_type=np.array([0, 0], dtype=np.uint8),
        payload=np.array(["", ""], dtype="S32"),
        move_type_vocab_keys=np.array([b"TRAVEL", b"PRINT"], dtype="S32"),
        move_type_vocab_vals=np.array([0, 1], dtype=np.uint8),
        event_type_vocab_keys=np.array([b""], dtype="S32"),
        event_type_vocab_vals=np.array([0], dtype=np.uint8),
    )

    assert list_preview_layers(root) == [3]
    paths = extract_layer_preview_paths(root, 3)

    assert len(paths) == 1
    assert paths[0].path_type == PathType.RESIN_PRINT
    assert paths[0].start == pytest.approx((1.0, 3.0, 0.6))
    assert paths[0].end == pytest.approx((2.0, 3.0, 0.6))


def test_legacy_custom_paths_are_routed_by_layer_height(tmp_path):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
    )

    root = tmp_path / "legacy_height_job"
    _write_npz(
        root / "layer_0000" / "job_layer_0000_type_Custom_occ_0001.npz",
        x=[0, 1],
        y=[0, 0],
        z=[2.0, 2.0],
        e=[0.0, 0.0],
        tool_id=[2, 2],
        move_type=[0, 0],
        src_line=["1", "2"],
        layer_index=[0, 0],
    )
    _write_npz(
        root / "layer_0000" / "job_layer_0000_type_PRINT_occ_0001.npz",
        x=[0, 1],
        y=[1, 1],
        z=[0.5, 0.5],
        e=[0.0, 0.5],
        tool_id=[2, 2],
        move_type=[1, 1],
        src_line=["3", "4"],
        layer_index=[0, 0],
    )
    _write_npz(
        root / "layer_0002" / "job_layer_0002_type_PRINT_occ_0001.npz",
        x=[2, 3],
        y=[1, 1],
        z=[2.0, 2.0],
        e=[0.5, 1.0],
        tool_id=[2, 2],
        move_type=[1, 1],
        src_line=["5", "6"],
        layer_index=[2, 2],
    )

    for npz_path in root.glob("layer_*/*.npz"):
        with np.load(str(npz_path)) as data:
            arrays = {name: data[name] for name in data.files}
        arrays.pop("layer_index", None)
        arrays.pop("preview_layer_index", None)
        np.savez_compressed(str(npz_path), **arrays)

    layer0_paths = extract_layer_preview_paths(root, 0)
    layer1_paths = extract_layer_preview_paths(root, 1)

    assert [path.path_type for path in layer0_paths] == [
        PathType.RESIN_PRINT,
    ]
    assert [path.path_type for path in layer1_paths] == [
        PathType.TRAVEL,
        PathType.RESIN_PRINT,
    ]
    assert layer1_paths[0].start == pytest.approx((0.0, 0.0, 2.0))
    assert layer1_paths[0].end == pytest.approx((1.0, 0.0, 2.0))


def test_legacy_split_paths_are_split_by_physical_z_layers(tmp_path):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
        list_preview_layers,
    )

    root = tmp_path / "legacy_physical_layers"
    _write_npz(
        root / "layer_0000" / "job_layer_0000_type_Custom_occ_0001.npz",
        x=[0, 1, 0, 1],
        y=[0, 0, 2, 2],
        z=[0.5, 0.5, 1.0, 1.0],
        e=[0.0, 0.0, 0.0, 0.0],
        tool_id=[2, 2, 2, 2],
        move_type=[0, 0, 0, 0],
        src_line=["1", "2", "3", "4"],
        layer_index=[0, 0, 0, 0],
    )
    _write_npz(
        root / "layer_0001" / "job_layer_0001_type_PRINT_occ_0001.npz",
        x=[10, 11],
        y=[0, 0],
        z=[0.5, 0.5],
        e=[0.0, 0.5],
        tool_id=[2, 2],
        move_type=[1, 1],
        src_line=["5", "6"],
        layer_index=[1, 1],
    )
    _write_npz(
        root / "layer_0002" / "job_layer_0002_type_PRINT_occ_0001.npz",
        x=[20, 21],
        y=[0, 0],
        z=[1.0, 1.0],
        e=[0.5, 1.0],
        tool_id=[2, 2],
        move_type=[1, 1],
        src_line=["7", "8"],
        layer_index=[2, 2],
    )

    for npz_path in root.glob("layer_*/*.npz"):
        with np.load(str(npz_path)) as data:
            arrays = {name: data[name] for name in data.files}
        arrays.pop("layer_index", None)
        arrays.pop("preview_layer_index", None)
        np.savez_compressed(str(npz_path), **arrays)

    assert list_preview_layers(root) == [0, 1]

    layer0_paths = extract_layer_preview_paths(root, 0)
    layer1_paths = extract_layer_preview_paths(root, 1)

    assert [path.path_type for path in layer0_paths] == [
        PathType.TRAVEL,
        PathType.RESIN_PRINT,
    ]
    assert [path.path_type for path in layer1_paths] == [
        PathType.TRAVEL,
        PathType.RESIN_PRINT,
    ]
    layer0_z = {point[2] for path in layer0_paths for point in path.points}
    layer1_z = {point[2] for path in layer1_paths for point in path.points}
    assert layer0_z == {0.5}
    assert layer1_z == {1.0}


def test_list_preview_layers_prefers_preview_layer_indices_for_display_layers(tmp_path):
    from gcode_planner.path_preview import (
        extract_layer_preview_paths,
        list_preview_layers,
    )

    root = tmp_path / "modern_job"
    _write_npz(
        root / "modern_job.npz",
        x=[0, 1, 2, 3],
        y=[0, 0, 0, 0],
        z=[0.2, 0.2, 0.4, 0.4],
        e=[0.0, 0.2, 0.4, 0.6],
        tool_id=[2, 2, 2, 2],
        move_type=[1, 1, 1, 1],
        src_line=["1", "2", "3", "4"],
        layer_index=[0, 0, 1, 1],
        preview_layer_index=[-4, -4, -3, -3],
    )
    preview_dir = root / "layer_previews"
    preview_dir.mkdir(parents=True)
    for name in ("layer_-004.png", "layer_-003.png", "layer_0000.png"):
        (preview_dir / name).write_bytes(b"stale")

    assert list_preview_layers(root) == [-4, -3]
    valve_paths = extract_layer_preview_paths(root, -4)
    assert valve_paths
    assert {round(point[2], 6) for path in valve_paths for point in path.points} == {0.2}


def test_extract_preview_paths_can_limit_returned_paths(tmp_path):
    from gcode_planner.path_preview import extract_layer_preview_paths

    root = tmp_path / "limited_job"
    _write_npz(
        root / "limited_job.npz",
        x=[0, 1, 2, 3, 4],
        y=[0, 0, 1, 1, 2],
        z=[0.2] * 5,
        e=[0, 0, 0.2, 0.2, 0.4],
        tool_id=[2, 2, 2, 2, 2],
        move_type=[0, 1, 1, 0, 1],
        src_line=["1", "2", "3", "4", "5"],
        layer_index=[0] * 5,
    )

    paths = extract_layer_preview_paths(root, 0, max_paths=2)

    assert len(paths) == 2


def test_extract_preview_paths_can_downsample_large_layers_before_building_rows(tmp_path):
    from gcode_planner.path_preview import extract_layer_preview_paths

    root = tmp_path / "large_layer"
    count = 101
    _write_npz(
        root / "large_layer.npz",
        x=list(range(count)),
        y=[0.0] * count,
        z=[0.2] * count,
        e=[float(i) * 0.1 for i in range(count)],
        tool_id=[2] * count,
        move_type=[1] * count,
        src_line=[str(i) for i in range(count)],
        layer_index=[0] * count,
    )

    paths = extract_layer_preview_paths(root, 0, max_rows=11)

    assert len(paths) == 1
    assert len(paths[0].points) <= 11
    assert paths[0].start == pytest.approx((0.0, 0.0, 0.2))
    assert paths[0].end == pytest.approx((100.0, 0.0, 0.2))


def test_zero_length_print_paths_are_omitted(tmp_path):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
    )

    root = tmp_path / "zero_length_print"
    _write_npz(
        root / "zero_length_print.npz",
        x=[0, 0, 1, 2, 3],
        y=[0, 0, 0, 0, 0],
        z=[0.2] * 5,
        e=[0.0, 0.5, 0.5, 1.0, 1.5],
        tool_id=[2, 2, 2, 2, 2],
        move_type=[1, 1, 0, 1, 1],
        src_line=["1", "2", "3", "4", "5"],
        layer_index=[0] * 5,
    )

    paths = extract_layer_preview_paths(root, 0)

    assert [path.path_type for path in paths] == [
        PathType.TRAVEL,
        PathType.RESIN_PRINT,
    ]
    assert paths[1].start == pytest.approx((2.0, 0.0, 0.2))
    assert paths[1].end == pytest.approx((3.0, 0.0, 0.2))


def test_preview_paths_break_across_noncontiguous_rows_after_layer_filtering(tmp_path):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
    )

    root = tmp_path / "seq_gap"
    _write_npz(
        root / "seq_gap.npz",
        seq=[10, 11, 20000, 20001, 20002],
        x=[0.0, 0.0, 50.0, 51.0, 52.0],
        y=[0.0, 0.0, 50.0, 50.0, 50.0],
        z=[0.2] * 5,
        e=[0.0, -1.0, 10.0, 10.5, 11.0],
        tool_id=[2, 2, 2, 2, 2],
        move_type=[1, 1, 3, 3, 3],
        src_line=["25", "25", "905-1006", "905-1006", "905-1006"],
        layer_index=[0] * 5,
        preview_layer_index=[0] * 5,
    )

    paths = extract_layer_preview_paths(root, 0)
    print_paths = [path for path in paths if path.path_type == PathType.RESIN_PRINT]

    assert len(print_paths) == 1
    assert print_paths[0].start == pytest.approx((50.0, 50.0, 0.2))
    assert print_paths[0].end == pytest.approx((52.0, 50.0, 0.2))
    assert not any(
        path.start == pytest.approx((0.0, 0.0, 0.2))
        and path.end == pytest.approx((50.0, 50.0, 0.2))
        for path in print_paths
    )


def test_print_rows_without_extrusion_are_treated_as_travel(tmp_path):
    from gcode_planner.path_preview import (
        PathType,
        extract_layer_preview_paths,
    )

    root = tmp_path / "non_extruding_print"
    _write_npz(
        root / "non_extruding_print.npz",
        x=[0, 1, 2, 3],
        y=[0, 0, 0, 0],
        z=[0.2] * 4,
        e=[1.0, 1.0, 1.0, 1.5],
        tool_id=[2, 2, 2, 2],
        move_type=[1, 1, 1, 1],
        src_line=["1", "2", "3", "4"],
        layer_index=[0] * 4,
    )

    paths = extract_layer_preview_paths(root, 0)

    assert [path.path_type for path in paths] == [
        PathType.TRAVEL,
        PathType.RESIN_PRINT,
    ]
    assert paths[1].start == pytest.approx((2.0, 0.0, 0.2))
    assert paths[1].end == pytest.approx((3.0, 0.0, 0.2))
