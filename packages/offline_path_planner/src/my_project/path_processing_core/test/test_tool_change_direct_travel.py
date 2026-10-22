"""Tool changes lift, switch, then bridge directly to the next path."""

import json

import numpy as np
import pytest

from path_processing_core.npz_exporter import export_npz
from path_processing_core.types import (
    ExtrudeWait,
    MCommand,
    MoveCommand,
    Position,
    ResetECommand,
    ToolChangeCommand,
)


def _pose(x, y, z=0.5):
    return Position(x, y, z, 0.0, 0.0, 0.0)


def _move(kind, start, end, line, layer, subtype, raw=""):
    return MoveCommand(
        kind,
        "G0" if kind == "TRAVEL" else "G1",
        start,
        end,
        2.0 if kind == "PRINT" else 0.0,
        2.0 if kind == "PRINT" else 0.0,
        600.0,
        line,
        layer=layer,
        subtype=subtype,
        raw=raw,
    )


def _commands():
    return [
        _move("TRAVEL", _pose(0, 0), _pose(0, 0), 0, 0, "TRAVEL"),
        _move("PRINT", _pose(0, 0), _pose(10, 0), 1, 0, "RESIN_PRINT"),
        ToolChangeCommand("TOOL_CHANGE", 0, 2, layer=0, subtype="FIBER_PRINT"),
        ResetECommand("RESET_E", 0.0, 3, layer=0, subtype="FIBER_PRINT"),
        ExtrudeWait("EXTRUDE_WAIT", 0.04, 0.0, 600.0, 4),
        ExtrudeWait("EXTRUDE_WAIT", 0.08, 0.0, 600.0, 5),
        _move(
            "TRAVEL",
            _pose(10, 0),
            _pose(20, 5, 0.6),
            6,
            0,
            "TRAVEL",
            raw="external_npz_start_xy_travel",
        ),
        _move("PRINT", _pose(20, 5, 0.6), _pose(30, 5, 0.6), 7, 0, "FIBER_PRINT"),
        ToolChangeCommand("TOOL_CHANGE", 1, 8, layer=1, subtype="RESIN_PRINT"),
        ResetECommand("RESET_E", 0.0, 9, layer=1, subtype="RESIN_PRINT"),
        _move(
            "TRAVEL",
            _pose(30, 5, 0.6),
            _pose(0, 10, 1.0),
            10,
            1,
            "TRAVEL",
            raw="external_npz_travel",
        ),
        _move("PRINT", _pose(0, 10, 1.0), _pose(10, 10, 1.0), 11, 1, "RESIN_PRINT"),
    ]


def _read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _vocab(data, name):
    return {
        int(value): key.decode("utf-8").rstrip("\x00")
        for key, value in zip(
            data[f"{name}_vocab_keys"],
            data[f"{name}_vocab_vals"],
        )
    }


def _xyz(data, indices):
    return np.column_stack(
        [data[key][indices].astype(np.float64) for key in ("x", "y", "z")]
    )


@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (3.0, 4.0, 2.0)])
def test_both_tool_change_directions_use_one_direct_bridge(tmp_path, offset):
    output = tmp_path / "direct.npz"
    export_npz(
        _commands(),
        str(output),
        tool_offset=offset,
        tool_change_safe_lift_mm=10.0,
        enable_extrude_wait=True,
    )
    data = _read(output)
    roles = _vocab(data, "core_injection_role")
    events = _vocab(data, "event_type")
    role_names = np.asarray([roles[int(value)] for value in data["core_injection_role"]])
    event_names = np.asarray([
        events[int(value)] if int(data["event_flag"][i]) else ""
        for i, value in enumerate(data["event_type"])
    ])

    manifest = json.loads(str(data["core_injection_manifest"].item()))
    tool_blocks = [
        block for block in manifest["blocks"]
        if block.get("kind") == "tool_change"
    ]
    assert [block["to_tool"] for block in tool_blocks] == [1, 2]

    expected_destinations = {
        1: np.asarray((20.0, 5.0, 0.6)) + np.asarray(offset),
        2: np.asarray((0.0, 10.0, 1.0)),
    }
    raw_origins = (
        np.asarray((10.0, 0.0, 0.5)),
        np.asarray((30.0, 5.0, 0.6)) + np.asarray(offset),
    )
    horizontal_targets = (
        raw_origins[0] + np.asarray((offset[0], offset[1], 0.0)),
        raw_origins[1] - np.asarray((offset[0], offset[1], 0.0)),
    )
    for block, raw_endpoint, horizontal, destinations in zip(
        tool_blocks,
        raw_origins,
        horizontal_targets,
        (expected_destinations[1], expected_destinations[2]),
    ):
        block_mask = data["core_injection_block_id"] == int(block["id"])
        event_indices = np.flatnonzero(
            block_mask
            & (data["event_flag"] != 0)
            & np.asarray([
                event_names[i] in {"tool_change_cf", "tool_change_resin"}
                for i in range(len(event_names))
            ])
        )
        assert len(event_indices) == 1
        event_index = int(event_indices[0])

        pre_indices = np.flatnonzero(block_mask & (role_names == "tool_change_pre"))
        pre_xyz = _xyz(data, pre_indices)
        assert np.allclose(pre_xyz[0], raw_endpoint, atol=1e-9)
        assert np.allclose(
            pre_xyz[np.isclose(pre_xyz[:, 2], raw_endpoint[2], atol=1e-9)][-1],
            horizontal,
            atol=1e-6,
        )
        lifted = horizontal + np.asarray((0.0, 0.0, 10.0))
        assert np.allclose(pre_xyz[-1], lifted, atol=1e-6)
        assert np.all(np.diff(pre_xyz[:, 2]) >= -1e-9)
        assert np.allclose(_xyz(data, [event_index])[0], pre_xyz[-1], atol=1e-9)

        bridge_indices = np.flatnonzero(block_mask & (role_names == "tool_change_bridge"))
        bridge_xyz = _xyz(data, bridge_indices)
        assert len(bridge_indices) >= 2
        assert np.allclose(bridge_xyz[0], _xyz(data, [event_index])[0], atol=1e-9)
        assert np.allclose(
            bridge_xyz[-1],
            destinations,
            atol=1e-6,
        )
        assert np.max(np.linalg.norm(np.diff(bridge_xyz, axis=0), axis=1)) < 0.05
        assert not np.any(
            np.all(
                np.isclose(
                    _xyz(data, np.arange(event_index + 1, len(data["seq"]))),
                    raw_endpoint,
                    atol=1e-6,
                ),
                axis=1,
            )
        )

    assert np.max(np.linalg.norm(np.diff(_xyz(data, np.arange(len(data["seq"]))), axis=0), axis=1)) < 0.05


def test_cut_settle_parameter_controls_both_terminal_holds(tmp_path):
    commands = [
        _move("TRAVEL", _pose(0, 0), _pose(0, 0), 0, 0, "TRAVEL"),
        _move("PRINT", _pose(0, 0), _pose(10, 0), 1, 0, "FIBER_PRINT"),
        MCommand(
            type="M_COMMAND",
            code="CUT",
            params={"P": 1.0},
            line=2,
            layer=0,
            subtype="FIBER_PRINT",
            raw="external_npz_cut",
        ),
    ]
    timings = {}
    for settle in (0.25, 1.0):
        output = tmp_path / f"cut_{settle}.npz"
        result = export_npz(
            commands,
            str(output),
            initial_tool_id=1,
            cut_lift_mm=2.0,
            cut_wait_s=0.0,
            cut_settle_s=settle,
            external_npz_cut_absolute_e=True,
        )
        timings[settle] = float(result["planned_total_time_s"])
        data = _read(output)
        manifest = json.loads(str(data["core_injection_manifest"].item()))
        assert manifest["base_parameters"]["cut_settle_s"] == pytest.approx(settle)
        cut_block = next(
            block for block in manifest["blocks"] if block.get("kind") == "cut"
        )
        assert "cut_settle_s" in cut_block["injectable_parameters"]

    assert timings[1.0] - timings[0.25] == pytest.approx(1.5, abs=0.01)


def test_cut_emits_two_blocking_absolute_e_release_barriers(tmp_path):
    commands = [
        _move("TRAVEL", _pose(0, 0), _pose(0, 0), 0, 0, "TRAVEL"),
        _move("PRINT", _pose(0, 0), _pose(10, 0), 1, 0, "FIBER_PRINT"),
        MCommand(
            type="M_COMMAND",
            code="CUT",
            params={"P": 1.0},
            line=2,
            layer=0,
            subtype="FIBER_PRINT",
            raw="external_npz_cut",
        ),
    ]
    output = tmp_path / "cut_sync.npz"
    export_npz(
        commands,
        str(output),
        initial_tool_id=1,
        cut_lift_mm=2.0,
        cut_wait_s=0.0,
        cut_settle_s=0.25,
        external_npz_cut_absolute_e=True,
    )

    data = _read(output)
    events = _vocab(data, "event_type")
    event_names = np.asarray([
        events[int(value)] if int(data["event_flag"][index]) else ""
        for index, value in enumerate(data["event_type"])
    ])
    cut_indices = np.flatnonzero(event_names == "cut")
    sync_indices = np.flatnonzero(event_names == "extrude_sync")

    assert len(cut_indices) == 1
    assert len(sync_indices) == 2
    cut_index = int(cut_indices[0])
    assert bytes(data["payload"][cut_index]).split(b"\x00", 1)[0] == b"1 sync_v1:2"
    assert np.all(data["core_injection_block_id"][sync_indices]
                  == data["core_injection_block_id"][cut_index])
    assert np.array_equal(data["trigger_seq"][sync_indices],
                          data["seq"][sync_indices])

    sync_payloads = [
        bytes(data["payload"][index]).split(b"\x00", 1)[0].decode("ascii").split()
        for index in sync_indices
    ]
    assert [int(fields[0]) for fields in sync_payloads] == [1, 1]
    assert [float(fields[1]) for fields in sync_payloads] == [2.0, -2.0]
    assert [int(fields[2]) for fields in sync_payloads] == [30000, 30000]
    assert np.allclose(data["z"][sync_indices], 2.5)
    assert np.allclose(data["e"][sync_indices], [2.0, -2.0])
