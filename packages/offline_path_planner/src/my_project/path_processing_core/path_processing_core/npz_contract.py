"""Version and validate the frozen system NPZ consumer contract."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


SYSTEM_NPZ_CONTRACT_NAME = "kuka_system_trajectory"
SYSTEM_NPZ_CONTRACT_VERSION = 1
SYSTEM_NPZ_CONTRACT_ID = (
    f"{SYSTEM_NPZ_CONTRACT_NAME}_v{SYSTEM_NPZ_CONTRACT_VERSION}"
)

SYSTEM_NPZ_DTYPES = {
    "seq": np.dtype(np.uint32),
    "x": np.dtype(np.float32),
    "y": np.dtype(np.float32),
    "z": np.dtype(np.float32),
    "a": np.dtype(np.float32),
    "b": np.dtype(np.float32),
    "c": np.dtype(np.float32),
    "e": np.dtype(np.float32),
    "tool_id": np.dtype(np.uint8),
    "move_type": np.dtype(np.uint8),
    "src_line": np.dtype("S32"),
    "event_flag": np.dtype(np.uint8),
    "event_type": np.dtype(np.uint8),
    "payload": np.dtype("S32"),
    "trigger_seq": np.dtype(np.int32),
    "layer_index": np.dtype(np.uint32),
    "total_layers": np.dtype(np.uint32),
    "preview_layer_index": np.dtype(np.int32),
    "path_id": np.dtype(np.uint32),
    "path_end_flag": np.dtype(np.uint8),
    "planned_time_s": np.dtype(np.float32),
    "move_type_vocab_keys": np.dtype("S32"),
    "move_type_vocab_vals": np.dtype(np.uint8),
    "event_type_vocab_keys": np.dtype("S32"),
    "event_type_vocab_vals": np.dtype(np.uint8),
}

SYSTEM_NPZ_ROW_FIELDS = (
    "seq",
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "e",
    "tool_id",
    "move_type",
    "src_line",
    "event_flag",
    "event_type",
    "payload",
    "trigger_seq",
    "layer_index",
    "total_layers",
    "preview_layer_index",
    "path_id",
    "path_end_flag",
    "planned_time_s",
)

MOVE_TYPE_VOCAB = {
    "TRAVEL": 0,
    "PRINT": 1,
    "TRAVEL_FIT": 2,
    "PRINT_FIT": 3,
    "EVENT": 4,
}

EVENT_TYPE_VOCAB = {
    "": 0,
    "heat_cf": 1,
    "heat_resin": 2,
    "fan_cf": 3,
    "fan_resin": 4,
    "extrude_reset": 5,
    "tool_change_cf": 6,
    "tool_change_resin": 7,
    "cut": 8,
}


class NpzContractError(ValueError):
    """Raised when a system NPZ does not satisfy a supported contract."""


def detect_system_npz_contract(arrays: Mapping[str, np.ndarray]) -> int | None:
    """Return version 1 for archives containing every frozen v1 field."""
    if set(SYSTEM_NPZ_DTYPES).issubset(arrays):
        return SYSTEM_NPZ_CONTRACT_VERSION
    return None


def validate_system_npz_contract(
    source: str | Path | Mapping[str, np.ndarray],
) -> int:
    """Validate a system NPZ path or loaded mapping and return its version."""
    if isinstance(source, (str, Path)):
        with np.load(Path(source), allow_pickle=False) as archive:
            return _validate_loaded_arrays(archive)
    return _validate_loaded_arrays(source)


def _validate_loaded_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    missing = sorted(set(SYSTEM_NPZ_DTYPES).difference(arrays))
    if missing:
        raise NpzContractError(
            "system NPZ is missing v1 fields: " + ", ".join(missing)
        )

    for name, expected_dtype in SYSTEM_NPZ_DTYPES.items():
        value = np.asarray(arrays[name])
        if value.ndim != 1:
            raise NpzContractError(
                f"system NPZ field {name!r} must be one-dimensional, "
                f"got shape {value.shape}"
            )
        if value.dtype != expected_dtype:
            raise NpzContractError(
                f"system NPZ field {name!r} must use dtype {expected_dtype}, "
                f"got {value.dtype}"
            )

    row_count = len(arrays["seq"])
    wrong_lengths = [
        f"{name}={len(arrays[name])}"
        for name in SYSTEM_NPZ_ROW_FIELDS
        if len(arrays[name]) != row_count
    ]
    if wrong_lengths:
        raise NpzContractError(
            f"system NPZ row fields must all contain {row_count} rows: "
            + ", ".join(wrong_lengths)
        )

    _validate_vocab(
        arrays,
        keys_name="move_type_vocab_keys",
        values_name="move_type_vocab_vals",
        expected=MOVE_TYPE_VOCAB,
    )
    _validate_vocab(
        arrays,
        keys_name="event_type_vocab_keys",
        values_name="event_type_vocab_vals",
        expected=EVENT_TYPE_VOCAB,
    )
    return SYSTEM_NPZ_CONTRACT_VERSION


def _validate_vocab(
    arrays: Mapping[str, np.ndarray],
    *,
    keys_name: str,
    values_name: str,
    expected: Mapping[str, int],
) -> None:
    raw_keys = np.asarray(arrays[keys_name])
    raw_values = np.asarray(arrays[values_name])
    keys = [value.decode("utf-8") for value in raw_keys.tolist()]
    values = [int(value) for value in raw_values.tolist()]
    actual = dict(zip(keys, values))
    if (
        len(keys) != len(values)
        or len(keys) != len(expected)
        or actual != dict(expected)
    ):
        raise NpzContractError(
            f"system NPZ vocabulary {keys_name!r}/{values_name!r} "
            f"must equal {dict(expected)!r}, got {actual!r}"
        )
