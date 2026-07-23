from pathlib import Path

import numpy as np
import pytest

from path_processing_core.npz_contract import (
    EVENT_TYPE_VOCAB,
    MOVE_TYPE_VOCAB,
    NpzContractError,
    SYSTEM_NPZ_CONTRACT_VERSION,
    SYSTEM_NPZ_DTYPES,
    SYSTEM_NPZ_ROW_FIELDS,
    detect_system_npz_contract,
    validate_system_npz_contract,
)


def _valid_arrays(row_count=2):
    arrays = {
        name: np.zeros(row_count, dtype=dtype)
        for name, dtype in SYSTEM_NPZ_DTYPES.items()
        if name in SYSTEM_NPZ_ROW_FIELDS
    }
    arrays["move_type_vocab_keys"] = np.asarray(
        list(MOVE_TYPE_VOCAB), dtype="S32"
    )
    arrays["move_type_vocab_vals"] = np.asarray(
        list(MOVE_TYPE_VOCAB.values()), dtype=np.uint8
    )
    arrays["event_type_vocab_keys"] = np.asarray(
        list(EVENT_TYPE_VOCAB), dtype="S32"
    )
    arrays["event_type_vocab_vals"] = np.asarray(
        list(EVENT_TYPE_VOCAB.values()), dtype=np.uint8
    )
    return arrays


def _offline_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "handoff" / "golden").is_dir():
            return parent
    raise AssertionError("offline planner handoff root not found")


def test_detects_and_validates_frozen_v1_arrays():
    arrays = _valid_arrays()

    assert detect_system_npz_contract(arrays) == SYSTEM_NPZ_CONTRACT_VERSION
    assert validate_system_npz_contract(arrays) == SYSTEM_NPZ_CONTRACT_VERSION


def test_validator_rejects_missing_or_changed_v1_fields():
    missing = _valid_arrays()
    del missing["planned_time_s"]
    assert detect_system_npz_contract(missing) is None
    with pytest.raises(NpzContractError, match="planned_time_s"):
        validate_system_npz_contract(missing)

    changed_dtype = _valid_arrays()
    changed_dtype["e"] = changed_dtype["e"].astype(np.float64)
    assert detect_system_npz_contract(changed_dtype) is None
    with pytest.raises(NpzContractError, match="dtype float32"):
        validate_system_npz_contract(changed_dtype)


def test_handoff_golden_archive_is_system_contract_v1():
    golden = _offline_root() / "handoff" / "golden" / "external-template-system.npz"

    assert validate_system_npz_contract(golden) == SYSTEM_NPZ_CONTRACT_VERSION
