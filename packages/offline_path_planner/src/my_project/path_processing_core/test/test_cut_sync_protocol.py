import math

import pytest

from path_processing_core.cut_sync import (
    cut_holds,
    cut_sync_payload,
    sync_payload,
)


def test_cut_holds_are_independent_and_default_to_three_seconds():
    assert cut_holds(3.0, None, None) == (3.0, 3.0)
    assert cut_holds(1.5, 4.0, 5.0) == (4.0, 5.0)


def test_cut_holds_reject_invalid_values():
    with pytest.raises(ValueError):
        cut_holds(-1.0, None, None)
    with pytest.raises(ValueError):
        cut_holds(3.0, None, math.inf)


def test_cut_event_declares_cut_sync_v1_and_two_members():
    assert cut_sync_payload("1", 2) == "1 sync_v1:2"


def test_cut_event_declaration_is_idempotent_and_rejects_conflicts():
    assert cut_sync_payload("1 sync_v1:2", 2) == "1 sync_v1:2"
    with pytest.raises(ValueError):
        cut_sync_payload("1 sync_v1:1", 2)


def test_sync_timeout_is_a_floor_even_when_rqt_wait_is_short():
    assert sync_payload(1, 20.0, 0.0).endswith(" 30000")
    assert sync_payload(2, -20.0, 45.0).endswith(" 55000")


def test_sync_timeout_is_capped():
    assert sync_payload(1, 0.0, 10000.0).endswith(" 600000")
