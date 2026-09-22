from __future__ import annotations

import numpy as np

from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths
from kuka_slicer.fiber_interlayers import (
    FLAT_RESIN_INTERLAYER_POLICY_SOURCE,
    plan_flat_resin_interlayers,
)
from kuka_slicer.ui_server import expand_fiber_template_for_resin_layers


def test_flat_interlayer_policy_keeps_top_cap_and_can_skip_process_prefix():
    schedule = plan_flat_resin_interlayers(
        [3, 4, 5, 6],
        skip_initial_interfaces=1,
    )

    assert schedule.after_resin_layer_indices == (4, 5)
    assert schedule.physical_interface_window == (5, 6)
    assert [
        schedule.insertion_count_before(index) for index in (3, 4, 5, 6)
    ] == [0, 0, 1, 2]


def test_ordinary_flat_slicer_records_the_shared_interlayer_policy():
    resin_paths = [
        MaterialPaths(
            index,
            "R",
            [np.asarray([[0.0, 0.0, z], [1.0, 0.0, z]], dtype=np.float64)],
        )
        for index, z in enumerate((0.4, 0.9, 1.4))
    ]
    job = ExternalSourceJob(
        material_paths=resin_paths,
        meta={"slicing": {"z_max": 1.4}},
    )

    paths = expand_fiber_template_for_resin_layers(
        job,
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
    )

    assert sorted(paths) == [1]
    assert job.meta["slicing"]["fiber_initial_resin_only_layer_count"] == 1
    assert job.meta["slicing"]["fiber_layer_interface_policy"] == (
        FLAT_RESIN_INTERLAYER_POLICY_SOURCE
    )
