"""Fiber lift is local to a resin layer and never changes the resin Z grid."""
import copy

import numpy as np
import pytest

from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths, TravelPaths
from kuka_slicer.ui_server import _ensure_offline_planner_import_paths, expand_fiber_template_for_resin_layers
from kuka_slicer.surface_preview.server import conformal_lattice_config_payload, planar_lattice_config_payload
from kuka_slicer.conformal_lattice.pipeline import run_conformal_lattice_pipeline
from kuka_slicer.conformal_lattice.path_bridge import ExtrusionVolumeModel
from kuka_slicer.conformal_lattice.fiber_reinforcement import (
    ContinuousCourseFiberSettings, apply_continuous_course_fiber_strategy,
)


@pytest.mark.parametrize('first_height', [.3, .4])
def test_prusa_fiber_attachment_keeps_final_npz_resin_grid_and_local_lift(tmp_path, first_height):
    _ensure_offline_planner_import_paths()
    from external_npz_preprocessor.source_npz import SourceJob, LayerPaths, MaterialPath
    from external_npz_preprocessor.source_gcode import with_fiber_paths
    from external_npz_preprocessor.converter import source_job_to_parsed_commands
    from external_npz_preprocessor.process_params import ProcessParams
    from path_processing_core.npz_exporter import export_npz

    heights = first_height + .5 * np.arange(6)
    paths = [np.array([[10., 10., z], [12., 10., z]]) for z in heights]
    job = ExternalSourceJob(
        material_paths=[MaterialPaths(i, 'R', [p.copy()]) for i, p in enumerate(paths)],
        travel_paths=[TravelPaths(i, [p.copy()]) for i, p in enumerate(paths)],
        meta={'slicing': {'z_max': float(heights[-1])}},
    )
    native = SourceJob(meta={'core_input_frame': 'print_plane'}, layers=[
        LayerPaths(index=i, resin_paths=[MaterialPath('R', 0, np.column_stack((p, np.zeros((len(p), 3)))))])
        for i, p in enumerate(paths)
    ])
    fibers = expand_fiber_template_for_resin_layers(job, [[[10., 10., 0.], [12., 10., 0.]]])
    for group, original in zip(job.material_paths, paths):
        np.testing.assert_array_equal(group.paths[0], original)
    for group, original in zip(job.travel_paths, paths):
        np.testing.assert_array_equal(group.paths[0], original)
    assert job.meta['slicing']['z_max'] == heights[-1]
    assert sorted(fibers) == [1, 2, 3, 4]  # Existing interface policy is unchanged.
    commands = source_job_to_parsed_commands(with_fiber_paths(native, fibers), ProcessParams(primeline_enabled=False))
    output = tmp_path/'height.npz'
    export_npz(commands, str(output), dt=.004)
    with np.load(output) as data:
        print_mask = np.isin(data['move_type'], [1, 3]) & (data['core_injection_role'] == 0) & (data['event_flag'] == 0)
        moving = np.r_[False, (np.abs(np.diff(data['x'])) + np.abs(np.diff(data['y']))) > 1e-7]
        for layer, z in enumerate(heights):
            resin = print_mask & moving & (data['layer_index'] == layer) & (data['tool_id'] == 2)
            assert resin.any()
            np.testing.assert_allclose(data['z'][resin], z, atol=1e-6)
            if layer in fibers:
                fiber = print_mask & moving & (data['layer_index'] == layer) & (data['tool_id'] == 1)
                assert fiber.any()
                np.testing.assert_allclose(data['z'][fiber], z + .1, atol=1e-6)


@pytest.mark.parametrize('planar', [False, True])
@pytest.mark.parametrize('preplanned', [False, True])
def test_honeycomb_fiber_keeps_all_resin_layers_and_uses_local_surface_z(planar, preplanned):
    factory = planar_lattice_config_payload if planar else conformal_lattice_config_payload
    config = factory({'part_length_mm': ['24'], 'part_width_mm': ['12'],
                      'part_height_mm': ['3'], 'wall_width_mm': ['2'],
                      'base_cell_size_mm': ['4'], 'surface_start_layer': ['1'],
                      'amplitude_mm': ['0.4'], 'samples_x': ['12'], 'samples_y': ['8']})
    runs = [run_conformal_lattice_pipeline(
        config, physical_layer_height_mm=.5, fiber_layer_height_mm=.1,
        plan_for_continuous_fiber=enabled, extrusion=ExtrusionVolumeModel(1., 1.),
    ) for enabled in [False, True]]
    plain, with_fiber = runs
    plan = with_fiber.layer_embedding.report['physical_stack_plan']
    assert plan['resin_layer_count'] == 6
    assert plan['planned_final_height_mm'] == 3.
    np.testing.assert_array_equal(with_fiber.layer_embedding.node_positions_xyz, plain.layer_embedding.node_positions_xyz)
    np.testing.assert_allclose(with_fiber.layer_embedding.report['base_z_by_layer_mm'], .25 + .5*np.arange(6))
    run = with_fiber if preplanned else plain
    base = run.path_graph.to_external_base_source_job()
    base.meta['physical_stack_plan'] = plan
    before, after = copy.deepcopy(base), copy.deepcopy(base)
    selected = tuple(plan['fiber_layer_indices'])
    assert len(selected) > 1
    settings = dict(first_after_resin_layer_physical=selected[0]+1,
                    last_after_resin_layer_physical=selected[-1]+1,
                    resin_z_preplanned_for_fiber=preplanned,
                    planned_resin_layer_indices=selected,
                    compensate_fiber_gap_extrusion=False)
    for enabled, job in [(False, before), (True, after)]:
        apply_continuous_course_fiber_strategy(
            source_job=job, graph=run.path_graph,
            course_paths_by_layer=run.continuous_course_paths_by_layer,
            settings=ContinuousCourseFiberSettings(enabled=enabled, **settings),
            fiber_layer_height_mm=.1, fiber_e_per_mm=1.,
        )
    resin_before = [g for g in before.material_paths if g.material == 'R']
    resin_after = [g for g in after.material_paths if g.material == 'R']
    for left, right in zip(resin_before, resin_after):
        assert len(left.paths) == len(right.paths)
        for a, b in zip(left.paths, right.paths):
            np.testing.assert_array_equal(a, b)  # XYZ, ABC and path ordering.
        for a, b in zip(left.extrusion, right.extrusion):
            np.testing.assert_array_equal(a, b)
    for group in before.travel_paths:
        current = next(g for g in after.travel_paths if g.layer_index == group.layer_index)
        for a, b in zip(group.paths, current.paths):
            np.testing.assert_array_equal(a, b)
    for group in (g for g in after.material_paths if g.material == 'F'):
        resin = next(g for g in resin_after if g.layer_index == group.layer_index)
        for fiber in group.paths:
            # Course ordering can reverse a path. Match unchanged XY, then
            # require the sole geometric difference to be +0.1 in global Z.
            candidates = [p for raw in resin.paths for p in (raw, raw[::-1])
                          if raw.shape == fiber.shape and np.allclose(p[:, :2], fiber[:, :2], atol=1e-8)]
            assert candidates
            assert any(np.allclose(fiber[:, 2]-p[:, 2], .1, atol=1e-8)
                       and np.allclose(fiber[:, 3:], p[:, 3:], atol=1e-8) for p in candidates)
