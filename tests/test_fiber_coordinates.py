"""Coordinate contracts for imported fibers and print-plane part placement."""
import json

import numpy as np
import pytest

from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths, TravelPaths
from kuka_slicer.slicer import normalize_job_xy_origin
from kuka_slicer.stl_io import Mesh
from kuka_slicer.ui_server import (
    _ensure_offline_planner_import_paths,
    _part_xy_center,
    align_fiber_template_paths_to_resin,
    load_fiber_template_json,
)


@pytest.mark.parametrize("coordinate_system", [None, "project_default", "slicer_xy"])
@pytest.mark.parametrize("target", [(0., 0.), (95., 60.), (-20., 125.)])
def test_stl_frame_fibers_and_resin_receive_same_placement(tmp_path, coordinate_system, target):
    # Off-origin STL, partial fibers, and asymmetric Brim/raft extents must
    # not change the coordinate origin or infer an alignment from path bounds.
    stl_center = (45., 30.)
    fiber = np.array([[49.716594, 1., 0.], [60., 10., 0.]])
    payload = {"paths": [fiber[:, :2].tolist()]}
    if coordinate_system is not None:
        payload["coordinate_system"] = coordinate_system
    source = tmp_path / "fiber.json"
    source.write_text(json.dumps(payload))
    template = load_fiber_template_json(source)
    resin = np.array([[1., 1., .5], [89., 59., .5]])
    auxiliary = np.array([[-40., -8., .2], [100., 63., .2]])
    travel = np.array([[1., 1., .5], [50., 1., .5]])
    job = ExternalSourceJob(
        material_paths=[MaterialPaths(0, "R", [resin.copy(), auxiliary.copy()])],
        travel_paths=[TravelPaths(0, [travel.copy()])],
    )
    aligned = align_fiber_template_paths_to_resin(job, template, part_center_xy=stl_center)
    np.testing.assert_allclose(aligned[0], fiber)
    job.material_paths.append(MaterialPaths(0, "F", [np.array(aligned[0])]))
    delta = np.array(target) - stl_center
    translation = normalize_job_xy_origin(
        job, target_xy=target, reference_material="R", reference_xy=stl_center,
    )
    assert translation == pytest.approx(delta)
    for actual, original in [(job.material_paths[0].paths[0], resin),
                             (job.material_paths[0].paths[1], auxiliary),
                             (job.material_paths[1].paths[0], fiber),
                             (job.travel_paths[0].paths[0], travel)]:
        np.testing.assert_allclose(actual[:, :2], original[:, :2] + delta)
        np.testing.assert_array_equal(actual[:, 2], original[:, 2])


@pytest.mark.parametrize("axis,expected", [("z", (45., 30.)), ("y", (45., 2.5)), ("x", (30., 2.5))])
def test_part_reference_uses_oriented_stl_bounds(axis, expected):
    mesh = Mesh(np.array([[[0., 0., 0.], [90., 60., 0.], [90., 60., 5.]]]))
    assert _part_xy_center(mesh, axis) == pytest.approx(expected)


def test_centered_fiber_cannot_silently_use_resin_path_bounds(tmp_path):
    source = tmp_path / "fiber.json"
    source.write_text(json.dumps({"coordinate_system": "part_center_xy", "paths": [[[-2., -1.], [2., 1.]]]}))
    template = load_fiber_template_json(source)
    with pytest.raises(ValueError, match="oriented STL center"):
        align_fiber_template_paths_to_resin(ExternalSourceJob(material_paths=[]), template)


@pytest.mark.parametrize("target", [(95., 60.), (-20., 125.)])
def test_core_preserves_placed_materials_and_absolute_primeline(target):
    _ensure_offline_planner_import_paths()
    from external_npz_preprocessor.converter import source_job_to_parsed_commands
    from external_npz_preprocessor.process_params import ProcessParams
    from kuka_slicer.external_npz import external_source_job_to_core_source_job

    resin = np.array([[1., 1., .5], [89., 59., .5]])
    fiber = np.array([[10., 5., .6], [50., 20., .6]])
    job = ExternalSourceJob(
        material_paths=[MaterialPaths(0, "R", [resin.copy()]), MaterialPaths(0, "F", [fiber.copy()])],
    )
    normalize_job_xy_origin(job, target_xy=target, reference_xy=(45., 30.))
    job.meta["core_input_frame"] = "print_plane"
    commands = source_job_to_parsed_commands(
        external_source_job_to_core_source_job(job),
        ProcessParams(start_x_mm=target[0], start_y_mm=target[1],
                      primeline_x_mm=0., primeline_y_mm=-10., primeline_length_mm=80.),
    )
    deposits = [c for c in commands if c.raw == "external_npz_print_source"]
    assert len(deposits) == 2
    delta = np.array(target) - [45., 30.]
    for command, original in zip(deposits, [resin, fiber]):
        actual = [[command.start_pos.x, command.start_pos.y], [command.pos.x, command.pos.y]]
        np.testing.assert_allclose(actual, original[:, :2] + delta)
    primeline = [c for c in commands if c.raw == "external_npz_primeline"]
    assert len(primeline) == 1
    assert (primeline[0].start_pos.x, primeline[0].start_pos.y) == pytest.approx((0., -10.))
    assert (primeline[0].pos.x, primeline[0].pos.y) == pytest.approx((80., -10.))
