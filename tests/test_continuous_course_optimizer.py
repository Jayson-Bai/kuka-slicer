"""Optimal safe travel, unchanged strokes and stable process boundaries."""

from itertools import permutations, product
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, box

from kuka_slicer.conformal_lattice.continuous_course import build_continuous_course_plan
from kuka_slicer.conformal_lattice.course_optimizer import _solve_cached, optimize_block, route_length
from kuka_slicer.conformal_lattice.fiber_reinforcement import (
    ContinuousCourseFiberSettings,
    apply_continuous_course_fiber_strategy,
    _continuous_boundary_router,
    _continuous_course_sequence,
    _optimize_continuous_records,
    _corner_start_outer_record,
)
from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths, external_source_job_to_core_source_job


def _fixture(edge=12):
    spec = SimpleNamespace(
        part={"length_mm": 150., "width_mm": 60.},
        lattice={"base_cell_size_mm": edge, "wall_width_mm": 2.},
    )
    plan = build_continuous_course_plan(spec)
    records = []
    for p in plan.paths_xy:
        poses = np.c_[p, np.ones(len(p)), np.zeros((len(p), 3))]
        # Uneven E increments also test the direction transform.
        e = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1) * .31)]
        records.append((poses, e, "conformal_continuous_course_fragment"))
    ordered = [r for g in _continuous_course_sequence(records, current_xy=np.zeros(2)) for r in g]
    ring = np.c_[np.array([[0, 0], [150, 0], [150, 60], [0, 60], [0, 0]], float),
                 np.ones(5), np.zeros((5, 3))]
    return ordered, _continuous_boundary_router((0, 0, 150, 60), [ring])


@pytest.mark.parametrize("edge", [9, 10, 11, 12])
def test_boundary_entry_preserves_strokes_and_shortens_total_safe_travel(edge):
    before, router = _fixture(edge)
    after, reports = _optimize_continuous_records(before, router)
    report = reports[0]
    assert report["optimal"]
    assert report["after_mm"] <= report["before_mm"]
    assert len(after) == len(before)
    remaining = list(before)
    for path, profile, role in after:
        matches = [(i, p, e) for i, (p, e, r) in enumerate(remaining)
                   if r == role and (np.array_equal(p, path) or np.array_equal(p[::-1], path))]
        assert len(matches) == 1
        index, original, extrusion = matches[0]
        expected = extrusion if np.array_equal(original, path) else extrusion[-1] - extrusion[::-1]
        np.testing.assert_allclose(profile, expected, rtol=0., atol=1e-13)
        assert profile[0] == 0 and np.all(np.diff(profile) >= 0)
        remaining.pop(index)
    assert not remaining
    for a, b in zip(after, after[1:]):
        route = router.route(a[0][-1], b[0][0])
        assert not LineString(route[:, :2]).intersects(box(1e-6, 1e-6, 150 - 1e-6, 60 - 1e-6))
    # From the lower-left contour seam the corner-adjacent candidate must be
    # the nearest reachable endpoint, including small bottom-edge fragments.
    entry = np.array([0., 0., 1., 0., 0., 0.])
    nearest = min(route_length(router.route(entry, p[k])) for p, _, _ in before for k in [0, -1])
    assert route_length(router.route(entry, after[0][0][0])) == pytest.approx(nearest)
    if edge == 12:
        np.testing.assert_allclose(after[0][0][0, :2], [0., 4.215390309173472])
        assert not np.array_equal(after[0][0], before[0][0])


def test_optimal_cost_matches_independent_exhaustive_permutation_and_direction_search():
    records, router = _fixture()
    paths = [records[i][0] for i in [0, 2, 4, 8, 10, 13]]
    visits, report = optimize_block(paths, router)
    n = len(paths)
    costs = {}
    for i, p in enumerate(paths):
        for j, q in enumerate(paths):
            if i != j:
                for a, b in product((False, True), repeat=2):
                    costs[(i, a), (j, b)] = route_length(router.route(p[0 if a else -1], q[-1 if b else 0]))
    best = float("inf")
    for order in permutations(range(1, n - 1)):
        for directions in product((False, True), repeat=n - 2):
            candidate = [(0, False), *zip(order, directions), (n - 1, False)]
            best = min(best, sum(costs[a, b] for a, b in zip(candidate, candidate[1:])))
    assert report["optimal"]
    assert report["after_mm"] == pytest.approx(best, abs=1e-10)
    assert visits[0] == (0, False) and visits[-1] == (n - 1, False)


@pytest.mark.parametrize("peak", [2., 20.])
def test_router_uses_3d_length_and_preserves_curved_boundary_samples(peak):
    ring = np.array([[0, 0, 0, 0, 0, 0], [5, 0, peak, 0, 12, 0],
                     [10, 0, 0, 0, 0, 0], [10, 10, 0, 0, 0, 0],
                     [0, 10, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], float)
    router = _continuous_boundary_router((0, 0, 10, 10), [ring])
    route = router.route(ring[0], ring[2])
    assert route_length(route) == pytest.approx(min(2 * np.hypot(5, peak), 30))
    if peak == 2:
        np.testing.assert_array_equal(route, ring[:3])
    else:
        assert np.all(route[:, 2] == 0)
        assert np.max(route[:, 1]) == 10
    assert not LineString(route[:, :2]).intersects(box(.001, .001, 9.999, 9.999))


def test_opposite_boundary_endpoints_never_take_a_through_pore_chord():
    _, router = _fixture()
    start = np.array([75., 0., 1., 0., 0., 0.])
    end = np.array([75., 60., 1., 0., 0., 0.])
    route = router.route(start, end)
    assert route_length(route) == pytest.approx(210.)  # Straight chord would be 60.
    assert not LineString(route[:, :2]).intersects(box(.001, .001, 149.999, 59.999))
    with pytest.raises(ValueError, match="hole-safe boundary approach"):
        router.route(np.array([75., 30., 1., 0., 0., 0.]), end)


def test_noncourse_process_records_and_block_membership_are_unchanged():
    records, router = _fixture()
    fixed = (records[0][0].copy(), records[0][1].copy(), "conformal_partition_wall")
    original = [fixed, *records[:9], fixed, *records[9:]]
    result, reports = _optimize_continuous_records(original, router)
    assert result[0] is fixed and result[10] is fixed
    for start, end in [(1, 10), (11, 16)]:
        canonical = lambda p: min(p.tobytes(), p[::-1].tobytes())
        assert sorted(canonical(r[0]) for r in result[start:end]) == sorted(canonical(r[0]) for r in original[start:end])
    assert len(reports) == 2 and all(report["optimal"] for report in reports)


@pytest.mark.parametrize("entry_xy", [(0., 0.), (150., 0.), (0., 60.)])
def test_boundary_start_and_free_exit_match_exhaustive_search(entry_xy):
    records, router = _fixture()
    paths = [records[i][0] for i in [0, 2, 4, 8, 10, 13]]
    entry = np.array([*entry_xy, 1., 0., 0., 0.])
    exit = np.array([150., 60., 1., 0., 0., 0.])
    visits, report = optimize_block(paths, router, entry=entry, exit=exit)
    states = list(product(range(len(paths)), (False, True)))
    first_points = {s: paths[s[0]][-1 if s[1] else 0] for s in states}
    last_points = {s: paths[s[0]][0 if s[1] else -1] for s in states}
    distance = lambda p, q: route_length(router.route(p, q))
    incoming = {s: distance(entry, first_points[s]) for s in states}
    outgoing = {s: distance(last_points[s], exit) for s in states}
    candidates = set()
    for x, y in [(0., 0.), (150., 0.), (150., 60.), (0., 60.)]:
        dist = {s: distance(np.array([x, y, 1., 0., 0., 0.]), first_points[s]) for s in states}
        candidates.update(s for s in states if dist[s] <= min(dist.values()) + 1e-7)
    first = min(candidates, key=lambda s: (incoming[s], s))
    assert visits[0] == first
    costs = {(a, b): distance(last_points[a], first_points[b]) for a in states for b in states if a[0] != b[0]}
    best = float('inf')
    for order in permutations(i for i in range(len(paths)) if i != first[0]):
        for directions in product((False, True), repeat=len(paths) - 1):
            tour = [first, *zip(order, directions)]
            best = min(best, incoming[first] + sum(costs[a, b] for a, b in zip(tour, tour[1:])) + outgoing[tour[-1]])
    assert report['optimal']
    assert report['after_mm'] == pytest.approx(best, abs=1e-9)


def test_outer_seam_rotation_preserves_closed_geometry_abc_and_segment_extrusion():
    path = np.array([[5, 0, 2, 0, 2, 0], [10, 0, 1, 0, 3, 0],
                     [10, 10, 4, 0, 4, 0], [0, 10, 3, 0, 5, 0],
                     [0, 0, 1, 0, 1, 0], [5, 0, 2, 0, 2, 0]], float)
    e = np.array([0., 1., 3., 7., 8., 9.])
    rotated, extrusion, role = _corner_start_outer_record((path, e, 'conformal_outer_boundary'))
    np.testing.assert_array_equal(rotated[0], path[4])
    np.testing.assert_array_equal(rotated[0], rotated[-1])
    assert role == 'conformal_outer_boundary' and len(rotated) == len(path)
    edges = lambda p, profile: {(a.tobytes(), b.tobytes()): de for a, b, de in zip(p, p[1:], np.diff(profile))}
    assert edges(rotated, extrusion) == edges(path, e)
    assert extrusion[-1] == e[-1]


def test_single_course_can_reverse_to_boundary_entry_and_include_exit():
    records, router = _fixture()
    path = records[0][0]
    visits, report = optimize_block([path], router, entry=path[-1], exit=path[0])
    assert visits == [(0, True)]
    assert report['after_mm'] == 0. and report['optimal']


def test_large_solver_budget_reports_approximation_and_never_worsens_baseline():
    count = 21
    rng = np.random.default_rng(6)
    costs = rng.uniform(.1, 100., (count * 2, count * 2))
    order, exact = _solve_cached(count, costs.tobytes())
    assert not exact
    assert order[0] == 0 and order[-1] == 2 * (count - 1)
    assert sorted(state // 2 for state in order) == list(range(count))
    baseline = sum(costs[2 * i, 2 * (i + 1)] for i in range(count - 1))
    assert sum(costs[a, b] for a, b in zip(order, order[1:])) <= baseline
    exits = rng.uniform(.1, 100., count * 2)
    order, exact = _solve_cached(count, costs.tobytes(), 7, exits.tobytes(), False)
    baseline = [7, *(2 * i for i in range(count) if i != 3)]
    total = lambda tour: sum(costs[a, b] for a, b in zip(tour, tour[1:])) + exits[tour[-1]]
    assert not exact and order[0] == 7 and sorted(s // 2 for s in order) == list(range(count))
    assert total(order) <= total(baseline)


def test_optimized_resin_and_fiber_share_visits_with_valid_core_travel_indexes():
    records, _ = _fixture()
    metadata = {"config": {
        "extrusion": {"bead_cross_section_area_mm2": 1., "e_volume_per_unit_mm3": 1.},
        "additional": {"part": {"length_mm": 150., "width_mm": 60.}},
    }}
    graph = SimpleNamespace(metadata=metadata)
    layers, groups = [], []
    for layer in range(3):
        courses = []
        for path, _, _ in records:
            xyz = path[:, :3].copy()
            x, y = xyz[:, 0] * np.pi / 150, xyz[:, 1] * np.pi / 60
            xyz[:, 2] = .25 + .5 * layer + .1 * np.sin(x) * np.sin(y)
            normals = np.c_[-.1 * np.pi / 150 * np.cos(x) * np.sin(y),
                            -.1 * np.pi / 60 * np.sin(x) * np.cos(y), np.ones(len(xyz))]
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            courses.append((xyz, normals))
        layers.append(tuple(courses))
        ring = np.c_[np.array([[0, 0], [150, 0], [150, 60], [0, 60], [0, 0]], float),
                     np.full(5, .25 + .5 * layer), np.zeros((5, 3))]
        groups.append(MaterialPaths(layer, "R", [ring], [np.array([0., 150., 210., 360., 420.])]))
    job = ExternalSourceJob(material_paths=groups, travel_paths=[], meta={
        "path_roles": {"R": {str(i): ["conformal_outer_boundary"] for i in range(3)}},
    })
    result = apply_continuous_course_fiber_strategy(
        source_job=job, graph=graph, course_paths_by_layer=tuple(layers),
        settings=ContinuousCourseFiberSettings(True, 1, 2),
        fiber_layer_height_mm=.1, fiber_e_per_mm=.7,
    )
    assert result.total_path_count == 28
    for layer in result.resin_layer_indices:
        resin = next(g for g in job.material_paths if g.material == "R" and g.layer_index == layer)
        fiber = next(g for g in job.material_paths if g.material == "F" and g.layer_index == layer)
        for r, f in zip(resin.paths[1:], fiber.paths):
            np.testing.assert_array_equal(r[:, [0, 1, 3, 4, 5]], f[:, [0, 1, 3, 4, 5]])
            np.testing.assert_allclose(f[:, 2] - r[:, 2], .1, atol=1e-15)
        travel = next(g.paths for g in job.travel_paths if g.layer_index == layer)
        records = job.meta["motion_order"][str(layer)]
        pending = None
        for record in records:
            if record["kind"] in ("travel", "fiber_travel"):
                pending = travel[record["index"]]
            elif pending is not None:
                paths = resin.paths if record["kind"] == "deposit" else fiber.paths
                index = record["index"]
                np.testing.assert_array_equal(pending[0], paths[index - 1][-1])
                np.testing.assert_array_equal(pending[-1], paths[index][0])
                pending = None
    from kuka_slicer.ui_server import _ensure_offline_planner_import_paths
    _ensure_offline_planner_import_paths()
    core = external_source_job_to_core_source_job(job)
    assert len(core.layers) == 3
    for layer in core.layers:
        expected = next(g.paths for g in job.material_paths if g.material == "R" and g.layer_index == layer.index)
        for path, reference in zip(layer.resin_paths, expected):
            np.testing.assert_array_equal(path.points, reference)
