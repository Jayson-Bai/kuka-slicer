from __future__ import annotations

import json

import numpy as np

from kuka_slicer.conformal_lattice import generate_conformal_lattice_geometry, solve_phase_coordinates, validate_realized_fill_ratio
from tests.test_conformal_lattice_lattice_generator import _inputs
from tests.test_conformal_lattice_phase_coordinates import _inputs as _variable_inputs


def test_measured_wall_coverage_is_independent_from_target_and_reports_error_metrics():
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="inset")
    result = validate_realized_fill_ratio(domain, parameterization, fields, phase, geometry, samples_per_triangle_side=8)

    assert np.all(result.evaluation_mask)
    assert result.report["target_not_reused_as_realized"] is True
    assert result.report["measurement_model"].startswith("union_of_3d")
    assert result.report["mae"] > 0.0
    assert result.report["p95_absolute_error"] >= result.report["mae"]
    assert not np.allclose(result.target_fill_ratio_per_cell, result.realized_fill_ratio_per_cell)
    assert np.all(result.suggested_cell_scale_factor_per_cell > 1.0)


def test_geometry_npz_replaces_gate5_nan_placeholder_only_when_validation_is_supplied(tmp_path):
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="inset")
    validation = validate_realized_fill_ratio(domain, parameterization, fields, phase, geometry, samples_per_triangle_side=6)
    output = geometry.save_npz(
        tmp_path / "conformal_lattice_geometry_v1.npz",
        domain=domain,
        parameterization=parameterization,
        design_fields=fields,
        orientation=orientation,
        phase=phase,
        fill_validation=validation,
    )

    with np.load(output) as saved:
        np.testing.assert_allclose(saved["realized_fill_ratio_per_cell"], validation.realized_fill_ratio_per_cell)
        metadata = json.loads(str(saved["meta"]))
        assert metadata["fill_ratio_validation"]["evaluated_cell_count"] == len(validation.realized_fill_ratio_per_cell)


def test_variable_target_field_is_measured_as_a_per_cell_window_average():
    domain, parameterization, fields, orientation = _variable_inputs(eta=np.linspace(0.2, 0.6, 20))
    phase = solve_phase_coordinates(domain, parameterization, fields, orientation)
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="clip")
    validation = validate_realized_fill_ratio(domain, parameterization, fields, phase, geometry, samples_per_triangle_side=5)

    measured_targets = validation.target_fill_ratio_per_cell[validation.evaluation_mask]
    assert np.ptp(measured_targets) > 0.1
    assert fields.target_fill_ratio.min() <= measured_targets.min() <= measured_targets.max() <= fields.target_fill_ratio.max()
