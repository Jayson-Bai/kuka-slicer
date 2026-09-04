from __future__ import annotations

import json

import numpy as np
import pytest

from kuka_slicer.conformal_lattice import (
    ExtrusionVolumeModel,
    load_conformal_lattice_spec,
    run_conformal_lattice_pipeline,
    write_conformal_lattice_outputs,
)
from kuka_slicer.conformal_lattice.contracts import double_sine_source_sha256
from kuka_slicer.surface_preview.server import conformal_lattice_config_payload


def _spec(*, surface_start_layer: int = 0):
    source = {
        "provider": "double_sine",
        "source_file": "generated://pipeline-test-double-sine",
        "domain": "outer_boundary_only",
        "reference_stl": {
            "file_name": "honeycomb.stl",
            "sha256": "a" * 64,
            "build_axis": "z",
            "xy_bounds_mm": [0.0, 0.0, 8.0, 7.0],
        },
        "double_sine": {
            "type": "double_sine_product",
            "amplitude_mm": 0.2,
            "wavelength_x_mm": 30.0,
            "wavelength_y_mm": 35.0,
            "phase_x_rad": 0.0,
            "phase_y_rad": 0.0,
            "z_reference_mm": 0.0,
            "xy_bounds_mm": [0.0, 0.0, 8.0, 7.0],
            "samples": [8, 7],
        },
    }
    source["sha256"] = double_sine_source_sha256(source)
    return load_conformal_lattice_spec(
        {
            "format": "conformal_lattice_spec_v1",
            "units": "mm",
            "source_surface": source,
            "parameterization": {
                "method": "lscm",
                "anchor_strategy": "farthest_boundary_pair",
                "seam_strategy": "none",
            },
            "lattice": {
                "family": "triangular_dual_hex",
                "wall_width_mm": 0.5,
                "base_cell_size_mm": 2.0,
                "boundary_mode": "inset",
                "phase_origin": [0.0, 0.0],
            },
            "fill_field": {"mode": "fixed_cell_size", "drivers": []},
            "orientation_field": {"mode": "global_axis", "angle_deg": 0.0, "constraints": []},
            "layer_embedding": {
                "mode": "symmetric_shape_morphing",
                "transition": "smoothstep",
                "surface_start_layer": surface_start_layer,
            },
            "quality_limits": {},
            "random_seed": 0,
        }
    )


def test_pipeline_runs_gates_one_to_eight_and_only_enables_paths_with_an_explicit_e_model(tmp_path):
    run = run_conformal_lattice_pipeline(
        _spec(),
        logical_layer_count=4,
        extrusion=ExtrusionVolumeModel(
            bead_cross_section_area_mm2=0.2,
            e_volume_per_unit_mm3=0.1,
            preview_line_width_mm=0.6,
        ),
        fill_samples_per_triangle_side=3,
    )

    assert run.design_fields.target_cell_size_mm == pytest.approx(
        np.full(len(run.domain.vertices), 2.0)
    )
    assert run.layer_embedding.report["alpha_by_layer"] == [0.0, 1.0, 1.0, 0.0]
    assert run.path_graph is not None
    assert run.report["path_npz_export_available"] is True
    assert run.preview_payload()["read_only"] is True
    main_preview = run.main_preview_payload(planning_line_width_mm=0.6)
    assert main_preview["preview_source"] == "conformal_lattice_external_source_job"
    assert main_preview["geometry_mode"] == "surface_3d"
    assert main_preview["line_widths"]["resin"] == pytest.approx(0.6)
    assert main_preview["conformal_lattice"]["uses_existing_main_canvas"] is True
    assert len(main_preview["layers"]) == 4
    assert all(0 < len(layer["resin_paths"]) < len(run.path_graph.edge_ids) for layer in main_preview["layers"])

    outputs = write_conformal_lattice_outputs(run, tmp_path)
    assert set(outputs) == {"geometry", "paths"}
    with np.load(outputs["geometry"], allow_pickle=False) as archive:
        assert archive["target_cell_size_mm_per_vertex"] == pytest.approx(2.0)
    with np.load(outputs["paths"], allow_pickle=False) as archive:
        metadata = json.loads(str(archive["meta"]))
    assert metadata["format"] == "external_layer_paths_v1"
    assert metadata["conformal_lattice_path_bridge"]["trail_partition_status"] == "planned_from_conformal_structural_graph"
    from kuka_slicer.ui_server import _preview_payload_from_source_npz

    loaded_preview = _preview_payload_from_source_npz(outputs["paths"].read_bytes(), outputs["paths"].name)
    assert loaded_preview["preview_source"] == "conformal_lattice_external_source_npz"
    assert loaded_preview["line_widths"]["resin"] == pytest.approx(0.6)
    assert loaded_preview["conformal_lattice"]["uses_existing_main_canvas"] is True


def test_pipeline_keeps_path_export_disabled_without_process_e_conversion(tmp_path):
    run = run_conformal_lattice_pipeline(_spec(), logical_layer_count=4, fill_samples_per_triangle_side=3)

    assert run.path_graph is None
    assert run.report["path_npz_export_available"] is False
    assert "bead_cross_section_area_mm2" in run.report["path_npz_requirement"]
    outputs = write_conformal_lattice_outputs(run, tmp_path)
    assert set(outputs) == {"geometry"}
    assert not (tmp_path / "external_layer_paths_v1.npz").exists()
    with pytest.raises(ValueError, match="path graph"):
        run.main_preview_payload(planning_line_width_mm=0.6)


@pytest.mark.parametrize(
    ("logical_layer_count", "surface_start_layer", "error"),
    [(0, 0, "logical_layer_count"), (4, 3, "surface_start_layer")],
)
def test_pipeline_rejects_invalid_logical_layer_progression(logical_layer_count, surface_start_layer, error):
    with pytest.raises(ValueError, match=error):
        run_conformal_lattice_pipeline(_spec(surface_start_layer=surface_start_layer), logical_layer_count=logical_layer_count)


def test_rectangular_physical_part_derives_monotonic_layer_centres_and_requires_the_matching_count():
    spec = conformal_lattice_config_payload(
        {
            "part_length_mm": ["8"], "part_width_mm": ["7"], "part_height_mm": ["10"],
            "layer_height_mm": ["2"], "samples_x": ["8"], "samples_y": ["7"],
            "wall_width_mm": ["2"], "base_cell_size_mm": ["5"], "surface_start_layer": ["1"],
        }
    )

    run = run_conformal_lattice_pipeline(spec, extrusion=ExtrusionVolumeModel(0.2, 0.1))

    assert run.layer_embedding.report["base_z_by_layer_mm"] == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0])
    assert run.layer_embedding.report["surface_start_layer"] == 1
    with pytest.raises(ValueError, match="logical_layer_count"):
        run_conformal_lattice_pipeline(spec, logical_layer_count=4)
