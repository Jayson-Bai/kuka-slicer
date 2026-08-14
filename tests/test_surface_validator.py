from __future__ import annotations

import json

import numpy as np

from kuka_slicer.cli import main
from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths, TravelPaths, write_external_source_npz
from kuka_slicer.extrusion_compensator import compensate_extrusion
from kuka_slicer.surface_mapper import LayerProgression, SurfaceMappingPlan, load_surface_target, map_source_job, read_source_npz
from kuka_slicer.surface_validator import render_html_report, validate_surface_job


def _target():
    return load_surface_target(
        {
            "format": "graded_surface_v1",
            "units": "mm",
            "coordinate_system": {"plane": "XY", "build_axis": "Z", "origin": "stl_xy_min"},
            "domain": {"source": {"file_name": "honeycomb.stl", "sha256": "same-model", "xy_bounds_mm": [0, 0, 20, 20]}},
            "surface": {
                "type": "double_sine_product",
                "amplitude_mm": 0.2,
                "wavelength_x_mm": 20.0,
                "wavelength_y_mm": 20.0,
                "phase_x_rad": 0.0,
                "phase_y_rad": 0.0,
                "z_reference_mm": 0.0,
            },
        }
    )


def _flat_source(tmp_path, *, include_travel: bool = False):
    output = tmp_path / "flat.npz"
    material_paths = []
    for layer in range(4):
        z = 0.5 + layer * 0.5
        material_paths.append(
            MaterialPaths(
                layer,
                "R",
                [np.asarray([[5.0, 5.0, z], [6.0, 5.0, z], [5.0, 5.0, z]], dtype=np.float64)],
                extrusion=[np.asarray([float(layer), float(layer + 1), float(layer + 2)])],
            )
        )
    travel_paths = [
        TravelPaths(1, [np.asarray([[5.0, 5.0, 1.0], [120.0, 5.0, 1.0]], dtype=np.float64)])
    ] if include_travel else []
    write_external_source_npz(
        ExternalSourceJob(
            material_paths=material_paths,
            travel_paths=travel_paths,
            meta={"source_model": {"file_name": "honeycomb.stl", "sha256": "same-model"}},
        ),
        output,
    )
    return read_source_npz(output.read_bytes(), source_name=output.name)


def _curved_source(flat):
    result = map_source_job(flat, _target(), SurfaceMappingPlan(LayerProgression(0, 3)))
    return compensate_extrusion(result.flat_source, result.source).source


def _check(report, name):
    return next(check for check in report.checks if check.name == name)


def test_validator_reports_mapped_geometry_e_and_unassessed_topology(tmp_path):
    flat = _flat_source(tmp_path)
    report = validate_surface_job(flat, _curved_source(flat), _target())

    assert _check(report, "路径契约与 Z 安全").status == "pass"
    assert _check(report, "逐点 Z 映射").status == "pass"
    assert _check(report, "E 弧长补偿").status == "pass"
    assert _check(report, "T 空走风险").status == "pass"
    assert _check(report, "XY 拓扑与边界").status == "warning"
    assert report.payload()["geometry"]["actual_max_height_mm"] > 0
    assert "曲面路径可打印性验证报告" in render_html_report(report)


def test_validator_reports_honeycomb_contract_without_claiming_stl_geometry_proof(tmp_path):
    flat = _flat_source(tmp_path)
    flat.meta["path_roles"] = {"R": {str(layer): ["outer_contour"] for layer in range(4)}}
    flat.meta["honeycomb_centerline_pathing"] = {
        "format": "honeycomb_macro_partition_zero_e_v1",
        "topology": "stl_honeycomb_macro_partition_zero_e",
        "topology_change": "none; wall graph is derived from the source STL section",
        "wall_edge_count": 693,
        "repeated_wall_edge_count": 0,
    }

    report = validate_surface_job(flat, _curved_source(flat), _target())
    topology = _check(report, "XY 拓扑与边界")

    assert topology.status == "warning"
    assert "生产蜂窝元数据" in topology.summary
    assert "输入不含 STL 实体" in topology.summary
    assert topology.details["wall_edge_count"] == 693
    assert topology.details["repeated_wall_edge_count"] == 0


def test_validator_rejects_changed_xy_and_reports_t_risk(tmp_path):
    flat = _flat_source(tmp_path, include_travel=True)
    curved = _curved_source(flat)
    curved.arrays["layer_0001_R"][0, 1, 0] += 0.01

    report = validate_surface_job(flat, curved, _target())

    assert _check(report, "路径契约与 Z 安全").status == "fail"
    assert _check(report, "坡度、dz 与采样密度").status == "pass"
    assert _check(report, "T 空走风险").status == "warning"
    assert report.status == "fail"


def test_validator_recreates_mapper_resampling_grid_without_mutating_flat_input(tmp_path):
    flat = _flat_source(tmp_path)
    for layer in range(4):
        flat.arrays[f"layer_{layer:04d}_R"][0, :, 0] = [5.0, 10.0, 5.0]
    curved = _curved_source(flat)

    report = validate_surface_job(flat, curved, _target())

    assert curved.arrays["layer_0001_R"].shape[1] > flat.arrays["layer_0001_R"].shape[1]
    assert report.checks[0].status == "pass"  # path contract and Z safety
    assert report.checks[3].status == "pass"  # mapped geometry
    assert report.checks[6].status == "pass"  # arc-length E compensation


def test_surface_validate_cli_writes_json_and_html_without_mutating_npz(tmp_path):
    flat = _flat_source(tmp_path)
    curved = _curved_source(flat)
    flat_path = tmp_path / "flat.npz"
    curved_path = tmp_path / "curved.npz"
    target_path = tmp_path / "surface.json"
    report_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    flat_path.write_bytes(flat.to_bytes())
    curved_path.write_bytes(curved.to_bytes())
    target_path.write_text(json.dumps(_target().raw_config), encoding="utf-8")

    assert main(["surface-validate", str(flat_path), str(curved_path), str(target_path), str(report_path), "--html", str(html_path)]) == 0

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["format"] == "surface_validation_report_v1"
    assert payload["overall_status"] == "warning"
    assert "曲面路径可打印性验证报告" in html_path.read_text(encoding="utf-8")
    assert read_source_npz(flat_path.read_bytes()).arrays["layer_0001_R_E"][0, 1] == flat.arrays["layer_0001_R_E"][0, 1]
