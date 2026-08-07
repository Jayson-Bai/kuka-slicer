from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pytest

from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths, write_external_source_npz
from kuka_slicer.surface_mapper import (
    LayerProgression,
    SurfaceMappingPlan,
    load_surface_target,
    map_source_job,
    read_source_npz,
)
from kuka_slicer.surface_mapper.server import (
    SurfaceMapperHandler,
    mapping_preview_payload,
    surface_mapper_html,
)


def _target(*, amplitude_mm: float = 1.0, phase_x_rad: float = 0.0, sha256: str = "same-model"):
    return load_surface_target(
        {
            "format": "graded_surface_v1",
            "units": "mm",
            "coordinate_system": {"plane": "XY", "build_axis": "Z", "origin": "stl_xy_min"},
            "domain": {"source": {"file_name": "honeycomb.stl", "sha256": sha256, "xy_bounds_mm": [0, 0, 10, 10]}},
            "surface": {
                "type": "double_sine_product",
                "amplitude_mm": amplitude_mm,
                "wavelength_x_mm": 20.0,
                "wavelength_y_mm": 20.0,
                "phase_x_rad": phase_x_rad,
                "phase_y_rad": 0.0,
                "z_reference_mm": 0.0,
            },
        }
    )


def _source(tmp_path, *, source_sha256: str = "same-model"):
    output = tmp_path / "flat.npz"
    job = ExternalSourceJob(
        material_paths=[
            MaterialPaths(0, "R", [np.asarray([[5, 5, 0.5], [6, 5, 0.5]], dtype=np.float64)]),
            MaterialPaths(1, "R", [np.asarray([[5, 5, 1.0], [6, 5, 1.0]], dtype=np.float64)], extrusion=[np.asarray([0.0, 1.0])]),
            MaterialPaths(2, "R", [np.asarray([[5, 5, 1.5], [6, 5, 1.5]], dtype=np.float64)]),
            MaterialPaths(3, "R", [np.asarray([[5, 5, 2.0], [6, 5, 2.0]], dtype=np.float64)]),
        ],
        meta={"source_model": {"file_name": "honeycomb.stl", "sha256": source_sha256}},
    )
    write_external_source_npz(job, output)
    return read_source_npz(output.read_bytes(), source_name=output.name)


def test_mapper_changes_only_z_by_logical_layer_and_preserves_extrusion(tmp_path):
    source = _source(tmp_path)
    result = map_source_job(source, _target(), SurfaceMappingPlan(LayerProgression(0, 3)))

    mapped = result.source
    assert np.array_equal(mapped.arrays["layer_0000_R"][..., :2], source.arrays["layer_0000_R"][..., :2], equal_nan=True)
    assert mapped.arrays["layer_0000_R"][0, 0, 2] == pytest.approx(0.5)
    assert mapped.arrays["layer_0001_R"][0, 0, 2] == pytest.approx(2.0)
    assert np.array_equal(mapped.arrays["layer_0001_R_E"], source.arrays["layer_0001_R_E"], equal_nan=True)
    assert mapped.meta["surface_mapping"]["progression"]["basis"] == "logical_layer_index"
    assert mapped.meta["surface_mapping"]["extrusion"] == "preserved_unrecalculated"


def test_mapper_rejects_negative_z_instead_of_silently_raising_the_path(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="negative Z"):
        map_source_job(
            source,
            _target(amplitude_mm=2.0, phase_x_rad=np.pi),
            SurfaceMappingPlan(LayerProgression(0, 3)),
        )


def test_mapper_rejects_a_surface_exported_from_another_stl(tmp_path):
    with pytest.raises(ValueError, match="different STL"):
        map_source_job(
            _source(tmp_path),
            _target(sha256="another-model"),
            SurfaceMappingPlan(LayerProgression(0, 1)),
        )


def test_mapper_accepts_verified_prusa_setup_paths_outside_the_part_domain(tmp_path):
    source = _source(tmp_path)
    source.arrays["layer_0000_T"] = np.asarray(
        [[[-8.0, -8.0, 0.5], [-7.0, -8.0, 0.5]]], dtype=np.float64
    )

    result = map_source_job(source, _target(), SurfaceMappingPlan(LayerProgression(0, 3)))

    assert result.source.arrays["layer_0000_T"][0, 0, 2] == pytest.approx(0.5)


def test_symmetric_progression_uses_logical_layer_index_not_z():
    progression = LayerProgression(2, 9)

    assert progression.surface_return_layer == 7
    assert progression.peak_layers == (4, 5)
    assert [progression.alpha(index) for index in range(10)] == pytest.approx(
        [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]
    )


def test_mapper_preview_and_web_shell_expose_the_separate_mapping_controls(tmp_path):
    source = _source(tmp_path)
    payload = mapping_preview_payload(source, _target(), SurfaceMappingPlan(LayerProgression(0, 3)))

    assert payload["plan"]["alpha_by_layer"] == {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0}
    assert payload["result"]["extrusion"]["status"] == "arc_length_ratio_compensated"
    html = surface_mapper_html()
    assert 'id="sourceFile"' in html
    assert 'id="targetFile"' in html
    assert 'id="startLayer"' in html
    assert "曲面起始层" in html
    assert "曲面完成层" not in html
    assert "Z 安全抬升" not in html
    assert 'id="sectionY"' in html
    assert 'id="sectionCanvas"' in html
    assert 'id="extrusionInfo"' in html
    assert "本版本不重算 E" not in html
    assert "/api/map?${params().toString()}" in html


def test_mapper_preview_samples_each_logical_layer_on_an_xz_section(tmp_path):
    source = _source(tmp_path)
    payload = mapping_preview_payload(
        source,
        _target(),
        SurfaceMappingPlan(LayerProgression(0, 3)),
    )

    section = payload["cross_section"]
    assert section["section_y_mm"] == pytest.approx(5.0)
    assert len(section["x_mm"]) == 181
    assert [layer["logical_layer"] for layer in section["layers"]] == [0, 1, 2, 3]
    assert section["layers"][0]["z_mm"][90] == pytest.approx(0.5)
    assert section["layers"][1]["z_mm"][90] == pytest.approx(2.0)


def test_mapped_npz_is_re_readable_and_records_mapping_metadata(tmp_path):
    source = _source(tmp_path)
    mapped = map_source_job(source, _target(), SurfaceMappingPlan(LayerProgression(0, 3))).source

    reloaded = read_source_npz(mapped.to_bytes())

    assert reloaded.meta["surface_mapping"]["format"] == "surface_mapping_v1"
    assert reloaded.meta["surface_mapping"]["xy"] == "preserved"


def test_mapper_http_api_imports_previews_and_exports_without_writing_server_files(tmp_path):
    source = _source(tmp_path)
    target = _target()
    server = ThreadingHTTPServer(("127.0.0.1", 0), SurfaceMapperHandler)
    server.source_jobs = {}
    server.surface_targets = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        source_response = _post(base + "/api/source-npz", source.to_bytes())
        target_response = _post(
            base + "/api/surface-config",
            json.dumps(target.raw_config).encode("utf-8"),
        )
        params = urlencode(
            {
                "source_id": source_response["source_id"],
                "target_id": target_response["target_id"],
                "surface_start_layer": 0,
            }
        )
        with urlopen(base + "/api/preview?" + params) as response:
            preview = json.loads(response.read())
        exported = _post_raw(base + "/api/map?" + params)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert preview["ok"] is True
    assert preview["plan"]["alpha_by_layer"] == {"0": 0.0, "1": 1.0, "2": 1.0, "3": 0.0}
    exported_source = read_source_npz(exported)
    assert exported_source.meta["surface_mapping"]["format"] == "surface_mapping_v1"
    assert exported_source.meta["surface_mapping"]["extrusion"] == "arc_length_ratio_compensated"
    assert exported_source.meta["extrusion_compensation"]["replaced_arrays"] == ["layer_0001_R_E"]
    assert exported_source.arrays["layer_0001_R_E"][0, 1] > source.arrays["layer_0001_R_E"][0, 1]


def _post(url: str, data: bytes) -> dict[str, object]:
    return json.loads(_post_raw(url, data))


def _post_raw(url: str, data: bytes = b"") -> bytes:
    request = Request(url, data=data, method="POST")
    with urlopen(request) as response:
        return response.read()
