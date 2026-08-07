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


def _target(*, phase_x_rad: float = 0.0, sha256: str = "same-model"):
    return load_surface_target(
        {
            "format": "graded_surface_v1",
            "units": "mm",
            "coordinate_system": {"plane": "XY", "build_axis": "Z", "origin": "stl_xy_min"},
            "domain": {"source": {"file_name": "honeycomb.stl", "sha256": sha256, "xy_bounds_mm": [0, 0, 10, 10]}},
            "surface": {
                "type": "double_sine_product",
                "amplitude_mm": 1.0,
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
        ],
        meta={"source_model": {"file_name": "honeycomb.stl", "sha256": source_sha256}},
    )
    write_external_source_npz(job, output)
    return read_source_npz(output.read_bytes(), source_name=output.name)


def test_mapper_changes_only_z_by_logical_layer_and_preserves_extrusion(tmp_path):
    source = _source(tmp_path)
    result = map_source_job(source, _target(), SurfaceMappingPlan(LayerProgression(0, 1, curve="linear")))

    mapped = result.source
    assert np.array_equal(mapped.arrays["layer_0000_R"][..., :2], source.arrays["layer_0000_R"][..., :2], equal_nan=True)
    assert mapped.arrays["layer_0000_R"][0, 0, 2] == pytest.approx(0.5)
    assert mapped.arrays["layer_0001_R"][0, 0, 2] == pytest.approx(2.0)
    assert np.array_equal(mapped.arrays["layer_0001_R_E"], source.arrays["layer_0001_R_E"], equal_nan=True)
    assert mapped.meta["surface_mapping"]["progression"]["basis"] == "logical_layer_index"
    assert mapped.meta["surface_mapping"]["extrusion"] == "preserved_unrecalculated"


def test_mapper_auto_offset_keeps_mapped_path_above_source_lowest_z(tmp_path):
    source = _source(tmp_path)
    result = map_source_job(
        source,
        _target(phase_x_rad=np.pi),
        SurfaceMappingPlan(LayerProgression(0, 1, curve="linear")),
    )

    assert result.applied_z_offset_mm == pytest.approx(0.5)
    assert result.mapped_z_bounds_mm[0] == pytest.approx(source.z_bounds_mm[0])


def test_mapper_rejects_a_surface_exported_from_another_stl(tmp_path):
    with pytest.raises(ValueError, match="different STL"):
        map_source_job(
            _source(tmp_path),
            _target(sha256="another-model"),
            SurfaceMappingPlan(LayerProgression(0, 1)),
        )


def test_smoothstep_progression_uses_logical_layer_index_not_z():
    progression = LayerProgression(4, 6, curve="smoothstep")

    assert progression.alpha(3) == 0.0
    assert progression.alpha(4) == 0.0
    assert progression.alpha(5) == pytest.approx(0.5)
    assert progression.alpha(6) == 1.0


def test_mapper_preview_and_web_shell_expose_the_separate_mapping_controls(tmp_path):
    source = _source(tmp_path)
    payload = mapping_preview_payload(source, _target(), SurfaceMappingPlan(LayerProgression(0, 1)))

    assert payload["plan"]["alpha_by_layer"] == {0: 0.0, 1: 1.0}
    assert payload["result"]["extrusion"] == "preserved_unrecalculated"
    html = surface_mapper_html()
    assert 'id="sourceFile"' in html
    assert 'id="targetFile"' in html
    assert 'id="startLayer"' in html
    assert "/api/map?${params().toString()}" in html


def test_mapped_npz_is_re_readable_and_records_mapping_metadata(tmp_path):
    source = _source(tmp_path)
    mapped = map_source_job(source, _target(), SurfaceMappingPlan(LayerProgression(0, 1))).source

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
                "start_logical_layer": 0,
                "end_logical_layer": 1,
                "curve": "smoothstep",
                "offset_mode": "auto",
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
    assert preview["plan"]["alpha_by_layer"] == {"0": 0.0, "1": 1.0}
    assert read_source_npz(exported).meta["surface_mapping"]["format"] == "surface_mapping_v1"


def _post(url: str, data: bytes) -> dict[str, object]:
    return json.loads(_post_raw(url, data))


def _post_raw(url: str, data: bytes = b"") -> bytes:
    request = Request(url, data=data, method="POST")
    with urlopen(request) as response:
        return response.read()
