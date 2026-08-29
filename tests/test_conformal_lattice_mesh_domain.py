from __future__ import annotations

from io import BytesIO
import hashlib

import numpy as np
import pytest

from kuka_slicer.conformal_lattice import (
    build_double_sine_surface_domain,
    cut_mesh_along_edges,
    load_conformal_lattice_spec,
    load_triangle_mesh_domain,
    prepare_surface_mesh_domain,
)
from kuka_slicer.conformal_lattice.contracts import double_sine_source_sha256


def _spec(*, provider: str, source_file: str, sha256: str, source_extra: dict[str, object] | None = None):
    source = {
        "provider": provider,
        "source_file": source_file,
        "sha256": sha256,
        "domain": "outer_boundary_only",
    }
    if source_extra:
        source.update(source_extra)
    return {
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
            "wall_width_mm": 2.0,
            "base_cell_size_mm": 5.0,
            "boundary_mode": "clip",
            "phase_origin": [0.0, 0.0],
        },
        "fill_field": {"mode": "weighted_composite", "minimum": None, "maximum": None, "smoothing_length_mm": None, "drivers": []},
        "orientation_field": {"mode": "global_axis", "angle_deg": 0.0, "constraints": []},
        "layer_embedding": {"mode": "target_surface_normal_stack"},
        "quality_limits": {},
        "random_seed": 0,
    }


def _double_sine_spec():
    source = {
        "provider": "double_sine",
        "source_file": "generated://double-sine-baseline",
        "sha256": "0" * 64,
        "domain": "outer_boundary_only",
        "double_sine": {
            "amplitude_mm": 1.5,
            "wavelength_x_mm": 20.0,
            "wavelength_y_mm": 30.0,
            "phase_x_rad": 0.0,
            "phase_y_rad": 0.0,
            "z_reference_mm": 0.5,
            "xy_bounds_mm": [0.0, 0.0, 20.0, 30.0],
            "samples": [5, 4],
        },
    }
    source["sha256"] = double_sine_source_sha256(source)
    return _spec(
        provider="double_sine",
        source_file=str(source["source_file"]),
        sha256=str(source["sha256"]),
        source_extra={"double_sine": source["double_sine"]},
    )


def test_double_sine_domain_is_deterministic_and_has_one_boundary_loop():
    spec = load_conformal_lattice_spec(_double_sine_spec())

    first = build_double_sine_surface_domain(spec)
    second = build_double_sine_surface_domain(spec)

    assert first.vertices.shape == (20, 3)
    assert first.faces.shape == (24, 3)
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)
    assert first.vertices[0] == pytest.approx([0.0, 0.0, 0.5])
    assert len(first.boundary_loops) == 1
    assert first.report["provider"] == "double_sine"
    assert first.report["input_sha256"] == spec.source_sha256


def test_mesh_preparation_merges_vertices_removes_degenerate_faces_and_records_provenance():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2], [4, 2, 3], [0, 1, 0]], dtype=np.int64)

    domain = prepare_surface_mesh_domain(vertices, faces)

    assert domain.vertices.shape == (4, 3)
    assert domain.faces.shape == (2, 3)
    assert len(domain.boundary_loops) == 1
    assert domain.report["merged_vertex_count"] == 1
    assert domain.report["removed_degenerate_face_count"] == 1
    assert domain.source_face_index.tolist() == [0, 1]


def test_external_indexed_npz_is_hash_bound_and_loaded_as_a_surface_domain():
    vertices = np.asarray([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    output = BytesIO()
    np.savez(output, vertices=vertices, faces=faces)
    payload = output.getvalue()
    spec = load_conformal_lattice_spec(
        _spec(provider="triangle_mesh", source_file="surface.npz", sha256=hashlib.sha256(payload).hexdigest())
    )

    domain = load_triangle_mesh_domain(spec, payload)

    assert domain.vertices == pytest.approx(vertices)
    assert domain.faces.tolist() == faces.tolist()
    assert domain.report["provider"] == "triangle_mesh"
    assert domain.report["source_file"] == "surface.npz"


def test_external_mesh_hash_mismatch_is_a_hard_failure():
    spec = load_conformal_lattice_spec(_spec(provider="triangle_mesh", source_file="surface.npz", sha256="0" * 64))
    output = BytesIO()
    np.savez(output, vertices=np.zeros((3, 3)), faces=np.asarray([[0, 1, 2]]))

    with pytest.raises(ValueError, match="SHA-256"):
        load_triangle_mesh_domain(spec, output.getvalue())


def test_nonmanifold_and_duplicate_meshes_are_rejected_instead_of_silently_repaired():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(ValueError, match="non-manifold edge"):
        prepare_surface_mesh_domain(vertices, np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]]))
    with pytest.raises(ValueError, match="duplicate triangles"):
        prepare_surface_mesh_domain(vertices[:3], np.asarray([[0, 1, 2], [2, 1, 0]]))


def test_nonadjacent_crossing_triangles_are_rejected_as_self_intersection():
    vertices = np.asarray(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 0.5, 0]],
        dtype=float,
    )
    with pytest.raises(ValueError, match="self-intersecting"):
        prepare_surface_mesh_domain(vertices, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64))


def test_explicit_seam_duplicates_vertices_and_exposes_new_boundaries():
    domain = prepare_surface_mesh_domain(
        np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )

    cut = cut_mesh_along_edges(domain, [(0, 2)])

    assert cut.vertices.shape == (6, 3)
    assert cut.report["seam_edge_count"] == 1
    assert cut.report["cut_vertex_count"] == 2
    assert cut.report["connected_component_count"] == 2
    assert len(cut.boundary_loops) == 2


def test_legacy_surface_mapping_contract_is_not_accepted_as_a_conformal_spec():
    with pytest.raises(ValueError, match="conformal_lattice_spec_v1"):
        load_conformal_lattice_spec({"format": "graded_surface_v1", "units": "mm"})
