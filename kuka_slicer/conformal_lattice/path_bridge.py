"""Stable structural-edge bridge to the existing external NPZ contract.

This module is intentionally a bridge, not a path planner.  It preserves one
validated lattice edge as one depositing path, retains an auditable edge ID,
and calculates deposition from the *actual 3-D* edge length.  Trail ordering,
zero-E connectors, travel routing, and Core XYZABC processing remain later
responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from ..external_npz import ExternalSourceJob, MaterialPaths, TravelPaths, write_external_source_npz
from .contracts import canonical_json_sha256
from .layer_embedding import LayerEmbedding
from .lattice_generator import ConformalLatticeGeometry


_EDGE_ID_BITS = 32
_EDGE_ID_COMPONENT_MAX = (1 << _EDGE_ID_BITS) - 1


@dataclass(frozen=True, slots=True)
class ExtrusionVolumeModel:
    """Explicit conversion from 3-D deposited length to cumulative E.

    ``bead_cross_section_area_mm2`` defines the intended deposited volume per
    millimetre of centreline.  ``e_volume_per_unit_mm3`` defines the machine's
    E unit; it is deliberately required rather than inferred from a global
    printer profile or a flat XY reference path.
    """

    bead_cross_section_area_mm2: float
    e_volume_per_unit_mm3: float
    preview_line_width_mm: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("bead_cross_section_area_mm2", self.bead_cross_section_area_mm2),
            ("e_volume_per_unit_mm3", self.e_volume_per_unit_mm3),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.preview_line_width_mm is not None and (
            not np.isfinite(self.preview_line_width_mm) or self.preview_line_width_mm <= 0.0
        ):
            raise ValueError("preview_line_width_mm must be positive and finite when supplied")

    def metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "format": "conformal_edge_volume_e_v1",
            "bead_cross_section_area_mm2": float(self.bead_cross_section_area_mm2),
            "e_volume_per_unit_mm3": float(self.e_volume_per_unit_mm3),
            "formula": "delta_E = 3d_edge_length_mm * bead_cross_section_area_mm2 / e_volume_per_unit_mm3",
            "uses_actual_3d_length": True,
            "requires_xy_preservation": False,
        }
        if self.preview_line_width_mm is not None:
            metadata["preview_line_width_mm"] = float(self.preview_line_width_mm)
        return metadata


@dataclass(frozen=True, slots=True)
class ConformalLatticePathGraph:
    """Stable IDs and per-layer geometry before trail/partition planning."""

    node_ids: np.ndarray
    edge_ids: np.ndarray
    edge_node_ids: np.ndarray
    edge_parent_id: np.ndarray
    edge_segment_id: np.ndarray
    edge_source_triangle_id: np.ndarray
    layer_node_positions_xyz: np.ndarray
    edge_length_mm: np.ndarray
    deposited_volume_mm3: np.ndarray
    cumulative_extrusion_e: np.ndarray
    report: dict[str, object]
    metadata: dict[str, object]

    def to_external_source_job(self, *, material: Literal["R", "F"] = "R") -> ExternalSourceJob:
        """Adapt edge paths without joining, routing, or changing their IDs."""

        if material not in ("R", "F"):
            raise ValueError("material must be R or F")
        material_paths: list[MaterialPaths] = []
        travel_paths: list[TravelPaths] = []
        edge_ids_by_layer: dict[str, list[int]] = {}
        for layer_index, positions in enumerate(self.layer_node_positions_xyz):
            paths = [positions[node_ids].copy() for node_ids in self.edge_node_ids]
            extrusion = [values.copy() for values in self.cumulative_extrusion_e[layer_index]]
            material_paths.append(MaterialPaths(layer_index, material, paths, extrusion))
            # Emit the established empty T array explicitly.  No connector or
            # route is fabricated before the later trail/partition gate.
            travel_paths.append(TravelPaths(layer_index, []))
            edge_ids_by_layer[str(layer_index)] = [int(value) for value in self.edge_ids]

        bridge_meta = {
            **self.metadata,
            "material": material,
            "path_order": "ascending_stable_edge_id; one two-point path per structural edge",
            "edge_ids_by_layer": edge_ids_by_layer,
            "trail_partition_status": "not_planned; no edge joining, connector, or travel routing has been applied",
            "core_handoff": "external_layer_paths_v1 XYZ only; downstream Core remains responsible for final XYZABC",
        }
        job_metadata: dict[str, object] = {
            "conformal_lattice_path_bridge": bridge_meta,
            "path_roles": {
                material: {
                    str(layer): ["conformal_structural_edge"] * len(self.edge_ids)
                    for layer in range(len(self.layer_node_positions_xyz))
                }
            },
            "extrusion_compensation": {
                "format": "conformal_edge_volume_e_v1",
                "scope": "per-edge cumulative E from actual 3-D arc length and explicit bead volume",
                "requires_xy_preservation": False,
                "replaces_legacy_arc_length_ratio": True,
            },
        }
        extrusion = self.metadata.get("config", {}).get("extrusion") if isinstance(self.metadata.get("config"), Mapping) else None
        preview_line_width = extrusion.get("preview_line_width_mm") if isinstance(extrusion, Mapping) else None
        if isinstance(preview_line_width, (int, float)) and not isinstance(preview_line_width, bool):
            job_metadata["slicing"] = {"resolved_config": {"line_width": float(preview_line_width)}}
        return ExternalSourceJob(
            material_paths=material_paths,
            travel_paths=travel_paths,
            meta=job_metadata,
        )


def build_conformal_lattice_path_graph(
    geometry: ConformalLatticeGeometry,
    extrusion: ExtrusionVolumeModel,
    *,
    layer_embedding: LayerEmbedding | None = None,
    config_metadata: Mapping[str, object] | None = None,
) -> ConformalLatticePathGraph:
    """Turn verified structural edges into a deterministic, un-routed graph.

    The geometry's parent/segment IDs are encoded into 64-bit edge IDs.  This
    makes an edge's identity independent of the in-memory row ordering while
    keeping the source edge provenance recoverable without a side lookup.
    """

    _validate_geometry(geometry)
    source_surface_sha256, solver_seed = _required_geometry_provenance(geometry)
    positions, embedding_mode = _layer_positions(geometry, layer_embedding)
    edge_ids, edge_nodes, parent_ids, segment_ids, source_faces = _stable_edges(geometry)
    starts = positions[:, edge_nodes[:, 0], :]
    ends = positions[:, edge_nodes[:, 1], :]
    lengths = np.linalg.norm(ends - starts, axis=2)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 1e-12):
        raise ValueError("validated structural edges must have finite non-zero 3-D length in every output layer")
    volume = lengths * float(extrusion.bead_cross_section_area_mm2)
    cumulative_e = np.stack(
        (np.zeros_like(volume), volume / float(extrusion.e_volume_per_unit_mm3)),
        axis=2,
    )
    config = {
        "layer_embedding_mode": embedding_mode,
        "extrusion": extrusion.metadata(),
        "additional": dict(config_metadata or {}),
    }
    metadata = {
        "format": "conformal_lattice_path_graph_v1",
        "source_surface_sha256": source_surface_sha256,
        "geometry_format": geometry.metadata.get("format"),
        "geometry_config_sha256": geometry.metadata.get("config_sha256"),
        "random_seed": solver_seed,
        "solver_seed": solver_seed,
        "config": config,
        "config_sha256": canonical_json_sha256(config),
        "quality": {
            "zero_length_edge_count": 0,
            "minimum_edge_length_mm": float(np.min(lengths)),
            "maximum_edge_length_mm": float(np.max(lengths)),
            "all_edge_paths_are_two_points": True,
            "edge_identity_unique": True,
        },
    }
    report = {
        "layer_count": int(len(positions)),
        "node_count": int(len(geometry.lattice_nodes_xyz)),
        "edge_count_per_layer": int(len(edge_ids)),
        "total_deposited_volume_mm3": float(np.sum(volume)),
        "trail_partition_status": "deferred_to_later_gate",
        "legacy_surface_mapper_used": False,
        "legacy_honeycomb_pathing_used": False,
    }
    return ConformalLatticePathGraph(
        node_ids=_readonly(np.arange(len(geometry.lattice_nodes_xyz), dtype=np.int64)),
        edge_ids=_readonly(edge_ids),
        edge_node_ids=_readonly(edge_nodes),
        edge_parent_id=_readonly(parent_ids),
        edge_segment_id=_readonly(segment_ids),
        edge_source_triangle_id=_readonly(source_faces),
        layer_node_positions_xyz=_readonly(positions),
        edge_length_mm=_readonly(lengths),
        deposited_volume_mm3=_readonly(volume),
        cumulative_extrusion_e=_readonly(cumulative_e),
        report=report,
        metadata=metadata,
    )


def write_conformal_lattice_external_npz(
    graph: ConformalLatticePathGraph,
    output_path: str | Path,
    *,
    material: Literal["R", "F"] = "R",
) -> Path:
    """Write an ``external_layer_paths_v1`` source NPZ from a path graph."""

    destination = Path(output_path)
    job = graph.to_external_source_job(material=material)
    write_external_source_npz(job, destination)
    return destination


def _validate_geometry(geometry: ConformalLatticeGeometry) -> None:
    edge_count = len(geometry.lattice_edges)
    if edge_count == 0:
        raise ValueError("path bridge requires at least one validated structural edge")
    if geometry.lattice_edges.shape != (edge_count, 2):
        raise ValueError("lattice_edges must have shape (edge_count, 2)")
    for name, values in (
        ("lattice_edge_parent_id", geometry.lattice_edge_parent_id),
        ("lattice_edge_segment_id", geometry.lattice_edge_segment_id),
        ("lattice_edge_source_triangle_id", geometry.lattice_edge_source_triangle_id),
    ):
        if np.asarray(values).shape != (edge_count,):
            raise ValueError(f"{name} must contain one value per lattice edge")
    if not np.all((geometry.lattice_edges >= 0) & (geometry.lattice_edges < len(geometry.lattice_nodes_xyz))):
        raise ValueError("lattice edge nodes are outside the lattice node table")


def _required_geometry_provenance(geometry: ConformalLatticeGeometry) -> tuple[str, int]:
    source_surface_sha256 = geometry.metadata.get("source_surface_sha256")
    solver_seed = geometry.metadata.get("random_seed")
    if not isinstance(source_surface_sha256, str) or len(source_surface_sha256) != 64:
        raise ValueError("path bridge requires the geometry source_surface_sha256 provenance")
    if not isinstance(solver_seed, int) or isinstance(solver_seed, bool) or solver_seed < 0:
        raise ValueError("path bridge requires the geometry non-negative random_seed provenance")
    return source_surface_sha256, solver_seed


def _layer_positions(
    geometry: ConformalLatticeGeometry,
    layer_embedding: LayerEmbedding | None,
) -> tuple[np.ndarray, str]:
    if layer_embedding is None:
        return np.asarray(geometry.lattice_nodes_xyz, dtype=np.float64)[None, :, :], "surface_geometry_only"
    positions = np.asarray(layer_embedding.node_positions_xyz, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != geometry.lattice_nodes_xyz.shape or not np.all(np.isfinite(positions)):
        raise ValueError("layer embedding positions must be finite and match geometry lattice nodes")
    if not np.array_equal(layer_embedding.lattice_edges, geometry.lattice_edges):
        raise ValueError("layer embedding must preserve the exact validated lattice edge topology")
    return positions, layer_embedding.mode


def _stable_edges(
    geometry: ConformalLatticeGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent = np.asarray(geometry.lattice_edge_parent_id, dtype=np.int64)
    segment = np.asarray(geometry.lattice_edge_segment_id, dtype=np.int64)
    if np.any(parent < 0) or np.any(parent > _EDGE_ID_COMPONENT_MAX) or np.any(segment < 0) or np.any(segment > _EDGE_ID_COMPONENT_MAX):
        raise ValueError("edge parent and segment IDs must fit the stable 32-bit ID components")
    edge_ids = (parent << _EDGE_ID_BITS) | segment
    if len(np.unique(edge_ids)) != len(edge_ids):
        raise ValueError("geometry contains duplicate parent/segment stable edge IDs")
    raw_nodes = np.asarray(geometry.lattice_edges, dtype=np.int64)
    nodes = np.sort(raw_nodes, axis=1)
    faces = np.asarray(geometry.lattice_edge_source_triangle_id, dtype=np.int64)
    order = np.argsort(edge_ids, kind="stable")
    return edge_ids[order], nodes[order], parent[order], segment[order], faces[order]


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result
