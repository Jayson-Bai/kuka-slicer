"""Layer embedding for conformal lattice geometry without path generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .lattice_generator import ConformalLatticeGeometry
from .mesh_domain import SurfaceMeshDomain
from .orientation_field import OrientationField


LayerEmbeddingMode = Literal["target_surface_normal_stack", "symmetric_shape_morphing"]


@dataclass(frozen=True, slots=True)
class LayerEmbedding:
    """Shared-topology node positions for a structural layer stack."""

    mode: LayerEmbeddingMode
    node_positions_xyz: np.ndarray
    layer_offsets_mm: np.ndarray
    lattice_edges: np.ndarray
    source_triangle_id_per_node: np.ndarray
    barycentric_weights_per_node: np.ndarray
    report: dict[str, object]

    def preview_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "node_positions_xyz": self.node_positions_xyz.tolist(),
            "lattice_edges": self.lattice_edges.tolist(),
            "layer_offsets_mm": self.layer_offsets_mm.tolist(),
            "report": self.report,
        }


def embed_lattice_layers(
    domain: SurfaceMeshDomain,
    orientation: OrientationField,
    geometry: ConformalLatticeGeometry,
    *,
    mode: LayerEmbeddingMode = "target_surface_normal_stack",
    layer_offsets_mm: np.ndarray | tuple[float, ...] | None = None,
    symmetric_layer_count: int | None = None,
    surface_start_layer: int = 0,
    flat_reference_nodes_xyz: np.ndarray | None = None,
    base_z_by_layer_mm: np.ndarray | tuple[float, ...] | None = None,
) -> LayerEmbedding:
    """Embed one lattice topology into normal-stack or compatibility layers.

    Normal-stack layers are the research mode: every layer shares node/edge
    identities and offsets the target surface along interpolated vertex normals.
    The symmetric mode intentionally labels itself as a morphology transition;
    it does not claim each intermediate layer is conformal.
    """

    _validate_inputs(domain, orientation, geometry, mode)
    normals = _node_normals(domain, orientation, geometry)
    if mode == "target_surface_normal_stack":
        offsets = _offsets(layer_offsets_mm)
        positions = geometry.lattice_nodes_xyz[None, :, :] + offsets[:, None, None] * normals[None, :, :]
        report = {
            "mode": mode,
            "strict_conformal_claim": "target surface only; normal offsets require separate offset-surface self-intersection validation",
            "topology_shared_across_layers": True,
            "node_count_per_layer": int(len(geometry.lattice_nodes_xyz)),
            "normal_offset_range_mm": {"min": float(np.min(offsets)), "max": float(np.max(offsets))},
        }
    elif mode == "symmetric_shape_morphing":
        if symmetric_layer_count is None:
            raise ValueError("symmetric_shape_morphing requires symmetric_layer_count")
        if flat_reference_nodes_xyz is None:
            raise ValueError("symmetric_shape_morphing requires an explicit flat_reference_nodes_xyz array")
        flat = np.asarray(flat_reference_nodes_xyz, dtype=np.float64)
        if flat.shape != geometry.lattice_nodes_xyz.shape or not np.all(np.isfinite(flat)):
            raise ValueError("flat_reference_nodes_xyz must be finite and match lattice nodes")
        alphas = _symmetric_alphas(symmetric_layer_count, surface_start_layer=surface_start_layer)
        positions = flat[None, :, :] + alphas[:, None, None] * (geometry.lattice_nodes_xyz[None, :, :] - flat[None, :, :])
        base_z = _base_z_by_layer(base_z_by_layer_mm, symmetric_layer_count)
        positions[:, :, 2] += base_z[:, None]
        offsets = alphas
        final_layer = symmetric_layer_count - 1
        return_layer = final_layer - surface_start_layer
        report = {
            "mode": mode,
            "strict_conformal_claim": "not_claimed_for_intermediate_layers; this is a legacy-compatible morphology transition",
            "topology_shared_across_layers": True,
            "node_count_per_layer": int(len(geometry.lattice_nodes_xyz)),
            "surface_start_layer": surface_start_layer,
            "surface_return_layer": return_layer,
            "peak_layer_indices": np.flatnonzero(np.isclose(alphas, 1.0)).tolist(),
            "alpha_range": {"min": float(np.min(alphas)), "max": float(np.max(alphas))},
            "alpha_by_layer": alphas.tolist(),
            "base_z_by_layer_mm": base_z.tolist(),
        }
    else:
        raise ValueError("unsupported layer embedding mode")
    return LayerEmbedding(
        mode=mode,
        node_positions_xyz=_readonly(positions),
        layer_offsets_mm=_readonly(offsets),
        lattice_edges=geometry.lattice_edges,
        source_triangle_id_per_node=geometry.source_triangle_id_per_node,
        barycentric_weights_per_node=geometry.barycentric_weights_per_node,
        report=report,
    )


def _validate_inputs(
    domain: SurfaceMeshDomain,
    orientation: OrientationField,
    geometry: ConformalLatticeGeometry,
    mode: str,
) -> None:
    if mode not in ("target_surface_normal_stack", "symmetric_shape_morphing"):
        raise ValueError("unsupported layer embedding mode")
    if orientation.vertex_normals_xyz.shape != domain.vertices.shape:
        raise ValueError("orientation must belong to the supplied domain")
    if len(geometry.source_triangle_id_per_node) != len(geometry.lattice_nodes_xyz):
        raise ValueError("geometry node provenance is malformed")


def _node_normals(domain: SurfaceMeshDomain, orientation: OrientationField, geometry: ConformalLatticeGeometry) -> np.ndarray:
    normals = np.empty_like(geometry.lattice_nodes_xyz, dtype=np.float64)
    for index, (face_id, barycentric) in enumerate(zip(geometry.source_triangle_id_per_node, geometry.barycentric_weights_per_node)):
        normal = barycentric @ orientation.vertex_normals_xyz[domain.faces[face_id]]
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            raise ValueError("lattice node has an undefined interpolated normal")
        normals[index] = normal / length
    return normals


def _offsets(values: np.ndarray | tuple[float, ...] | None) -> np.ndarray:
    offsets = np.asarray((0.0,) if values is None else values, dtype=np.float64)
    if offsets.ndim != 1 or not len(offsets) or not np.all(np.isfinite(offsets)):
        raise ValueError("layer_offsets_mm must be a non-empty finite one-dimensional array")
    if len(np.unique(offsets)) != len(offsets):
        raise ValueError("layer_offsets_mm must not contain duplicate layers")
    return offsets


def _base_z_by_layer(values: np.ndarray | tuple[float, ...] | None, layer_count: int) -> np.ndarray:
    if values is None:
        return np.zeros(layer_count, dtype=np.float64)
    base_z = np.asarray(values, dtype=np.float64)
    if base_z.shape != (layer_count,) or not np.all(np.isfinite(base_z)):
        raise ValueError("base_z_by_layer_mm must contain one finite Z value for every symmetric layer")
    if np.any(np.diff(base_z) <= 0.0):
        raise ValueError("base_z_by_layer_mm must be strictly increasing")
    return base_z


def _symmetric_alphas(layer_count: int, *, surface_start_layer: int = 0) -> np.ndarray:
    """Return the exact old ``LayerProgression`` alpha semantics without importing it.

    The independent conformal package deliberately does not take a runtime
    dependency on ``surface_mapper``.  The accompanying regression tests compare
    this implementation against ``LayerProgression`` for odd/even stacks and the
    one/two-layer active-region exceptions.
    """

    if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count < 1:
        raise ValueError("symmetric_layer_count must be an integer >= 1")
    if not isinstance(surface_start_layer, int) or isinstance(surface_start_layer, bool) or surface_start_layer < 0:
        raise ValueError("surface_start_layer must be a non-negative integer")
    final_layer = layer_count - 1
    if surface_start_layer > final_layer // 2:
        raise ValueError("surface_start_layer must leave a symmetric curved region around the middle layer")
    return_layer = final_layer - surface_start_layer
    active_count = return_layer - surface_start_layer + 1
    alphas = np.zeros(layer_count, dtype=np.float64)
    if active_count <= 2:
        alphas[surface_start_layer : return_layer + 1] = 1.0
        return alphas
    indices = np.arange(surface_start_layer, return_layer + 1, dtype=np.float64)
    edge_distance = np.minimum(indices - surface_start_layer, return_layer - indices)
    half_transition_steps = (active_count - 1) // 2
    raw = edge_distance / half_transition_steps
    alphas[surface_start_layer : return_layer + 1] = raw * raw * (3.0 - 2.0 * raw)
    return alphas


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result
