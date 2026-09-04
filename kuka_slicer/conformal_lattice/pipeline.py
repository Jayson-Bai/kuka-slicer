"""Ordered, side-effect-free orchestration of the conformal lattice gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import ConformalLatticeSpec, load_conformal_lattice_spec
from .fill_ratio_validation import FillRatioValidation, validate_realized_fill_ratio
from .layer_embedding import LayerEmbedding, embed_lattice_layers
from .lattice_generator import ConformalLatticeGeometry, generate_conformal_lattice_geometry
from .mesh_domain import SurfaceMeshDomain, build_double_sine_surface_domain
from .orientation_field import OrientationField, build_orientation_field
from .parameterization import LSCMParameterization, parameterize_spec_lscm
from .path_bridge import ConformalLatticePathGraph, ExtrusionVolumeModel, build_conformal_lattice_path_graph, write_conformal_lattice_external_npz
from .phase_coordinates import PhaseCoordinates, solve_phase_coordinates
from .preview import conformal_lattice_preview_payload
from .scalar_fields import DesignFieldResult, compose_design_fields_from_spec


@dataclass(frozen=True, slots=True)
class ConformalLatticeRun:
    """All immutable outputs from Gates 1--8 for one validated UI config."""

    spec: ConformalLatticeSpec
    domain: SurfaceMeshDomain
    parameterization: LSCMParameterization
    design_fields: DesignFieldResult
    orientation: OrientationField
    phase: PhaseCoordinates
    geometry: ConformalLatticeGeometry
    fill_validation: FillRatioValidation
    layer_embedding: LayerEmbedding
    path_graph: ConformalLatticePathGraph | None

    def preview_payload(self) -> dict[str, object]:
        """Return Gate 7 diagnostics without changing any generated geometry."""

        return conformal_lattice_preview_payload(
            self.domain,
            self.parameterization,
            self.design_fields,
            self.orientation,
            self.phase,
            self.geometry,
            fill_validation=self.fill_validation,
            layer_embedding=self.layer_embedding,
        )

    def main_preview_payload(self, *, planning_line_width_mm: float) -> dict[str, object]:
        """Return the final user-facing payload for the existing main Canvas."""

        if self.path_graph is None:
            raise ValueError("main preview requires an explicit process E conversion and path graph")
        from .main_preview import main_preview_payload_from_conformal_path_graph

        return main_preview_payload_from_conformal_path_graph(
            self.path_graph,
            planning_line_width_mm=planning_line_width_mm,
        )

    @property
    def report(self) -> dict[str, object]:
        """Summarise output availability without pretending a path export exists."""

        return {
            "format": "conformal_lattice_run_v1",
            "source_surface_sha256": self.spec.source_sha256,
            "geometry_npz_export_available": True,
            "path_npz_export_available": self.path_graph is not None,
            "path_npz_requirement": None
            if self.path_graph is not None
            else "provide explicit bead_cross_section_area_mm2 and e_volume_per_unit_mm3 from a process preset",
            "layer_embedding": self.layer_embedding.report,
            "fill_ratio": self.fill_validation.report,
        }


def run_conformal_lattice_pipeline(
    config: ConformalLatticeSpec | bytes | str | Mapping[str, object],
    *,
    logical_layer_count: int | None = None,
    extrusion: ExtrusionVolumeModel | None = None,
    fill_samples_per_triangle_side: int = 6,
) -> ConformalLatticeRun:
    """Run Gates 1--8 in order for the supported double-sine UI workflow.

    ``logical_layer_count`` belongs to the slicer/process side of the interface,
    not to the analytical surface definition.  Path export stays unavailable
    until its explicit physical E conversion is supplied.
    """

    spec = config if isinstance(config, ConformalLatticeSpec) else load_conformal_lattice_spec(config)
    if spec.source_provider != "double_sine":
        raise ValueError("first-version UI pipeline supports only source_surface.provider=double_sine")
    reference = spec.source_surface.get("reference_stl")
    if spec.part:
        logical_layer_count, base_z_by_layer = _physical_layer_schedule(spec, logical_layer_count)
    elif not isinstance(reference, Mapping) or reference.get("build_axis") != "z":
        raise ValueError("first-version double-sine conformal workflow requires reference_stl.build_axis=z")
    else:
        base_z_by_layer = None
    if not isinstance(logical_layer_count, int) or isinstance(logical_layer_count, bool) or logical_layer_count < 1:
        raise ValueError("logical_layer_count must be an integer >= 1")
    if not isinstance(fill_samples_per_triangle_side, int) or fill_samples_per_triangle_side < 2:
        raise ValueError("fill_samples_per_triangle_side must be an integer >= 2")

    domain = build_double_sine_surface_domain(spec)
    parameterization = parameterize_spec_lscm(spec, domain)
    design_fields = compose_design_fields_from_spec(domain, spec)
    orientation = _orientation_from_spec(domain, spec)
    phase = solve_phase_coordinates(domain, parameterization, design_fields, orientation)
    boundary_mode = str(spec.lattice["boundary_mode"])
    if boundary_mode not in ("clip", "inset"):
        raise ValueError("first-version UI pipeline supports lattice.boundary_mode=clip or inset")
    geometry = generate_conformal_lattice_geometry(
        domain,
        parameterization,
        design_fields,
        orientation,
        phase,
        boundary_mode=boundary_mode,
        phase_origin=tuple(float(value) for value in spec.lattice["phase_origin"]),
        random_seed=spec.random_seed,
        config_metadata=spec.metadata(),
    )
    fill_validation = validate_realized_fill_ratio(
        domain,
        parameterization,
        design_fields,
        phase,
        geometry,
        wall_width_mm=float(spec.lattice["wall_width_mm"]),
        samples_per_triangle_side=fill_samples_per_triangle_side,
    )
    layer_embedding = _symmetric_layer_embedding(domain, orientation, geometry, spec, logical_layer_count, base_z_by_layer)
    path_graph = None if extrusion is None else build_conformal_lattice_path_graph(
        geometry,
        extrusion,
        layer_embedding=layer_embedding,
        node_normals_xyz=_lattice_node_normals(domain, orientation, geometry),
        wall_bead_count=int(spec.lattice.get("wall_bead_count", 1)),
        nominal_bead_width_mm=float(spec.manufacturing.get("nominal_bead_width_mm", 2.0)),
        config_metadata=spec.metadata(),
    )
    return ConformalLatticeRun(
        spec=spec,
        domain=domain,
        parameterization=parameterization,
        design_fields=design_fields,
        orientation=orientation,
        phase=phase,
        geometry=geometry,
        fill_validation=fill_validation,
        layer_embedding=layer_embedding,
        path_graph=path_graph,
    )


def write_conformal_lattice_outputs(
    run: ConformalLatticeRun,
    output_directory: str | Path,
    *,
    material: str = "R",
) -> dict[str, Path]:
    """Write fresh output sidecars; never overwrite an existing user artifact."""

    destination = Path(output_directory)
    if not destination.is_dir():
        raise ValueError("output_directory must be an existing directory")
    geometry_path = destination / "conformal_lattice_geometry_v1.npz"
    if geometry_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {geometry_path}")
    run.geometry.save_npz(
        geometry_path,
        domain=run.domain,
        parameterization=run.parameterization,
        design_fields=run.design_fields,
        orientation=run.orientation,
        phase=run.phase,
        fill_validation=run.fill_validation,
    )
    outputs = {"geometry": geometry_path}
    if run.path_graph is not None:
        path_path = destination / "external_layer_paths_v1.npz"
        if path_path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path_path}")
        write_conformal_lattice_external_npz(run.path_graph, path_path, material=material)  # type: ignore[arg-type]
        outputs["paths"] = path_path
    return outputs


def _orientation_from_spec(domain: SurfaceMeshDomain, spec: ConformalLatticeSpec) -> OrientationField:
    if spec.orientation_field.get("mode") != "global_axis":
        raise ValueError("first-version UI pipeline supports orientation_field.mode=global_axis")
    angle = math.radians(float(spec.orientation_field.get("angle_deg", 0.0)))
    return build_orientation_field(
        domain,
        mode="global_axis",
        global_axis_xyz=np.asarray([math.cos(angle), math.sin(angle), 0.0]),
    )


def _symmetric_layer_embedding(
    domain: SurfaceMeshDomain,
    orientation: OrientationField,
    geometry: ConformalLatticeGeometry,
    spec: ConformalLatticeSpec,
    logical_layer_count: int,
    base_z_by_layer: np.ndarray | None,
) -> LayerEmbedding:
    embedding = spec.layer_embedding
    if embedding.get("mode") != "symmetric_shape_morphing" or embedding.get("transition") != "smoothstep":
        raise ValueError("first-version UI pipeline supports only symmetric_shape_morphing with smoothstep")
    surface = spec.source_surface["double_sine"]
    if not isinstance(surface, Mapping):
        raise ValueError("double-sine source metadata is malformed")
    flat = np.array(geometry.lattice_nodes_xyz, copy=True)
    flat[:, 2] = float(surface["z_reference_mm"])
    return embed_lattice_layers(
        domain,
        orientation,
        geometry,
        mode="symmetric_shape_morphing",
        symmetric_layer_count=logical_layer_count,
        surface_start_layer=int(embedding["surface_start_layer"]),
        flat_reference_nodes_xyz=flat,
        base_z_by_layer_mm=base_z_by_layer,
    )


def _lattice_node_normals(
    domain: SurfaceMeshDomain,
    orientation: OrientationField,
    geometry: ConformalLatticeGeometry,
) -> np.ndarray:
    normals = np.empty_like(geometry.lattice_nodes_xyz, dtype=np.float64)
    for index, (face_id, barycentric) in enumerate(
        zip(geometry.source_triangle_id_per_node, geometry.barycentric_weights_per_node)
    ):
        normal = barycentric @ orientation.vertex_normals_xyz[domain.faces[face_id]]
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            raise ValueError("conformal lattice node has an undefined surface normal")
        normals[index] = normal / length
    return normals


def _physical_layer_schedule(spec: ConformalLatticeSpec, requested_count: int | None) -> tuple[int, np.ndarray]:
    """Return monotonic layer-centre Z values whose printed extent is the requested part height."""

    final_height = float(spec.part["final_height_mm"])
    nominal_height = float(spec.manufacturing["layer_height_mm"])
    count = int(math.ceil(final_height / nominal_height))
    if requested_count is not None and requested_count != count:
        raise ValueError("logical_layer_count must match the rectangular part final_height_mm and layer_height_mm")
    thicknesses = np.full(count, nominal_height, dtype=np.float64)
    thicknesses[-1] = final_height - nominal_height * (count - 1)
    centres = np.cumsum(thicknesses) - thicknesses * 0.5
    return count, centres
