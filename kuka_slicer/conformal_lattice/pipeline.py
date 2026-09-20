"""Ordered, side-effect-free orchestration of the conformal lattice gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import ConformalLatticeSpec, load_conformal_lattice_spec
from .continuous_course import ContinuousCoursePlan, build_continuous_course_plan, embed_continuous_course_plan
from .fill_ratio_validation import FillRatioValidation, validate_realized_fill_ratio
from .layer_embedding import LayerEmbedding, embed_lattice_layers
from .lattice_generator import (
    ConformalLatticeGeometry,
    choose_boundary_safe_phase_origin,
    generate_conformal_lattice_geometry,
)
from .mesh_domain import SurfaceMeshDomain, build_double_sine_surface_domain, build_planar_surface_domain
from .orientation_field import OrientationField, build_orientation_field
from .parameterization import LSCMParameterization, parameterize_spec_lscm
from .path_bridge import ConformalLatticePathGraph, ExtrusionVolumeModel, build_conformal_lattice_path_graph, write_conformal_lattice_external_npz
from .phase_coordinates import PhaseCoordinates, solve_phase_coordinates
from .preview import conformal_lattice_preview_payload
from .scalar_fields import DesignFieldResult, compose_design_fields_from_spec
from .surface_field import HeightField, height_field_from_spec


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
    # Gate 6 is an expensive diagnostic.  It is intentionally absent from
    # normal production exports, where it must not delay path planning/Core.
    fill_validation: FillRatioValidation | None
    layer_embedding: LayerEmbedding
    path_graph: ConformalLatticePathGraph | None
    # The production resin/fiber fill is the designer's red continuous-course
    # topology, mapped only after its planar geometry has been fixed.
    continuous_course_plan: ContinuousCoursePlan | None
    continuous_course_paths_by_layer: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]

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
            "fill_ratio": (
                self.fill_validation.report
                if self.fill_validation is not None
                else {
                    "status": "skipped",
                    "reason": "Gate 6 实际填充率测量仅用于质量诊断，未在生产导出中执行",
                }
            ),
        }


def _build_generated_surface_domain(
    spec: ConformalLatticeSpec,
    *,
    xy_bounds_mm: tuple[float, float, float, float] | None = None,
) -> SurfaceMeshDomain:
    """Dispatch generated geometry without leaking provider checks downstream."""

    if spec.source_provider == "double_sine":
        return build_double_sine_surface_domain(spec, xy_bounds_mm=xy_bounds_mm)
    if spec.source_provider == "planar":
        return build_planar_surface_domain(spec, xy_bounds_mm=xy_bounds_mm)
    raise ValueError(f"unsupported generated source provider: {spec.source_provider}")


def run_conformal_lattice_pipeline(
    config: ConformalLatticeSpec | bytes | str | Mapping[str, object],
    *,
    logical_layer_count: int | None = None,
    physical_layer_height_mm: float | None = None,
    extrusion: ExtrusionVolumeModel | None = None,
    validate_fill_ratio: bool = False,
    fill_samples_per_triangle_side: int = 6,
) -> ConformalLatticeRun:
    """Run Gates 1--8 for generated curved or planar honeycomb designs.

    ``logical_layer_count`` belongs to the slicer/process side of the interface,
    not to the analytical surface definition.  Path export stays unavailable
    until its explicit physical E conversion is supplied.  Gate 6 actual-fill
    measurement is opt-in because its per-cell/per-triangle sampling is a
    quality diagnostic, not an input to the one-stroke path or Core export.
    """

    spec = config if isinstance(config, ConformalLatticeSpec) else load_conformal_lattice_spec(config)
    if spec.source_provider not in ("double_sine", "planar"):
        raise ValueError("UI lattice pipeline supports source_surface.provider=double_sine or planar")
    reference = spec.source_surface.get("reference_stl")
    if spec.part:
        logical_layer_count, base_z_by_layer = _physical_layer_schedule(
            spec,
            logical_layer_count,
            physical_layer_height_mm=physical_layer_height_mm,
        )
    elif spec.source_provider == "planar":
        raise ValueError("planar lattice workflow requires a rectangular part")
    elif not isinstance(reference, Mapping) or reference.get("build_axis") != "z":
        raise ValueError("first-version double-sine conformal workflow requires reference_stl.build_axis=z")
    else:
        base_z_by_layer = None
    if not isinstance(logical_layer_count, int) or isinstance(logical_layer_count, bool) or logical_layer_count < 1:
        raise ValueError("logical_layer_count must be an integer >= 1")
    if not isinstance(fill_samples_per_triangle_side, int) or fill_samples_per_triangle_side < 2:
        raise ValueError("fill_samples_per_triangle_side must be an integer >= 2")
    if not isinstance(validate_fill_ratio, bool):
        raise ValueError("validate_fill_ratio must be a boolean")

    grip_end_length_mm = _symmetric_grip_end_length(spec)
    # The generated source field is always global. Only the honeycomb generator
    # is restricted to the central working region; perimeter and grip paths
    # below still use the original whole-part field.
    full_domain = _build_generated_surface_domain(spec)
    lattice_bounds = _central_lattice_bounds(spec, grip_end_length_mm)
    domain = full_domain if lattice_bounds is None else _build_generated_surface_domain(
        spec,
        xy_bounds_mm=lattice_bounds,
    )
    parameterization = parameterize_spec_lscm(spec, domain)
    design_fields = compose_design_fields_from_spec(domain, spec)
    orientation = _orientation_from_spec(domain, spec)
    phase = solve_phase_coordinates(domain, parameterization, design_fields, orientation)
    requested_boundary_mode = str(spec.lattice["boundary_mode"])
    # ``inset`` intentionally keeps all cells away from a boundary, which is
    # unsuitable for a load-bearing internal separator: it would recreate the
    # visible unconnected gap.  Partitioned parts therefore clip the central
    # honeycomb only at its shared walls.
    boundary_mode = "clip" if grip_end_length_mm > 0.0 else requested_boundary_mode
    if boundary_mode not in ("clip", "inset"):
        raise ValueError("first-version UI pipeline supports lattice.boundary_mode=clip or inset")
    requested_phase_origin = tuple(float(value) for value in spec.lattice["phase_origin"])
    boundary_phase_policy = str(spec.lattice.get("boundary_phase_policy", "manual"))
    effective_phase_origin, load_line_alignment = _resolved_phase_origin(spec, domain, phase)
    if load_line_alignment["enabled"]:
        boundary_phase_report = {
            "policy": "superseded_by_" + str(load_line_alignment.get("mode", "length_midplane_wall_alignment")),
            "requested_phase_origin": list(requested_phase_origin),
            "effective_phase_origin": list(effective_phase_origin),
        }
    elif boundary_phase_policy == "auto_avoid_outer_boundary_coincidence":
        effective_phase_origin, boundary_phase_report = choose_boundary_safe_phase_origin(
            domain, phase, requested_phase_origin
        )
    elif boundary_phase_policy == "manual":
        effective_phase_origin = requested_phase_origin
        boundary_phase_report = {
            "policy": "manual",
            "requested_phase_origin": list(requested_phase_origin),
            "effective_phase_origin": list(effective_phase_origin),
        }
    else:  # The contract loader rejects this; retain a local guard for direct specs.
        raise ValueError("unsupported lattice.boundary_phase_policy")
    geometry = generate_conformal_lattice_geometry(
        domain,
        parameterization,
        design_fields,
        orientation,
        phase,
        boundary_mode=boundary_mode,
        phase_origin=effective_phase_origin,
        random_seed=spec.random_seed,
        config_metadata={
            **spec.metadata(),
            "boundary_phase": boundary_phase_report,
            "load_line_alignment": load_line_alignment,
            "partition_connection": {
                "mode": "shared_separator_wall_endpoints" if grip_end_length_mm > 0.0 else "not_partitioned",
                "requested_boundary_mode": requested_boundary_mode,
                "effective_boundary_mode": boundary_mode,
                "honeycomb_xy_bounds_mm": list(lattice_bounds) if lattice_bounds is not None else None,
            },
        },
        load_line_alignment=load_line_alignment,
    )
    fill_validation = None
    if validate_fill_ratio:
        fill_validation = validate_realized_fill_ratio(
            domain,
            parameterization,
            design_fields,
            phase,
            geometry,
            wall_width_mm=float(spec.lattice["wall_width_mm"]),
            samples_per_triangle_side=fill_samples_per_triangle_side,
        )
    layer_embedding = _layer_embedding_for_spec(
        domain,
        orientation,
        geometry,
        spec,
        logical_layer_count,
        base_z_by_layer,
    )
    continuous_course_plan = build_continuous_course_plan(spec) if spec.part else None
    continuous_course_paths_by_layer = (
        embed_continuous_course_plan(continuous_course_plan, spec, layer_embedding)
        if continuous_course_plan is not None
        else ()
    )
    target_node_normals = _lattice_node_normals(domain, orientation, geometry)
    layer_tool_normals = _symmetric_tool_normals(target_node_normals, layer_embedding)
    boundary_orientation = orientation if domain is full_domain else _orientation_from_spec(full_domain, spec)
    outer_boundary, outer_boundary_tool_normals = (
        _layered_outer_boundary(full_domain, boundary_orientation, spec, layer_embedding)
        if spec.part
        else (None, None)
    )
    auxiliary_paths = _layered_partition_paths(spec, layer_embedding) if grip_end_length_mm > 0.0 else None
    path_graph = None if extrusion is None else build_conformal_lattice_path_graph(
        geometry,
        extrusion,
        layer_embedding=layer_embedding,
        node_normals_xyz=target_node_normals,
        wall_bead_count=int(spec.lattice.get("wall_bead_count", 1)),
        nominal_bead_width_mm=float(spec.manufacturing.get("nominal_bead_width_mm", 2.0)),
        config_metadata=spec.metadata(),
        outer_boundary_paths_xyz=outer_boundary,
        layer_tool_normals_xyz=layer_tool_normals,
        outer_boundary_tool_normals_xyz=outer_boundary_tool_normals,
        auxiliary_deposition_paths_by_layer=auxiliary_paths,
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
        continuous_course_plan=continuous_course_plan,
        continuous_course_paths_by_layer=continuous_course_paths_by_layer,
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


def _symmetric_grip_end_length(spec: ConformalLatticeSpec) -> float:
    """Read the optional geometry-only tensile grip partition from the spec."""

    if not spec.part:
        return 0.0
    value = float(spec.part.get("symmetric_grip_end_length_mm", 0.0))
    if value < 0.0 or not math.isfinite(value):
        raise ValueError("part.symmetric_grip_end_length_mm must be a finite non-negative value")
    return value


def _central_lattice_bounds(spec: ConformalLatticeSpec, grip_end_length_mm: float) -> tuple[float, float, float, float] | None:
    """Return the exact framed honeycomb region, including its connection line.

    The clipped honeycomb edge endpoints must land on the two explicit
    separator walls.  Moving this domain inward leaves a visible and physical
    gap, whereas sharing isolated endpoints with a wall creates the intended
    continuous structural joint.  Boundary-phase selection prevents a whole
    honeycomb edge from becoming collinear with the wall and being deposited
    twice.
    """

    if grip_end_length_mm <= 0.0:
        return None
    if not spec.part:
        raise ValueError("symmetric grip regions require a rectangular part")
    length = float(spec.part["length_mm"])
    width = float(spec.part["width_mm"])
    bounds = (grip_end_length_mm, 0.0, length - grip_end_length_mm, width)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("grip region and explicit wall clearance leave no printable central honeycomb area")
    return bounds


def _layered_partition_paths(
    spec: ConformalLatticeSpec,
    layer_embedding: LayerEmbedding,
) -> tuple[tuple[tuple[str, np.ndarray, np.ndarray], ...], ...]:
    """Reuse 2-D X-zigzag planning, then embed all paths with one surface law.

    No process controls are read from the JSON.  The portable config supplies
    only the grip range; the fixed resin bead width from the active conformal
    manufacturing contract supplies the planning envelope.
    """

    if not spec.part:
        raise ValueError("partition paths require a rectangular part")
    grip = _symmetric_grip_end_length(spec)
    if grip <= 0.0:
        return ()
    length = float(spec.part["length_mm"])
    width = float(spec.part["width_mm"])
    bead_width = float(spec.manufacturing["nominal_bead_width_mm"])
    # Reserve the global perimeter and both separator lines.  This avoids
    # depositing a second bead on their common edges while preserving a full
    # density resin fill within each end grip.
    left_bounds = (bead_width, bead_width, grip - bead_width, width - bead_width)
    right_bounds = (length - grip + bead_width, bead_width, length - bead_width, width - bead_width)
    if left_bounds[2] <= left_bounds[0] or right_bounds[2] <= right_bounds[0] or left_bounds[3] <= left_bounds[1]:
        raise ValueError("grip region is too narrow after reserving the perimeter and separator wall")
    left_zigzag = _horizontal_one_stroke_zigzag(left_bounds, bead_width)
    right_zigzag = _horizontal_one_stroke_zigzag(right_bounds, bead_width)
    points_per_separator = max(3, int(math.ceil(width / bead_width)) + 1)
    separator_left = np.column_stack((np.full(points_per_separator, grip), np.linspace(0.0, width, points_per_separator)))
    separator_right = np.column_stack((np.full(points_per_separator, length - grip), np.linspace(0.0, width, points_per_separator)))
    alpha = np.asarray(layer_embedding.report.get("alpha_by_layer"), dtype=np.float64)
    base_z = np.asarray(layer_embedding.report.get("base_z_by_layer_mm"), dtype=np.float64)
    if alpha.shape != base_z.shape or alpha.ndim != 1:
        raise ValueError("symmetric layer embedding is missing per-layer alpha/base-Z data")
    surface = height_field_from_spec(spec)
    result: list[tuple[tuple[str, np.ndarray, np.ndarray], ...]] = []
    for layer_alpha, layer_base_z in zip(alpha, base_z):
        result.append(
            (
                ("conformal_partition_wall", *_embed_planar_path(separator_left, surface, layer_alpha, layer_base_z)),
                ("conformal_partition_wall", *_embed_planar_path(separator_right, surface, layer_alpha, layer_base_z)),
                ("conformal_grip_zigzag_x_one_stroke", *_embed_planar_path(left_zigzag, surface, layer_alpha, layer_base_z)),
                ("conformal_grip_zigzag_x_one_stroke", *_embed_planar_path(right_zigzag, surface, layer_alpha, layer_base_z)),
            )
        )
    return tuple(result)


def _horizontal_one_stroke_zigzag(bounds: tuple[float, float, float, float], bead_width_mm: float) -> np.ndarray:
    """Adapter for the established resin zigzag planner; no duplicate planner."""

    from shapely.geometry import box
    from ..slicer import _solid_zigzag_infill_paths

    paths = _solid_zigzag_infill_paths(
        box(*bounds),
        spacing=bead_width_mm,
        line_width=bead_width_mm,
        angle_degrees=0.0,
        minimum_clearance=0.05,
        tolerance=1e-6,
        connect_adjacent=True,
        follow_boundaries=True,
    )
    if len(paths) != 1:
        raise ValueError("horizontal grip zigzag must resolve to exactly one continuous path")
    return np.asarray(paths[0], dtype=np.float64)


def _embed_planar_path(
    points_xy: np.ndarray,
    surface: HeightField,
    alpha: float,
    base_z_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply global H, smoothstep alpha, and the matching height-field normal."""

    xy = np.asarray(points_xy, dtype=np.float64)
    heights = np.asarray(surface.height(xy[:, 0], xy[:, 1]), dtype=np.float64)
    dz_dx, dz_dy = surface.gradient(xy[:, 0], xy[:, 1])
    xyz = np.column_stack((xy, base_z_mm + alpha * (heights - surface.z_reference_mm)))
    normals = np.column_stack((-alpha * dz_dx, -alpha * dz_dy, np.ones(len(xy))))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return xyz, normals


def _orientation_from_spec(domain: SurfaceMeshDomain, spec: ConformalLatticeSpec) -> OrientationField:
    if spec.orientation_field.get("mode") != "global_axis":
        raise ValueError("first-version UI pipeline supports orientation_field.mode=global_axis")
    angle = math.radians(float(spec.orientation_field.get("angle_deg", 0.0)))
    return build_orientation_field(
        domain,
        mode="global_axis",
        global_axis_xyz=np.asarray([math.cos(angle), math.sin(angle), 0.0]),
    )


def _resolved_phase_origin(
    spec: ConformalLatticeSpec,
    domain: SurfaceMeshDomain,
    phase: PhaseCoordinates,
) -> tuple[tuple[float, float], dict[str, object]]:
    """Resolve semantic X/Y honeycomb features in solved phase coordinates."""

    manual_origin = tuple(float(value) for value in spec.lattice["phase_origin"])
    request = spec.lattice.get("load_line_alignment")
    honeycomb_request = spec.lattice.get("honeycomb_feature_alignment")
    load_alignment = isinstance(request, Mapping) and request.get("enabled", False)
    explicit_feature_alignment = isinstance(honeycomb_request, Mapping)
    align_x = explicit_feature_alignment and honeycomb_request.get("align_x", False) is True
    align_y = explicit_feature_alignment and honeycomb_request.get("align_y", False) is True
    if not load_alignment and not align_x and not align_y:
        return manual_origin, {"enabled": False, "mode": "boundary_phase_policy"}

    angle_deg = float(spec.orientation_field.get("angle_deg", 0.0))
    if not math.isclose(math.sin(math.radians(angle_deg)), 0.0, abs_tol=1e-9):
        raise ValueError("honeycomb feature alignment requires a global grid direction parallel to X")

    part_center_xy = np.asarray(
        [float(spec.part["length_mm"]) / 2.0, float(spec.part["width_mm"]) / 2.0], dtype=np.float64
    )
    target_xy = np.array(part_center_xy, copy=True)
    if explicit_feature_alignment:
        target_xy[0] = float(honeycomb_request["target_x_mm"])
        target_xy[1] = float(honeycomb_request["target_y_mm"])
    target_phase = _phase_at_planar_point(domain, phase, target_xy)
    if load_alignment:
        # Historical bending semantics centre the whole local honeycomb phase
        # at the load point, not only its p coordinate.  Keep existing JSON
        # exports bit-for-bit meaningful.
        origin = target_phase - np.asarray([0.5, 0.0], dtype=np.float64)
    else:
        origin = np.asarray(manual_origin, dtype=np.float64)
        if align_x:
            origin[0] = target_phase[0] - 0.5
        if align_y:
            # In the normalized triangular-lattice phase basis, the requested
            # X-progressing zigzag centre-line is sqrt(3)/4 above the
            # cell-centre origin.
            origin[1] = target_phase[1] - math.sqrt(3.0) / 4.0
        if align_x or align_y:
            # A feature may otherwise land exactly on an LSCM-domain edge;
            # move one hundredth of a micron in phase space so inverse
            # mapping remains well-defined without a measurable shift.
            origin[1] += 1e-7
    mode = "honeycomb_center_features" if explicit_feature_alignment else "length_midplane_wall_alignment"
    return (float(origin[0]), float(origin[1])), {
        "enabled": True,
        "mode": mode,
        "align_x": bool(load_alignment or align_x),
        "align_y": bool(align_y),
        "target_xy_mm": target_xy.tolist(),
        "target_phase": target_phase.tolist(),
        "resolved_phase_origin": origin.tolist(),
        # Retained for existing bending exports and their downstream readers.
        **(
            {
                "axis": "x",
                "position": "part_length_midplane",
                "feature": "wall",
                "load_center_xy_mm": target_xy.tolist(),
                "load_center_phase": target_phase.tolist(),
            }
            if load_alignment
            else {}
        ),
    }


def _phase_at_planar_point(
    domain: SurfaceMeshDomain,
    phase: PhaseCoordinates,
    point_xy: np.ndarray,
) -> np.ndarray:
    """Interpolate solved phase coordinates at one XY point in the domain."""

    triangles = domain.vertices[domain.faces, :2]
    origin = triangles[:, 2, :]
    first = triangles[:, 0, :] - origin
    second = triangles[:, 1, :] - origin
    relative = point_xy - origin
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    valid = np.abs(determinant) > 1e-14
    first_weight = np.zeros(len(triangles), dtype=np.float64)
    second_weight = np.zeros(len(triangles), dtype=np.float64)
    first_weight[valid] = (
        relative[valid, 0] * second[valid, 1] - relative[valid, 1] * second[valid, 0]
    ) / determinant[valid]
    second_weight[valid] = (
        first[valid, 0] * relative[valid, 1] - first[valid, 1] * relative[valid, 0]
    ) / determinant[valid]
    third_weight = 1.0 - first_weight - second_weight
    matches = np.flatnonzero(
        valid & (first_weight >= -1e-10) & (second_weight >= -1e-10) & (third_weight >= -1e-10)
    )
    if len(matches) == 0:
        raise ValueError("length-midplane load centre is outside the conformal surface domain")
    face_index = int(matches[0])
    weights = np.asarray(
        [first_weight[face_index], second_weight[face_index], third_weight[face_index]], dtype=np.float64
    )
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum()
    phase_vertices = np.column_stack((phase.phi_p, phase.phi_q))[domain.faces[face_index]]
    return weights @ phase_vertices


def _layer_embedding_for_spec(
    domain: SurfaceMeshDomain,
    orientation: OrientationField,
    geometry: ConformalLatticeGeometry,
    spec: ConformalLatticeSpec,
    logical_layer_count: int,
    base_z_by_layer: np.ndarray | None,
) -> LayerEmbedding:
    embedding = spec.layer_embedding
    if spec.source_provider == "planar":
        if embedding.get("mode") != "planar_stack":
            raise ValueError("planar lattice pipeline requires layer_embedding.mode=planar_stack")
        if base_z_by_layer is None:
            raise ValueError("planar lattice pipeline requires physical layer Z positions")
        return embed_lattice_layers(
            domain,
            orientation,
            geometry,
            mode="planar_stack",
            layer_offsets_mm=base_z_by_layer,
        )
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


def _layered_outer_boundary(
    domain: SurfaceMeshDomain,
    orientation: OrientationField,
    spec: ConformalLatticeSpec,
    layer_embedding: LayerEmbedding,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the rectangular XY boundary with the selected layer schedule."""

    if len(domain.boundary_loops) != 1:
        raise ValueError("rectangular conformal production requires exactly one surface boundary loop")
    surface = height_field_from_spec(spec)
    alpha = np.asarray(layer_embedding.report.get("alpha_by_layer"), dtype=np.float64)
    base_z = np.asarray(layer_embedding.report.get("base_z_by_layer_mm"), dtype=np.float64)
    if alpha.shape != (len(layer_embedding.node_positions_xyz),) or base_z.shape != alpha.shape:
        raise ValueError("symmetric layer embedding report is missing boundary-compatible layer data")
    boundary_surface = np.asarray(domain.vertices[domain.boundary_loops[0]], dtype=np.float64)
    boundary_target_normals = np.asarray(orientation.vertex_normals_xyz[domain.boundary_loops[0]], dtype=np.float64)
    boundary_flat = np.array(boundary_surface, copy=True)
    boundary_flat[:, 2] = float(surface.z_reference_mm)
    paths = boundary_flat[None, :, :] + alpha[:, None, None] * (boundary_surface[None, :, :] - boundary_flat[None, :, :])
    paths[:, :, 2] += base_z[:, None]
    closed_paths = np.concatenate((paths, paths[:, :1, :]), axis=1)
    normals = _symmetric_tool_normals(boundary_target_normals, layer_embedding)
    return closed_paths, np.concatenate((normals, normals[:, :1, :]), axis=1)


def _symmetric_tool_normals(target_normals_xyz: np.ndarray, layer_embedding: LayerEmbedding) -> np.ndarray:
    """Interpolate height-field slopes with the same smoothstep alpha as XYZ."""

    target = np.asarray(target_normals_xyz, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or not np.all(np.isfinite(target)):
        raise ValueError("target surface normals must be a finite Nx3 array")
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    target[target[:, 2] < 0.0] *= -1.0
    if np.any(np.abs(target[:, 2]) <= 1e-9):
        raise ValueError("target surface normals are too close to horizontal for the KUKA height-field ABC convention")
    alpha = np.asarray(layer_embedding.report.get("alpha_by_layer"), dtype=np.float64)
    if alpha.shape != (len(layer_embedding.node_positions_xyz),):
        raise ValueError("symmetric layer embedding report is missing alpha_by_layer")
    slope_x = -target[:, 0] / target[:, 2]
    slope_y = -target[:, 1] / target[:, 2]
    normals = np.stack(
        (-alpha[:, None] * slope_x[None, :], -alpha[:, None] * slope_y[None, :], np.ones((len(alpha), len(target)))),
        axis=2,
    )
    return normals / np.linalg.norm(normals, axis=2, keepdims=True)


def _physical_layer_schedule(
    spec: ConformalLatticeSpec,
    requested_count: int | None,
    *,
    physical_layer_height_mm: float | None = None,
) -> tuple[int, np.ndarray]:
    """Return monotonic layer-centre Z values whose printed extent is the requested part height."""

    final_height = float(spec.part["final_height_mm"])
    nominal_height = float(spec.manufacturing["layer_height_mm"] if physical_layer_height_mm is None else physical_layer_height_mm)
    if not math.isfinite(nominal_height) or nominal_height <= 0.0:
        raise ValueError("physical_layer_height_mm must be positive and finite")
    count = int(math.ceil(final_height / nominal_height))
    if requested_count is not None and requested_count != count:
        raise ValueError("logical_layer_count must match the rectangular part final_height_mm and active physical layer height")
    thicknesses = np.full(count, nominal_height, dtype=np.float64)
    thicknesses[-1] = final_height - nominal_height * (count - 1)
    centres = np.cumsum(thicknesses) - thicknesses * 0.5
    return count, centres
