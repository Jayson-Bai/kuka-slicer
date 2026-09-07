"""Explicit scalar-design fields for conformal-lattice geometry.

These fields describe *target* fill ratio and cell scale.  They do not claim
to measure the actual wall-area fill ratio; that closed-loop measurement is
reserved for Gate 6.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from .mesh_domain import SurfaceMeshDomain

if TYPE_CHECKING:
    from .contracts import ConformalLatticeSpec


CurvatureMode = Literal["mean_abs", "gaussian_abs", "max_principal_abs"]


@dataclass(frozen=True, slots=True)
class ExternalScalarMetadata:
    """Provenance required for a non-generated scalar field."""

    source: str
    units: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.units.strip():
            raise ValueError("external scalar metadata requires non-empty source and units")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("external scalar metadata sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class FieldComponent:
    """One normalised density driver with an explicit coefficient."""

    name: str
    values: np.ndarray
    weight: float = 1.0
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CurvatureField:
    mean_abs: np.ndarray
    gaussian_abs: np.ndarray
    max_principal_abs: np.ndarray
    vertex_area_mm2: np.ndarray


@dataclass(frozen=True, slots=True)
class DesignFieldResult:
    """Inputs and constrained outputs for later phase-coordinate construction."""

    density_driver: np.ndarray
    unconstrained_target_fill_ratio: np.ndarray
    target_fill_ratio: np.ndarray
    target_cell_size_mm: np.ndarray
    report: dict[str, object]

    def heatmap_payload(self) -> dict[str, object]:
        """Return serialisable data for a UI without changing geometry."""

        return {
            "density_driver": self.density_driver.tolist(),
            "unconstrained_target_fill_ratio": self.unconstrained_target_fill_ratio.tolist(),
            "target_fill_ratio": self.target_fill_ratio.tolist(),
            "target_cell_size_mm": self.target_cell_size_mm.tolist(),
            "report": self.report,
        }


def constant_driver(vertex_count: int, value: float) -> np.ndarray:
    """Create a normalised, spatially constant density driver."""

    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    return _readonly(_normalised_values(np.full(vertex_count, value, dtype=np.float64), "constant driver"))


def curvature_field(domain: SurfaceMeshDomain) -> CurvatureField:
    """Estimate vertex curvature from the cotangent Laplace--Beltrami operator."""

    laplacian, vertex_area, angle_sum = cotangent_laplacian(domain)
    mean_vector = laplacian @ domain.vertices / vertex_area[:, None]
    mean_abs = 0.5 * np.linalg.norm(mean_vector, axis=1)
    boundary_vertices = {int(vertex) for loop in domain.boundary_loops for vertex in loop}
    reference_angle = np.asarray([math.pi if index in boundary_vertices else 2.0 * math.pi for index in range(len(domain.vertices))])
    gaussian_signed = (reference_angle - angle_sum) / vertex_area
    gaussian = np.abs(gaussian_signed)
    discriminant = np.maximum(mean_abs * mean_abs - gaussian_signed, 0.0)
    maximum_principal = mean_abs + np.sqrt(discriminant)
    return CurvatureField(
        mean_abs=_readonly(mean_abs),
        gaussian_abs=_readonly(gaussian),
        max_principal_abs=_readonly(maximum_principal),
        vertex_area_mm2=_readonly(vertex_area),
    )


def curvature_driver(domain: SurfaceMeshDomain, *, mode: CurvatureMode = "mean_abs") -> FieldComponent:
    """Create a [0, 1] geometry-driven density component, not an FEA proxy."""

    fields = curvature_field(domain)
    if mode == "mean_abs":
        values = fields.mean_abs
    elif mode == "gaussian_abs":
        values = fields.gaussian_abs
    elif mode == "max_principal_abs":
        values = fields.max_principal_abs
    else:
        raise ValueError("unsupported curvature driver mode")
    return FieldComponent(
        name=f"curvature:{mode}",
        values=_readonly(_minmax(values)),
        metadata={"mode": mode, "evidence_boundary": "geometric_adaptation_not_mechanical_optimization"},
    )


def roi_driver(
    domain: SurfaceMeshDomain,
    *,
    seed_vertices: Sequence[int],
    radius_mm: float,
    falloff: Literal["linear", "gaussian"] = "linear",
) -> FieldComponent:
    """Diffuse a user ROI from explicit seed vertices by mesh-geodesic distance."""

    if not seed_vertices:
        raise ValueError("ROI driver requires at least one seed vertex")
    if not math.isfinite(radius_mm) or radius_mm <= 0.0:
        raise ValueError("ROI radius_mm must be finite and positive")
    seeds = np.asarray([int(vertex) for vertex in seed_vertices], dtype=np.int64)
    if np.any(seeds < 0) or np.any(seeds >= len(domain.vertices)):
        raise ValueError("ROI seed vertex is out of range")
    graph = edge_length_graph(domain)
    distance = np.min(dijkstra(graph, directed=False, indices=seeds), axis=0)
    if falloff == "linear":
        values = np.maximum(0.0, 1.0 - distance / radius_mm)
    elif falloff == "gaussian":
        values = np.exp(-0.5 * np.square(distance / radius_mm))
    else:
        raise ValueError("ROI falloff must be linear or gaussian")
    return FieldComponent(
        name="roi",
        values=_readonly(values),
        metadata={"seed_vertices": seeds.tolist(), "radius_mm": radius_mm, "falloff": falloff, "distance_metric": "mesh_geodesic"},
    )


def external_scalar_driver(
    domain: SurfaceMeshDomain,
    values: np.ndarray,
    *,
    location: Literal["vertex", "face"],
    metadata: ExternalScalarMetadata,
) -> FieldComponent:
    """Import a scalar driver only with explicit source/unit/hash provenance."""

    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or not np.all(np.isfinite(raw)):
        raise ValueError("external scalar values must be a finite one-dimensional array")
    if location == "vertex":
        if len(raw) != len(domain.vertices):
            raise ValueError("external vertex scalar length must equal vertex count")
        vertex_values = raw
    elif location == "face":
        if len(raw) != len(domain.faces):
            raise ValueError("external face scalar length must equal face count")
        totals = np.zeros(len(domain.vertices), dtype=np.float64)
        counts = np.zeros(len(domain.vertices), dtype=np.int64)
        for face_index, face in enumerate(domain.faces):
            totals[face] += raw[face_index]
            counts[face] += 1
        vertex_values = totals / counts
    else:
        raise ValueError("external scalar location must be vertex or face")
    return FieldComponent(
        name="external_scalar",
        values=_readonly(_minmax(vertex_values)),
        metadata={"location": location, "source": metadata.source, "units": metadata.units, "sha256": metadata.sha256},
    )


def compose_design_fields(
    domain: SurfaceMeshDomain,
    *,
    wall_width_mm: float,
    eta_min: float | None = None,
    eta_max: float | None = None,
    components: Sequence[FieldComponent] = (),
    baseline_logit: float = 0.0,
    direct_target_fill_ratio: np.ndarray | None = None,
    target_cell_size_mm: float | None = None,
    smoothing_length_mm: float = 0.0,
    max_log_size_gradient: float | None = None,
    locked_vertices: np.ndarray | None = None,
) -> DesignFieldResult:
    """Combine fill drivers or use one fixed cell scale for the whole domain."""

    _positive(wall_width_mm, "wall_width_mm")
    if not math.isfinite(smoothing_length_mm) or smoothing_length_mm < 0.0:
        raise ValueError("smoothing_length_mm must be finite and non-negative")
    count = len(domain.vertices)
    locks = _locked_mask(locked_vertices, count)
    fixed_size = target_cell_size_mm is not None
    if fixed_size:
        _positive(float(target_cell_size_mm), "target_cell_size_mm")
        if components or direct_target_fill_ratio is not None:
            raise ValueError("target_cell_size_mm cannot be combined with fill-ratio drivers")
        if eta_min is not None or eta_max is not None:
            raise ValueError("target_cell_size_mm mode does not accept eta_min or eta_max")
        cell_size = np.full(count, float(target_cell_size_mm), dtype=np.float64)
        raw_eta = cell_size_to_fill_ratio(cell_size, wall_width_mm=wall_width_mm)
        if np.any(raw_eta >= 1.0):
            raise ValueError(
                "wall_width_mm and target_cell_size_mm produce a nominal fill ratio of at least 1"
            )
        smoothed = np.array(raw_eta, copy=True)
        density = np.zeros(count, dtype=np.float64)
        combination = {"mode": "fixed_target_cell_size", "target_cell_size_mm": float(target_cell_size_mm)}
        report_eta_min: float | None = None
        report_eta_max: float | None = None
    else:
        if eta_min is None or eta_max is None:
            raise ValueError("fill-ratio mode requires eta_min and eta_max")
        _positive(eta_min, "eta_min")
        _positive(eta_max, "eta_max")
        if eta_max <= eta_min or eta_max >= 1.0:
            raise ValueError("fill-ratio bounds must satisfy 0 < eta_min < eta_max < 1")
        if not math.isfinite(baseline_logit):
            raise ValueError("baseline_logit must be finite")
        if direct_target_fill_ratio is not None:
            raw_eta = np.asarray(direct_target_fill_ratio, dtype=np.float64)
            if raw_eta.shape != (count,) or not np.all(np.isfinite(raw_eta)):
                raise ValueError("direct_target_fill_ratio must be a finite per-vertex array")
            density = _minmax(raw_eta)
            combination = {"mode": "direct_target_fill_ratio"}
        else:
            combined = np.full(count, baseline_logit, dtype=np.float64)
            component_report = []
            for component in components:
                values = _normalised_values(component.values, f"component {component.name}")
                if values.shape != (count,) or not math.isfinite(component.weight):
                    raise ValueError(f"component {component.name} must match the mesh vertex count and have a finite weight")
                combined += component.weight * values
                component_report.append({"name": component.name, "weight": component.weight, "metadata": component.metadata})
            logistic = 1.0 / (1.0 + np.exp(-combined))
            raw_eta = eta_min + (eta_max - eta_min) * logistic
            density = _minmax(logistic)
            combination = {"mode": "weighted_composite", "baseline_logit": baseline_logit, "components": component_report}
        raw_eta = np.clip(raw_eta, eta_min, eta_max)
        smoothed = smooth_vertex_field(domain, raw_eta, smoothing_length_mm=smoothing_length_mm, locked_vertices=locks)
        smoothed = np.clip(smoothed, eta_min, eta_max)
        cell_size = fill_ratio_to_cell_size(smoothed, wall_width_mm=wall_width_mm)
        if max_log_size_gradient is not None:
            if not math.isfinite(max_log_size_gradient) or max_log_size_gradient <= 0.0:
                raise ValueError("max_log_size_gradient must be finite and positive")
            cell_size = limit_log_cell_size_gradient(
                domain, cell_size, max_log_gradient=max_log_size_gradient, locked_vertices=locks
            )
            smoothed = cell_size_to_fill_ratio(cell_size, wall_width_mm=wall_width_mm)
            smoothed = np.clip(smoothed, eta_min, eta_max)
            cell_size = fill_ratio_to_cell_size(smoothed, wall_width_mm=wall_width_mm)
        report_eta_min = eta_min
        report_eta_max = eta_max
    report = {
        "combination": combination,
        "wall_width_mm": wall_width_mm,
        "eta_min": report_eta_min,
        "eta_max": report_eta_max,
        "nominal_fill_ratio": float(raw_eta[0]) if fixed_size else None,
        "smoothing_length_mm": smoothing_length_mm,
        "max_log_size_gradient": max_log_size_gradient,
        "locked_vertex_count": int(np.count_nonzero(locks)),
        "unconstrained_eta_range": _range(raw_eta),
        "target_eta_range": _range(smoothed),
        "target_cell_size_mm_range": _range(cell_size),
        "smoothing_l2_delta": float(np.linalg.norm(smoothed - raw_eta)),
        "fill_ratio_note": "target value from thin-wall initial relation; actual wall-area fill ratio is validated in Gate 6",
    }
    return DesignFieldResult(
        density_driver=_readonly(density),
        unconstrained_target_fill_ratio=_readonly(raw_eta),
        target_fill_ratio=_readonly(smoothed),
        target_cell_size_mm=_readonly(cell_size),
        report=report,
    )


def compose_design_fields_from_spec(
    domain: SurfaceMeshDomain,
    spec: "ConformalLatticeSpec",
) -> DesignFieldResult:
    """Create the first-version fixed-scale field declared by a UI config."""

    if spec.fill_field.get("mode") != "fixed_cell_size":
        raise ValueError("first-version UI pipeline requires fill_field.mode=fixed_cell_size")
    return compose_design_fields(
        domain,
        wall_width_mm=float(spec.lattice["wall_width_mm"]),
        target_cell_size_mm=float(spec.lattice["base_cell_size_mm"]),
    )


def fill_ratio_to_cell_size(fill_ratio: np.ndarray, *, wall_width_mm: float) -> np.ndarray:
    """Thin-wall initial relation; never a substitute for Gate 6 measurement."""

    eta = np.asarray(fill_ratio, dtype=np.float64)
    if not np.all(np.isfinite(eta)) or np.any(eta <= 0.0):
        raise ValueError("fill ratio must be finite and positive")
    return 2.0 * wall_width_mm / (math.sqrt(3.0) * eta)


def cell_size_to_fill_ratio(cell_size_mm: np.ndarray, *, wall_width_mm: float) -> np.ndarray:
    values = np.asarray(cell_size_mm, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("cell size must be finite and positive")
    return 2.0 * wall_width_mm / (math.sqrt(3.0) * values)


def cotangent_laplacian(domain: SurfaceMeshDomain) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Return positive-semidefinite cotangent L, lumped mass, and angle sums."""

    weights: dict[tuple[int, int], float] = {}
    areas = np.zeros(len(domain.vertices), dtype=np.float64)
    angle_sum = np.zeros(len(domain.vertices), dtype=np.float64)
    for face in domain.faces:
        triangle = domain.vertices[face]
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        double_area = float(np.linalg.norm(cross))
        if double_area <= 1e-14:
            raise ValueError("cannot build a Laplacian from a degenerate triangle")
        area = 0.5 * double_area
        areas[face] += area / 3.0
        for local in range(3):
            center = triangle[local]
            first = triangle[(local + 1) % 3] - center
            second = triangle[(local + 2) % 3] - center
            angle_sum[int(face[local])] += math.acos(float(np.clip(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)), -1.0, 1.0)))
            opposite = _edge_key(int(face[(local + 1) % 3]), int(face[(local + 2) % 3]))
            weights[opposite] = weights.get(opposite, 0.0) + 0.5 * float(np.dot(first, second) / double_area)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    diagonal = np.zeros(len(domain.vertices), dtype=np.float64)
    for (left, right), weight in weights.items():
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((-weight, -weight))
        diagonal[left] += weight
        diagonal[right] += weight
    indices = np.arange(len(domain.vertices), dtype=np.int64)
    rows.extend(indices.tolist())
    columns.extend(indices.tolist())
    values.extend(diagonal.tolist())
    if np.any(areas <= 0.0):
        raise ValueError("mesh has a vertex with zero lumped area")
    return coo_matrix((values, (rows, columns)), shape=(len(indices), len(indices))).tocsr(), areas, angle_sum


def edge_length_graph(domain: SurfaceMeshDomain) -> csr_matrix:
    lengths: dict[tuple[int, int], float] = {}
    for face in domain.faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = _edge_key(int(first), int(second))
            lengths[key] = float(np.linalg.norm(domain.vertices[key[1]] - domain.vertices[key[0]]))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (left, right), length in lengths.items():
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((length, length))
    return coo_matrix((values, (rows, columns)), shape=(len(domain.vertices), len(domain.vertices))).tocsr()


def smooth_vertex_field(
    domain: SurfaceMeshDomain,
    values: np.ndarray,
    *,
    smoothing_length_mm: float,
    locked_vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Solve (M + tau L)x = Mx0 while preserving locked values exactly."""

    field = np.asarray(values, dtype=np.float64)
    if field.shape != (len(domain.vertices),) or not np.all(np.isfinite(field)):
        raise ValueError("field must be finite and defined at every vertex")
    locks = _locked_mask(locked_vertices, len(field))
    if smoothing_length_mm == 0.0:
        return np.array(field, copy=True)
    laplacian, mass, _ = cotangent_laplacian(domain)
    system = coo_matrix((mass, (np.arange(len(mass)), np.arange(len(mass)))), shape=laplacian.shape).tocsr()
    system = system + (smoothing_length_mm * smoothing_length_mm) * laplacian
    free = np.flatnonzero(~locks)
    if not len(free):
        return np.array(field, copy=True)
    locked = np.flatnonzero(locks)
    right_hand_side = mass[free] * field[free]
    if len(locked):
        right_hand_side -= system[free][:, locked] @ field[locked]
    output = np.array(field, copy=True)
    output[free] = spsolve(system[free][:, free], right_hand_side)
    return output


def limit_log_cell_size_gradient(
    domain: SurfaceMeshDomain,
    cell_size_mm: np.ndarray,
    *,
    max_log_gradient: float,
    locked_vertices: np.ndarray | None = None,
    max_iterations: int = 400,
) -> np.ndarray:
    """Project adjacent log cell sizes onto the requested relative-gradient limit."""

    values = np.asarray(cell_size_mm, dtype=np.float64)
    if values.shape != (len(domain.vertices),) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("cell_size_mm must be finite, positive, and per-vertex")
    locks = _locked_mask(locked_vertices, len(values))
    log_size = np.log(values).copy()
    edges = _mesh_edges(domain)
    for _ in range(max_iterations):
        changed = False
        for left, right, length in edges:
            limit = max_log_gradient * length
            difference = log_size[right] - log_size[left]
            if abs(difference) <= limit + 1e-12:
                continue
            if locks[left] and locks[right]:
                raise ValueError("locked vertices violate max_log_size_gradient")
            target_difference = math.copysign(limit, difference)
            if locks[left]:
                log_size[right] = log_size[left] + target_difference
            elif locks[right]:
                log_size[left] = log_size[right] - target_difference
            else:
                midpoint = 0.5 * (log_size[left] + log_size[right])
                log_size[left] = midpoint - 0.5 * target_difference
                log_size[right] = midpoint + 0.5 * target_difference
            changed = True
        if not changed:
            return np.exp(log_size)
    raise ValueError("could not satisfy max_log_size_gradient within iteration limit")


def _mesh_edges(domain: SurfaceMeshDomain) -> list[tuple[int, int, float]]:
    seen: set[tuple[int, int]] = set()
    edges = []
    for face in domain.faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = _edge_key(int(first), int(second))
            if key not in seen:
                seen.add(key)
                edges.append((key[0], key[1], float(np.linalg.norm(domain.vertices[key[1]] - domain.vertices[key[0]]))))
    return edges


def _edge_key(first: int, second: int) -> tuple[int, int]:
    if first == second:
        raise ValueError("mesh edge cannot have identical endpoints")
    return (first, second) if first < second else (second, first)


def _normalised_values(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


def _minmax(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    lower, upper = float(np.min(result)), float(np.max(result))
    return np.zeros_like(result) if upper - lower <= 1e-14 else (result - lower) / (upper - lower)


def _locked_mask(value: np.ndarray | None, count: int) -> np.ndarray:
    if value is None:
        return np.zeros(count, dtype=bool)
    result = np.asarray(value, dtype=bool)
    if result.shape != (count,):
        raise ValueError("locked_vertices must be a per-vertex Boolean array")
    return result


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _range(values: np.ndarray) -> list[float]:
    return [float(np.min(values)), float(np.max(values))]


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result
