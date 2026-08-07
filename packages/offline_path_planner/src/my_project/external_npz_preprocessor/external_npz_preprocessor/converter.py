"""Convert external source paths into gcode_planner parsed commands."""

from __future__ import annotations

import math

import numpy as np

from path_processing_core.types import (
    ExtrudeWait,
    MCommand,
    MoveCommand,
    ParsedCommandList,
    Position,
    ResetECommand,
    ToolChangeCommand,
)

from .process_params import ProcessParams
from .source_npz import (
    LayerPaths,
    MaterialPath,
    SourceJob,
    TravelPath,
    validate_layer_semantics,
)


_RESIN_GCODE_TOOL = 1
_FIBER_GCODE_TOOL = 0
_EPS = 1e-9
_PRIMELINE_ORDER = -1000000
_RESIN_LAYER_END_TRAVEL_MM = 20.0
_RESIN_LAYER_END_TRAVEL_RAW = "external_npz_resin_layer_end_travel"


def source_job_to_parsed_commands(job: SourceJob, params: ProcessParams) -> ParsedCommandList:
    travel_feed_mm_s = float(params.travel_feed_mm_s)
    if not math.isfinite(travel_feed_mm_s) or travel_feed_mm_s <= 0.0:
        raise ValueError("travel_feed_mm_s must be finite and > 0")
    _validate_logical_layer_sequence(job)

    # Brim continuity is decided before the source job reaches Core.  When the
    # optional Prusa adapter pass cannot form one safe deposited chain, retain
    # the native paths and travels; do not silently replace them with Core
    # travel bridges.
    commands: ParsedCommandList = []
    current_pose: Position | None = None
    current_tool: int | None = None
    current_e = 0.0
    line = 0

    line = _append_startup_head_events(commands, params, line, job)
    source_min_x, source_min_y = _job_source_xy_min(job)
    first_layer_indexes = _first_material_layer_indexes(job)
    source_travel_mode = any(layer.travel_paths for layer in job.layers)
    startup_travel_count = _startup_travel_count(job)

    initial_travel_added = False
    initial_print_prepare_done = False

    primeline_inserted = False
    first_fiber_in_job = True

    if source_travel_mode and startup_travel_count:
        first_layer = job.layers[0] if job.layers else None
        startup_source_min = (
            (source_min_x, source_min_y)
            if job.meta.get("startup_travel_source_frame") == "normalized_prusa"
            else (float(params.start_x_mm), float(params.start_y_mm))
        )
        if first_layer is not None:
            for travel_path in first_layer.travel_paths[:startup_travel_count]:
                line, current_pose = _append_source_travel_path(
                    commands,
                    travel_path,
                    current_pose,
                    params,
                    line,
                    first_layer.index,
                    raw="external_npz_start_xy_travel",
                    # The UI inserts this startup path after normalizing the
                    # Prusa job into the final machine frame.  Unlike source
                    # material/travel paths, it already contains the desired
                    # (0, 0) origin and must not receive the normal
                    # source-minimum -> start-XY transform a second time.
                    source_min_x=startup_source_min[0],
                    source_min_y=startup_source_min[1],
                )
            initial_travel_added = True

    # Layer iteration is intentionally driven only by the source logical key;
    # curved per-point Z is geometry, not a basis for regrouping or sorting.
    for layer in job.layers:
        resin_path_count = len(layer.resin_paths)
        resin_path_number = 0
        resin_layer_center_xy = _layer_resin_xy_center(layer.resin_paths)
        if resin_layer_center_xy is not None:
            resin_layer_center_xy = (
                resin_layer_center_xy[0] - source_min_x + float(params.start_x_mm),
                resin_layer_center_xy[1] - source_min_y + float(params.start_y_mm),
            )
        fiber_path_count = len(layer.fiber_paths)
        fiber_path_number = 0

        if (
            source_travel_mode
            and layer.resin_paths
            and current_pose is not None
            and layer.index != first_layer_indexes.get("R")
        ):
            first_source_travel = _source_travel_for_resin_path(job, layer, 0)
            if first_source_travel is not None:
                first_source_point = _offset_source_position(
                    _position_from_row(first_source_travel.points[0]),
                    params,
                    source_min_x=source_min_x,
                    source_min_y=source_min_y,
                )
                if abs(first_source_point.z - current_pose.z) > _EPS:
                    lift_target = Position(
                        x=current_pose.x,
                        y=current_pose.y,
                        z=first_source_point.z,
                        a=current_pose.a,
                        b=current_pose.b,
                        c=current_pose.c,
                    )
                    line = _append_linear_travel(
                        commands,
                        current_pose,
                        lift_target,
                        params,
                        line,
                        layer.index,
                        raw="external_npz_layer_lift",
                    )
                    current_pose = lift_target
                # When fiber paths are present, the previous layer ends at the
                # last fiber path rather than at the last resin path.  The
                # source Prusa travel intentionally begins at the previous
                # resin endpoint, so _append_source_travel_path() below adds
                # the Core fallback bridge from the actual current pose to the
                # source travel start before replaying the Prusa geometry.
        ordered_paths: list[MaterialPath] = []
        ordered_paths.extend(layer.resin_paths)
        ordered_paths.extend(layer.fiber_paths)
        if ordered_paths and params.primeline_enabled and not primeline_inserted:
            ordered_paths.insert(
                0,
                _make_resin_primeline_path(
                    source_min_x=source_min_x,
                    source_min_y=source_min_y,
                    params=params,
                ),
            )
            primeline_inserted = True
        for material_path in ordered_paths:
            is_primeline = _is_primeline_path(material_path)
            is_resin_source_path = material_path.material == "R" and not is_primeline
            is_last_resin_in_layer = (
                is_resin_source_path
                and resin_path_number == resin_path_count - 1
            )
            is_first_material_layer = (
                is_primeline
                or layer.index == first_layer_indexes.get(material_path.material)
            )
            destination_travel_feed_mm_s = _travel_feed_mm_s_for_destination(
                params, first_layer=is_first_material_layer
            )
            is_fiber = material_path.material == "F"
            is_first_fiber_in_layer = is_fiber and fiber_path_number == 0
            is_last_fiber_in_layer = (
                is_fiber and fiber_path_number == fiber_path_count - 1
            )
            tool = _tool_for_material(material_path.material)
            subtype = _subtype_for_material(material_path.material)
            first_pose = _offset_source_position(
                _position_from_row(material_path.points[0]),
                params,
                source_min_x=source_min_x,
                source_min_y=source_min_y,
            )
            skip_prusa_initial_travel = (
                is_resin_source_path
                and source_travel_mode
                and params.primeline_enabled
                and layer.index == first_layer_indexes.get("R")
                and resin_path_number == 0
            )
            source_travel = (
                _source_travel_for_resin_path(job, layer, resin_path_number)
                if is_resin_source_path
                and source_travel_mode
                and not skip_prusa_initial_travel
                else None
            )
            if source_travel is not None:
                line, current_pose = _append_source_travel_path(
                    commands,
                    source_travel,
                    current_pose,
                    params,
                    line,
                    layer.index,
                    raw="external_npz_prusa_travel",
                    source_min_x=source_min_x,
                    source_min_y=source_min_y,
                )
                initial_travel_added = True
            elif not initial_travel_added and not source_travel_mode:
                line = _append_initial_start_xy_travel(
                    commands,
                    params,
                    first_pose,
                    line,
                    layer.index,
                    feed_mm_s=destination_travel_feed_mm_s,
                )
                initial_travel_added = True
                current_pose = first_pose
            elif source_travel_mode and current_pose is None:
                current_pose = first_pose
            if current_tool != tool:
                commands.append(
                    ToolChangeCommand(
                        type="TOOL_CHANGE",
                        tool=tool,
                        line=line,
                        layer=layer.index,
                        subtype=subtype,
                        raw=f"T{tool}",
                    )
                )
                line += 1
                commands.append(
                    ResetECommand(
                        type="RESET_E",
                        val=0.0,
                        line=line,
                        layer=layer.index,
                        subtype=subtype,
                        raw="G92 E0",
                        pose=first_pose,
                    )
                )
                line += 1
                current_tool = tool
                current_e = 0.0

            if (
                current_pose is not None
                and _distance(current_pose, first_pose) > _EPS
            ):
                # Prusa owns only the source travel geometry it emitted.  If
                # that travel ends away from the deposited path start (for
                # example after an inserted fiber-layer Z offset), no source
                # motion covers the residual distance.  Make that missing
                # segment an explicit Core travel so it follows the same
                # seven-order, fixed-acceleration sampling path as every
                # other spatial motion rather than becoming a 4 ms pose jump.
                commands.append(
                    MoveCommand(
                        type="TRAVEL",
                        cmd="G0",
                        start_pos=current_pose,
                        pos=first_pose,
                        e_val=current_e,
                        delta_e=0.0,
                        feedrate=destination_travel_feed_mm_s * 60.0,
                        line=line,
                        layer=layer.index,
                        subtype="TRAVEL",
                        raw="external_npz_travel",
                    )
                )
                line += 1
                current_pose = first_pose

            e_per_mm = _e_per_mm_for_material(material_path.material, params)
            feedrate = _feed_mm_s_for_material(
                material_path.material,
                params,
                first_layer=is_first_material_layer,
            ) * 60.0
            source_positions = [
                _offset_source_position(
                    _position_from_row(row),
                    params,
                    source_min_x=source_min_x,
                    source_min_y=source_min_y,
                )
                for row in material_path.points
            ]
            source_e_profile = _normalize_source_e_profile(
                material_path.extrusion,
                len(source_positions),
            )
            previous_pose = source_positions[-1]

            if not initial_print_prepare_done:
                for wait in _path_retract_waits(material_path.material, params, line, layer.index, subtype):
                    commands.append(wait)
                    current_e += wait.delta_e
                    line += 1
                initial_print_prepare_done = True

            if is_first_fiber_in_layer:
                if first_fiber_in_job:
                    for boundary in _reset_boundary_commands(
                        params,
                        line,
                        layer.index,
                        subtype,
                        first_pose,
                        reset_raw="external_npz_fiber_prepare_reset",
                    ):
                        commands.append(boundary)
                        line += 1
                    current_e = 0.0
                    for wait in _path_retract_waits(
                        material_path.material,
                        params,
                        line,
                        layer.index,
                        subtype,
                        raw="external_npz_fiber_initial_retract",
                    ):
                        commands.append(wait)
                        current_e += wait.delta_e
                        line += 1

                for boundary in _reset_boundary_commands(
                    params,
                    line,
                    layer.index,
                    subtype,
                    first_pose,
                    reset_raw="external_npz_fiber_prime_reset",
                ):
                    commands.append(boundary)
                    line += 1
                current_e = 0.0
                for wait in _path_prime_waits(
                    material_path.material,
                    params,
                    line,
                    layer.index,
                    subtype,
                ):
                    commands.append(wait)
                    current_e += wait.delta_e
                    line += 1
                for boundary in _reset_boundary_commands(
                    params,
                    line,
                    layer.index,
                    subtype,
                    first_pose,
                    reset_raw="external_npz_fiber_print_reset",
                ):
                    commands.append(boundary)
                    line += 1
                current_e = 0.0
                first_fiber_in_job = False
            elif not is_fiber:
                for wait in _path_prime_waits(
                    material_path.material,
                    params,
                    line,
                    layer.index,
                    subtype,
                ):
                    commands.append(wait)
                    current_e += wait.delta_e
                    line += 1

            print_moves, current_e = _print_moves_from_positions(
                source_positions=source_positions,
                e_start=current_e,
                e_per_mm=e_per_mm,
                source_e_profile=source_e_profile,
                feedrate=feedrate,
                line=line,
                layer=layer.index,
                subtype=subtype,
            )
            # Print paths deliberately remain MoveCommand sequences.  Core is
            # the sole owner of corner handling, density refinement and global
            # B-spline fitting; wrapping them as an upstream POLYLINE would
            # bypass that shared planner.
            if is_primeline:
                for move in print_moves:
                    move.raw = "external_npz_primeline"
            commands.extend(print_moves)
            line += len(print_moves)

            if material_path.material == "F":
                commands.append(
                    MCommand(
                        type="M_COMMAND",
                        code="CUT",
                        params={"P": 1.0},
                        line=line,
                        layer=layer.index,
                        subtype=subtype,
                        raw="external_npz_cut",
                        tool=tool,
                    )
                )
                line += 1
                if is_last_fiber_in_layer:
                    for boundary in _reset_boundary_commands(
                        params,
                        line,
                        layer.index,
                        subtype,
                        previous_pose,
                        reset_raw="external_npz_fiber_layer_retract_reset",
                    ):
                        commands.append(boundary)
                        line += 1
                    current_e = 0.0
                    for wait in _path_retract_waits(
                        material_path.material,
                        params,
                        line,
                        layer.index,
                        subtype,
                        raw="external_npz_fiber_layer_retract",
                    ):
                        commands.append(wait)
                        current_e += wait.delta_e
                        line += 1

            if material_path.material != "F":
                for wait in _path_retract_waits(material_path.material, params, line, layer.index, subtype):
                    commands.append(wait)
                    current_e += wait.delta_e
                    line += 1

            current_pose = previous_pose
            for boundary in _path_reset_commands(
                params,
                line,
                layer.index,
                subtype,
                current_pose,
            ):
                commands.append(boundary)
                line += 1
            current_e = 0.0
            if (
                is_last_resin_in_layer
                and resin_layer_center_xy is not None
                and not source_travel_mode
            ):
                travel_target = _resin_layer_end_travel_target(
                    current_pose,
                    resin_layer_center_xy,
                    fallback_start=source_positions[-2],
                )
                commands.append(
                    MoveCommand(
                        type="TRAVEL",
                        cmd="G0",
                        start_pos=current_pose,
                        pos=travel_target,
                        e_val=0.0,
                        delta_e=0.0,
                        feedrate=destination_travel_feed_mm_s * 60.0,
                        line=line,
                        layer=layer.index,
                        subtype="TRAVEL",
                        raw=_RESIN_LAYER_END_TRAVEL_RAW,
                    )
                )
                line += 1
                current_pose = travel_target
            if is_resin_source_path:
                resin_path_number += 1
            if is_fiber:
                fiber_path_number += 1

    return commands


def _prepare_brim_one_stroke(job: SourceJob) -> SourceJob:
    """Order and orient Brim strokes for a continuous Core print sequence.

    Prusa emits Brim as several independent deposited strokes.  They must not
    be concatenated with extrusion: that would draw unwanted material across
    the Brim separation gap and would also break the per-path E reset contract.
    Instead, this stage greedily chooses the next nearest endpoint, reverses a
    stroke when that is shorter, and removes only the source travel events that
    belonged to the old Brim ordering.  The converter then emits its ordinary
    Core travel bridge between each retained stroke.
    """

    roles_root = job.meta.get("path_roles")
    if not isinstance(roles_root, dict):
        return job
    resin_roles = roles_root.get("R")
    motion_root = job.meta.get("motion_order")
    if not isinstance(resin_roles, dict) or not isinstance(motion_root, dict):
        return job

    changed = False
    new_layers: list[LayerPaths] = []
    new_motion = dict(motion_root)

    for layer in job.layers:
        roles = resin_roles.get(str(layer.index))
        if not isinstance(roles, list):
            new_layers.append(layer)
            continue
        brim_indices = [
            index
            for index, role in enumerate(roles)
            if role == "brim" and 0 <= index < len(layer.resin_paths)
        ]
        if len(brim_indices) < 2:
            new_layers.append(layer)
            continue

        selected = _nearest_brim_sequence(layer.resin_paths, brim_indices)
        reordered_paths = list(layer.resin_paths)
        old_to_new: dict[int, int] = {}
        for destination_index, (old_index, reverse) in zip(brim_indices, selected):
            reordered_paths[destination_index] = _orient_material_path(
                layer.resin_paths[old_index], reverse
            )
            old_to_new[old_index] = destination_index
        if any(old_index != destination for old_index, destination in old_to_new.items()):
            changed = True

        records = motion_root.get(str(layer.index))
        if isinstance(records, list):
            rewritten: list[dict[str, object]] = []
            for record_index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                kind = record.get("kind")
                index = record.get("index")
                if kind == "travel":
                    next_deposit = next(
                        (
                            candidate.get("index")
                            for candidate in records[record_index + 1 :]
                            if isinstance(candidate, dict)
                            and candidate.get("kind") == "deposit"
                            and isinstance(candidate.get("index"), int)
                        ),
                        None,
                    )
                    if isinstance(next_deposit, int) and next_deposit in brim_indices:
                        # Let Core generate a bridge from the actual current
                        # pose to the reordered Brim stroke.
                        continue
                    rewritten.append(record)
                elif kind == "deposit" and isinstance(index, int):
                    rewritten.append(
                        {**record, "index": old_to_new.get(index, index)}
                    )
                else:
                    rewritten.append(record)
            new_motion[str(layer.index)] = rewritten
            changed = True

        new_layers.append(
            LayerPaths(
                index=layer.index,
                resin_paths=reordered_paths,
                fiber_paths=layer.fiber_paths,
                travel_paths=layer.travel_paths,
            )
        )

    if not changed:
        return job
    meta = dict(job.meta)
    meta["motion_order"] = new_motion
    meta["brim_path_strategy"] = "nearest_endpoint_order_reverse_with_core_bridges"
    return SourceJob(meta=meta, layers=new_layers)


def _nearest_brim_sequence(
    paths: list[MaterialPath], indices: list[int]
) -> list[tuple[int, bool]]:
    remaining = set(indices)
    sequence: list[tuple[int, bool]] = []
    current: np.ndarray | None = None
    while remaining:
        if current is None:
            chosen = min(remaining)
            reverse = False
        else:
            candidates = [
                (
                    float(
                        np.linalg.norm(
                            current
                            - np.asarray(
                                paths[index].points[
                                    -1 if reverse_candidate else 0, :3
                                ],
                                dtype=np.float64,
                            )
                        )
                    ),
                    index,
                    reverse_candidate,
                )
                for index in remaining
                for reverse_candidate in (False, True)
            ]
            _, chosen, reverse = min(candidates, key=lambda item: item[0])
        sequence.append((chosen, reverse))
        remaining.remove(chosen)
        current = np.asarray(paths[chosen].points[0 if reverse else -1, :3], dtype=np.float64)
    return sequence


def _orient_material_path(path: MaterialPath, reverse: bool) -> MaterialPath:
    if not reverse:
        return path
    extrusion = None
    if path.extrusion is not None:
        values = np.asarray(path.extrusion, dtype=np.float64)
        normalized = values - values[0]
        extrusion = normalized[-1] - normalized[::-1]
    return MaterialPath(
        material=path.material,
        order=path.order,
        points=np.asarray(path.points)[::-1].copy(),
        extrusion=extrusion,
    )


def _make_resin_primeline_path(
    *,
    source_min_x: float,
    source_min_y: float,
    params: ProcessParams,
) -> MaterialPath:
    z = float(params.resin.layer_height_mm)
    a, b, c = params.default_abc
    x = float(source_min_x) + float(params.primeline_x_mm)
    y = float(source_min_y) + float(params.primeline_y_mm)
    length = max(0.0, float(params.primeline_length_mm))
    points = np.array(
        [
            [x, y, z, a, b, c],
            [x + length, y, z, a, b, c],
        ],
        dtype=np.float32,
    )
    return MaterialPath(material="R", order=_PRIMELINE_ORDER, points=points)


def _is_primeline_path(material_path: MaterialPath) -> bool:
    return material_path.material == "R" and int(material_path.order) == _PRIMELINE_ORDER


def _process_params_for_material(material: str, params: ProcessParams):
    if material == "R":
        return params.resin
    if material == "F":
        return params.fiber
    raise ValueError(f"unknown material: {material}")


def _path_prime_waits(
    material: str,
    params: ProcessParams,
    line: int,
    layer: int,
    subtype: str,
) -> list[ExtrudeWait]:
    process = _process_params_for_material(material, params)
    prime = _make_extrude_wait(
        delta_e=float(process.prime_length_mm),
        speed_mm_s=float(process.prime_speed_mm_s),
        line=line,
        layer=layer,
        subtype=subtype,
        raw="external_npz_prime",
    )
    if prime is None:
        return []

    waits = [prime]
    settle_s = float(params.prime_settle_s)
    if settle_s > 0.0:
        waits.append(
            ExtrudeWait(
                type="EXTRUDE_WAIT",
                wait_sec=settle_s,
                delta_e=0.0,
                feedrate=prime.feedrate,
                line=line + 1,
                layer=layer,
                subtype=subtype,
                raw="external_npz_prime_settle",
            )
        )
    return waits


def _reset_boundary_commands(
    params: ProcessParams,
    line: int,
    layer: int,
    subtype: str,
    pose: Position,
    *,
    reset_raw: str,
) -> list[ResetECommand | ExtrudeWait]:
    reset = ResetECommand(
        type="RESET_E",
        val=0.0,
        line=line,
        layer=layer,
        subtype=subtype,
        raw=reset_raw,
        pose=pose,
    )
    anchor = ExtrudeWait(
        type="EXTRUDE_WAIT",
        wait_sec=float(params.dt),
        delta_e=0.0,
        feedrate=float(params.travel_feed_mm_s) * 60.0,
        line=line + 1,
        layer=layer,
        subtype=subtype,
        raw="external_npz_reset_anchor",
    )
    return [reset, anchor]


def _path_reset_commands(
    params: ProcessParams,
    line: int,
    layer: int,
    subtype: str,
    pose: Position,
) -> list[ResetECommand | ExtrudeWait]:
    return _reset_boundary_commands(
        params,
        line,
        layer,
        subtype,
        pose,
        reset_raw="external_npz_path_reset",
    )


def _path_retract_waits(
    material: str,
    params: ProcessParams,
    line: int,
    layer: int,
    subtype: str,
    *,
    raw: str = "external_npz_retract",
) -> list[ExtrudeWait]:
    process = _process_params_for_material(material, params)
    retract = _make_extrude_wait(
        delta_e=-float(process.retract_length_mm),
        speed_mm_s=float(process.retract_speed_mm_s),
        line=line,
        layer=layer,
        subtype=subtype,
        raw=raw,
    )
    return [retract] if retract is not None else []


def _make_extrude_wait(
    *,
    delta_e: float,
    speed_mm_s: float,
    line: int,
    layer: int,
    subtype: str,
    raw: str,
) -> ExtrudeWait | None:
    if abs(delta_e) <= _EPS:
        return None
    if speed_mm_s <= 0.0:
        raise ValueError("extrude wait speed must be > 0")
    return ExtrudeWait(
        type="EXTRUDE_WAIT",
        wait_sec=abs(delta_e) / speed_mm_s,
        delta_e=delta_e,
        feedrate=speed_mm_s * 60.0,
        line=line,
        layer=layer,
        subtype=subtype,
        raw=raw,
    )


def _print_moves_from_positions(
    *,
    source_positions: list[Position],
    e_start: float,
    e_per_mm: float,
    feedrate: float,
    line: int,
    layer: int,
    subtype: str,
    source_e_profile: np.ndarray | None = None,
) -> tuple[list[MoveCommand], float]:
    if source_e_profile is not None:
        source_e_profile = _normalize_source_e_profile(
            source_e_profile,
            len(source_positions),
        )

    moves: list[MoveCommand] = []
    current_e = e_start
    previous = source_positions[0]
    for offset, next_pos in enumerate(source_positions[1:]):
        if source_e_profile is None:
            # Paths without a source E profile use the legacy uniform resin
            # model.  Once Prusa has supplied E, its cumulative profile is the
            # authority and e_per_mm must not regenerate or overwrite it.
            delta_e = _distance(previous, next_pos) * e_per_mm
            current_e += delta_e
        else:
            # Preserve the source E density on every geometric segment.  The
            # Core sampler later evaluates this profile at each 4 ms distance
            # sample and derives dE/dt from the actual sampled motion.
            target_e = e_start + float(source_e_profile[offset + 1])
            delta_e = target_e - current_e
            current_e = target_e
        moves.append(
            MoveCommand(
                type="PRINT",
                cmd="G1",
                start_pos=previous,
                pos=next_pos,
                e_val=current_e,
                delta_e=delta_e,
                feedrate=feedrate,
                line=line + offset,
                layer=layer,
                subtype=subtype,
                raw="external_npz_print_source",
            )
        )
        previous = next_pos
    return moves, current_e


def _normalize_source_e_profile(
    source_e_profile: np.ndarray | None,
    point_count: int,
) -> np.ndarray | None:
    if source_e_profile is None:
        return None
    profile = np.asarray(source_e_profile, dtype=np.float64)
    if profile.ndim != 1 or profile.shape[0] != point_count:
        raise ValueError(
            "source E profile must contain one value per source path point"
        )
    if not np.isfinite(profile).all():
        raise ValueError("source E profile must contain only finite values")
    if profile.shape[0] > 1 and np.any(np.diff(profile) < -1e-6):
        raise ValueError("source E profile must be cumulative and non-decreasing")
    # Prusa exports cumulative E across the entire job.  The runtime's path
    # reset makes the only meaningful source quantity the per-path profile.
    return (profile - profile[0]).astype(np.float64, copy=False)


def _job_source_xy_min(job: SourceJob) -> tuple[float, float]:
    min_x: float | None = None
    min_y: float | None = None
    for layer in job.layers:
        for material_path in [*layer.resin_paths, *layer.fiber_paths]:
            points = np.asarray(material_path.points)
            if points.size == 0:
                continue
            path_min_x = float(np.min(points[:, 0]))
            path_min_y = float(np.min(points[:, 1]))
            min_x = path_min_x if min_x is None else min(min_x, path_min_x)
            min_y = path_min_y if min_y is None else min(min_y, path_min_y)
    return (0.0 if min_x is None else min_x, 0.0 if min_y is None else min_y)


def _layer_resin_xy_center(
    resin_paths: list[MaterialPath],
) -> tuple[float, float] | None:
    """Return the XY bounding-box center for source resin geometry in one layer."""
    min_x: float | None = None
    max_x: float | None = None
    min_y: float | None = None
    max_y: float | None = None
    for material_path in resin_paths:
        points = np.asarray(material_path.points)
        if points.size == 0:
            continue
        path_min_x = float(np.min(points[:, 0]))
        path_max_x = float(np.max(points[:, 0]))
        path_min_y = float(np.min(points[:, 1]))
        path_max_y = float(np.max(points[:, 1]))
        min_x = path_min_x if min_x is None else min(min_x, path_min_x)
        max_x = path_max_x if max_x is None else max(max_x, path_max_x)
        min_y = path_min_y if min_y is None else min(min_y, path_min_y)
        max_y = path_max_y if max_y is None else max(max_y, path_max_y)
    if min_x is None or max_x is None or min_y is None or max_y is None:
        return None
    return ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)


def _resin_layer_end_travel_target(
    end_pose: Position,
    layer_center_xy: tuple[float, float],
    *,
    fallback_start: Position,
) -> Position:
    """Move 20 mm away from the layer center without changing Z or orientation."""
    dx = end_pose.x - layer_center_xy[0]
    dy = end_pose.y - layer_center_xy[1]
    xy_norm = math.hypot(dx, dy)
    if xy_norm <= _EPS:
        # A center endpoint has no radial direction; retain deterministic outward
        # motion by falling back to the final printed segment direction.
        dx = end_pose.x - fallback_start.x
        dy = end_pose.y - fallback_start.y
        xy_norm = math.hypot(dx, dy)
    if xy_norm <= _EPS:
        dx = 1.0
        dy = 0.0
        xy_norm = 1.0
    scale = _RESIN_LAYER_END_TRAVEL_MM / xy_norm
    return Position(
        x=end_pose.x + dx * scale,
        y=end_pose.y + dy * scale,
        z=end_pose.z,
        a=end_pose.a,
        b=end_pose.b,
        c=end_pose.c,
    )


def _offset_source_position(
    position: Position,
    params: ProcessParams,
    *,
    source_min_x: float,
    source_min_y: float,
) -> Position:
    return Position(
        x=position.x - source_min_x + float(params.start_x_mm),
        y=position.y - source_min_y + float(params.start_y_mm),
        z=position.z,
        a=position.a,
        b=position.b,
        c=position.c,
    )


def _append_initial_start_xy_travel(
    commands: ParsedCommandList,
    params: ProcessParams,
    first_pose: Position,
    line: int,
    layer: int,
    *,
    feed_mm_s: float,
) -> int:
    start_pose = Position(
        x=0.0,
        y=0.0,
        z=first_pose.z,
        a=float(params.default_a),
        b=float(params.default_b),
        c=float(params.default_c),
    )
    if _distance(start_pose, first_pose) <= _EPS:
        return line
    commands.append(
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=start_pose,
            pos=first_pose,
            e_val=0.0,
            delta_e=0.0,
            feedrate=float(feed_mm_s) * 60.0,
            line=line,
            layer=layer,
            subtype="TRAVEL",
            raw="external_npz_start_xy_travel",
        )
    )
    return line + 1


def _startup_travel_count(job: SourceJob) -> int:
    value = job.meta.get("startup_travel_count", 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _source_travel_for_resin_path(
    job: SourceJob,
    layer: LayerPaths,
    deposit_index: int,
) -> TravelPath | None:
    if not layer.travel_paths:
        return None
    motion_order = job.meta.get("motion_order", {})
    records = motion_order.get(str(layer.index), []) if isinstance(motion_order, dict) else []
    pending_travel: int | None = None
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            kind = record.get("kind")
            index = record.get("index")
            if kind == "travel" and isinstance(index, int):
                pending_travel = index
            elif (
                kind == "deposit"
                and index == deposit_index
                and pending_travel is not None
                and 0 <= pending_travel < len(layer.travel_paths)
            ):
                return layer.travel_paths[pending_travel]
    startup_count = _startup_travel_count(job) if layer.index == job.layers[0].index else 0
    fallback_index = deposit_index + startup_count
    if 0 <= fallback_index < len(layer.travel_paths):
        return layer.travel_paths[fallback_index]
    return None


def _append_linear_travel(
    commands: ParsedCommandList,
    start: Position,
    end: Position,
    params: ProcessParams,
    line: int,
    layer: int,
    *,
    raw: str,
) -> int:
    if _distance(start, end) <= _EPS:
        return line
    commands.append(
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=start,
            pos=end,
            e_val=0.0,
            delta_e=0.0,
            feedrate=float(params.travel_feed_mm_s) * 60.0,
            line=line,
            layer=layer,
            subtype="TRAVEL",
            raw=raw,
        )
    )
    return line + 1


def _append_source_travel_path(
    commands: ParsedCommandList,
    travel_path: TravelPath,
    current_pose: Position | None,
    params: ProcessParams,
    line: int,
    layer: int,
    *,
    raw: str,
    source_min_x: float = 0.0,
    source_min_y: float = 0.0,
) -> tuple[int, Position]:
    points = [
        _offset_source_position(
            _position_from_row(row),
            params,
            source_min_x=source_min_x,
            source_min_y=source_min_y,
        )
        for row in travel_path.points
    ]
    if len(points) < 2:
        raise ValueError("source travel path must contain at least two points")
    if current_pose is None:
        current_pose = points[0]
    else:
        line = _append_linear_travel(
            commands,
            current_pose,
            points[0],
            params,
            line,
            layer,
            raw="external_npz_travel" if raw != "external_npz_start_xy_travel" else raw,
        )
        current_pose = points[0]
    for point in points[1:]:
        line = _append_linear_travel(
            commands,
            current_pose,
            point,
            params,
            line,
            layer,
            raw=raw,
        )
        current_pose = point
    return line, current_pose


def _position_from_row(row: np.ndarray) -> Position:
    return Position(
        x=float(row[0]),
        y=float(row[1]),
        z=float(row[2]),
        a=float(row[3]),
        b=float(row[4]),
        c=float(row[5]),
    )


def _distance(a: Position, b: Position) -> float:
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)


def _tool_for_material(material: str) -> int:
    if material == "R":
        return _RESIN_GCODE_TOOL
    if material == "F":
        return _FIBER_GCODE_TOOL
    raise ValueError(f"unknown material: {material}")


def _subtype_for_material(material: str) -> str:
    if material == "R":
        return "RESIN_PRINT"
    if material == "F":
        return "FIBER_PRINT"
    raise ValueError(f"unknown material: {material}")


def _e_per_mm_for_material(material: str, params: ProcessParams) -> float:
    if material == "R":
        return params.resin.e_per_mm()
    if material == "F":
        return params.fiber.e_per_mm()
    raise ValueError(f"unknown material: {material}")


def _feed_mm_s_for_material(
    material: str,
    params: ProcessParams,
    *,
    first_layer: bool,
) -> float:
    process = _process_params_for_material(material, params)
    if first_layer:
        return float(process.first_layer_feed_mm_s)
    return float(process.feed_mm_s)


def _validate_logical_layer_sequence(job: SourceJob) -> None:
    """Keep curved jobs in explicit source-layer order before entering Core."""

    if job.meta.get("layer_semantics") is None:
        return
    validate_layer_semantics(job.meta)
    indexes = [layer.logical_layer_index for layer in job.layers]
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
        raise ValueError(
            "logical-layer source jobs must contain unique layer indexes in ascending order"
        )


def _travel_feed_mm_s_for_destination(
    params: ProcessParams,
    *,
    first_layer: bool,
) -> float:
    if first_layer:
        return float(params.first_layer_travel_feed_mm_s)
    return float(params.travel_feed_mm_s)


def _first_material_layer_indexes(job: SourceJob) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for layer in job.layers:
        if layer.resin_paths and "R" not in indexes:
            indexes["R"] = layer.index
        if layer.fiber_paths and "F" not in indexes:
            indexes["F"] = layer.index
    return indexes


def _job_materials(job: SourceJob) -> set[str]:
    materials: set[str] = set()
    for layer in job.layers:
        if layer.resin_paths:
            materials.add("R")
        if layer.fiber_paths:
            materials.add("F")
    return materials


def _append_startup_head_events(
    commands: ParsedCommandList,
    params: ProcessParams,
    line: int,
    job: SourceJob,
) -> int:
    active_materials = _job_materials(job)
    for material, code in (("R", "M106"), ("F", "M106"), ("R", "M104"), ("F", "M104")):
        if material not in active_materials:
            continue
        process = params.resin if material == "R" else params.fiber
        gcode_tool = _tool_for_material(material)
        subtype = _subtype_for_material(material)
        if code == "M104":
            if process.temperature_c <= 0:
                continue
            commands.append(
                MCommand(
                    type="M_COMMAND",
                    code="M104",
                    params={"S": float(process.temperature_c), "T": float(gcode_tool)},
                    line=line,
                    layer=0,
                    subtype=subtype,
                    raw=f"M104 T{gcode_tool} S{process.temperature_c}",
                    tool=gcode_tool,
                )
            )
        else:
            commands.append(
                MCommand(
                    type="M_COMMAND",
                    code="M106" if process.fan_enabled else "M107",
                    params={"T": float(gcode_tool)},
                    line=line,
                    layer=0,
                    subtype=subtype,
                    raw=("M106" if process.fan_enabled else "M107") + f" T{gcode_tool}",
                    tool=gcode_tool,
                )
            )
        line += 1
    return line
