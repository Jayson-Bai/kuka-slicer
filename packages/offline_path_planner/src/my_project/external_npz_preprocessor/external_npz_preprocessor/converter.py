"""Convert external source paths into gcode_planner parsed commands."""

from __future__ import annotations

import math

import numpy as np

from path_processing_core.bspline_approximation import GlobalSplinePlanner
from path_processing_core.polynomial_interpolator import (
    sample_global_curve_iter,
    _eval_bspline_point,
    _make_open_uniform_knots,
    _split_ctrl_components,
)
from path_processing_core.types import (
    ExtrudeWait,
    GlobalCurveCommand,
    MCommand,
    MoveCommand,
    ParsedCommandList,
    Position,
    ResetECommand,
    ToolChangeCommand,
)

from .process_params import ProcessParams
from .source_npz import MaterialPath, SourceJob


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

    commands: ParsedCommandList = []
    current_pose: Position | None = None
    current_tool: int | None = None
    current_e = 0.0
    line = 0

    line = _append_startup_head_events(commands, params, line, job)
    source_min_x, source_min_y = _job_source_xy_min(job)
    first_layer_indexes = _first_material_layer_indexes(job)

    initial_travel_added = False
    initial_print_prepare_done = False

    primeline_inserted = False
    first_fiber_in_job = True

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
        ordered_paths: list[MaterialPath] = []
        ordered_paths.extend(layer.resin_paths)
        ordered_paths.extend(layer.fiber_paths)
        if ordered_paths and not primeline_inserted:
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
            if not initial_travel_added:
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

            if current_pose is not None and _distance(current_pose, first_pose) > _EPS:
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
                feedrate=feedrate,
                line=line,
                layer=layer.index,
                subtype=subtype,
            )
            curve = _validated_spline_or_polyline(print_moves, source_positions, params)
            if is_primeline:
                curve.raw = "external_npz_primeline"
            curve.time_acc_s = _time_acc_s_for_material(
                material_path.material, params
            )
            curve.layer = layer.index
            curve.subtype = subtype
            commands.append(curve)
            line += 1

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
            if is_last_resin_in_layer and resin_layer_center_xy is not None:
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
) -> tuple[list[MoveCommand], float]:
    moves: list[MoveCommand] = []
    current_e = e_start
    previous = source_positions[0]
    for offset, next_pos in enumerate(source_positions[1:]):
        delta_e = _distance(previous, next_pos) * e_per_mm
        current_e += delta_e
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


def _validated_spline_or_polyline(
    moves: list[MoveCommand],
    source_positions: list[Position],
    params: ProcessParams,
) -> GlobalCurveCommand:
    if not moves:
        return GlobalCurveCommand(
            type="PRINT",
            cmd="POLYLINE",
            start_pos=source_positions[0],
            control_points=[],
            e_val=0.0,
            delta_e=0.0,
            feedrate=0.0,
            line=0,
            raw="external_npz_empty_polyline",
            original_moves=[],
        )

    smoothed_positions = _prepare_external_smoothing_positions(source_positions, params)
    raw = (
        "external_npz_smoothed_polyline"
        if _positions_changed(source_positions, smoothed_positions)
        else "external_npz_polyline"
    )
    candidate_positions = smoothed_positions
    candidate_max_angle = _sampled_polyline_max_turn_angle_deg(
        candidate_positions, moves, params
    )
    angle_threshold = max(0.0, float(params.corner_angle_deg))
    if candidate_max_angle > angle_threshold:
        repaired_positions = _rdp_fillet_positions(source_positions, params)
        if _positions_changed(source_positions, repaired_positions):
            repaired_max_angle = _sampled_polyline_max_turn_angle_deg(
                repaired_positions, moves, params
            )
            if repaired_max_angle <= candidate_max_angle:
                candidate_positions = repaired_positions
                raw = "external_npz_curvature_smoothed_polyline"

    if not _positions_changed(source_positions, candidate_positions):
        return _make_polyline_curve(moves, raw="external_npz_polyline")

    smoothed_moves, _ = _print_moves_from_positions(
        source_positions=candidate_positions,
        e_start=moves[0].e_val - moves[0].delta_e,
        e_per_mm=_e_per_mm_from_moves(moves),
        feedrate=moves[0].feedrate,
        line=moves[0].line,
        layer=moves[0].layer,
        subtype=moves[0].subtype,
    )
    return _make_polyline_curve(smoothed_moves, raw=raw)


def _sampled_polyline_max_turn_angle_deg(
    positions: list[Position],
    moves: list[MoveCommand],
    params: ProcessParams,
) -> float:
    if len(positions) < 3 or not moves:
        return 0.0
    probe_moves, _ = _print_moves_from_positions(
        source_positions=positions,
        e_start=moves[0].e_val - moves[0].delta_e,
        e_per_mm=_e_per_mm_from_moves(moves),
        feedrate=moves[0].feedrate,
        line=moves[0].line,
        layer=moves[0].layer,
        subtype=moves[0].subtype,
    )
    curve = _make_polyline_curve(probe_moves, raw="external_npz_probe_polyline")
    sampled_points = [
        sample.pos
        for sample in sample_global_curve_iter(
            curve,
            dt=max(_EPS, float(params.dt)),
            target_velocity=max(_EPS, float(moves[0].feedrate) / 60.0),
        )
    ]
    return _polyline_max_turn_angle_deg(sampled_points, min_segment_mm=0.02)


def _rdp_fillet_positions(
    positions: list[Position],
    params: ProcessParams,
) -> list[Position]:
    if len(positions) < 3:
        return positions
    epsilon = max(0.03, min(0.2, float(params.corner_retreat_max_mm) * 0.375))
    simplified = _rdp_simplify_positions(positions, epsilon)
    if len(simplified) < 3:
        return positions
    return _blend_sharp_corners(
        simplified,
        angle_threshold_deg=max(0.0, float(params.corner_angle_deg)),
        retreat_ratio=max(0.0, min(1.0, float(params.corner_retreat_ratio))),
        retreat_max_mm=max(0.0, float(params.corner_retreat_max_mm)),
        blend_segments=max(2, int(params.corner_blend_segments)),
    )


def _rdp_simplify_positions(points: list[Position], epsilon: float) -> list[Position]:
    if len(points) <= 2 or epsilon <= _EPS:
        return points
    max_distance = -1.0
    split_index = -1
    for index, point in enumerate(points[1:-1], start=1):
        distance = _point_to_segment_distance(point, points[0], points[-1])
        if distance > max_distance:
            max_distance = distance
            split_index = index
    if max_distance > epsilon and split_index > 0:
        left = _rdp_simplify_positions(points[: split_index + 1], epsilon)
        right = _rdp_simplify_positions(points[split_index:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _fit_validated_spline(
    moves: list[MoveCommand],
    source_positions: list[Position],
    params: ProcessParams,
) -> GlobalCurveCommand | None:
    curve = GlobalSplinePlanner().fit_global_curve(
        moves,
        corner_angle_deg=float(params.corner_angle_deg),
        corner_retreat_ratio=float(params.corner_retreat_ratio),
        density=max(0, int(params.density)),
        degree=max(1, int(params.degree)),
        max_fit_points=max(2, int(params.max_fit_points_per_segment)),
    )
    if curve is None or (curve.cmd or "").upper() != "SPLINE":
        return None

    max_error = max(0.0, float(params.spline_max_error_mm))
    max_angle = max(0.0, float(params.spline_max_angle_deg))
    if _curve_max_bidirectional_error_mm(curve, source_positions) > max_error:
        return None
    if _curve_max_turn_angle_deg(
        curve,
        source_positions,
        min_segment_mm=max(_EPS, float(params.source_merge_distance_mm)),
    ) > max_angle:
        return None

    curve.type = "PRINT"
    curve.raw = "external_npz_validated_bspline"
    curve.original_moves = list(moves)
    return curve


def _make_polyline_curve(moves: list[MoveCommand], raw: str) -> GlobalCurveCommand:
    first = moves[0]
    last = moves[-1]
    return GlobalCurveCommand(
        type="PRINT",
        cmd="POLYLINE",
        start_pos=first.start_pos,
        control_points=[move.pos for move in moves],
        e_val=last.e_val,
        delta_e=sum(move.delta_e for move in moves),
        feedrate=first.feedrate,
        line=first.line,
        raw=raw,
        original_moves=list(moves),
    )


def _curve_max_bidirectional_error_mm(
    curve: GlobalCurveCommand,
    source_positions: list[Position],
) -> float:
    curve_points = _sample_spline_positions(curve, source_positions)
    if len(curve_points) < 2 or len(source_positions) < 2:
        return 0.0

    max_curve_to_source = max(
        _point_to_polyline_distance(point, source_positions)
        for point in curve_points
    )
    max_source_to_curve = max(
        _point_to_polyline_distance(point, curve_points)
        for point in source_positions
    )
    return max(max_curve_to_source, max_source_to_curve)


def _curve_max_turn_angle_deg(
    curve: GlobalCurveCommand,
    source_positions: list[Position],
    *,
    min_segment_mm: float,
) -> float:
    points = _sample_spline_positions(
        curve,
        source_positions,
        sample_count=max(128, min(1000, len(source_positions) * 4)),
    )
    return _polyline_max_turn_angle_deg(points, min_segment_mm=min_segment_mm)


def _prepare_external_smoothing_positions(
    positions: list[Position],
    params: ProcessParams,
) -> list[Position]:
    if len(positions) < 3:
        return positions
    angle_threshold = max(0.0, float(params.corner_angle_deg))
    merge_distance = max(0.0, float(params.source_merge_distance_mm))
    if _polyline_max_turn_angle_deg(
        positions, min_segment_mm=max(_EPS, min(merge_distance, 0.02))
    ) < angle_threshold:
        return positions
    merged = _merge_short_source_segments(
        positions, min_distance=merge_distance
    )
    if len(merged) < 3:
        return merged
    return _blend_sharp_corners(
        merged,
        angle_threshold_deg=angle_threshold,
        retreat_ratio=max(0.0, min(1.0, float(params.corner_retreat_ratio))),
        retreat_max_mm=max(0.0, float(params.corner_retreat_max_mm)),
        blend_segments=max(2, int(params.corner_blend_segments)),
    )


def _merge_short_source_segments(
    positions: list[Position],
    *,
    min_distance: float,
) -> list[Position]:
    if len(positions) <= 2 or min_distance <= _EPS:
        return positions
    merged = [positions[0]]
    for point in positions[1:-1]:
        if _distance(merged[-1], point) >= min_distance:
            merged.append(point)
    if _distance(merged[-1], positions[-1]) > _EPS or len(merged) == 1:
        merged.append(positions[-1])
    else:
        merged[-1] = positions[-1]
    return merged


def _blend_sharp_corners(
    positions: list[Position],
    *,
    angle_threshold_deg: float,
    retreat_ratio: float,
    retreat_max_mm: float,
    blend_segments: int,
) -> list[Position]:
    retreats = [0.0] * len(positions)
    min_fillet_radius = max(0.08, min(0.2, retreat_max_mm * 0.2))
    nominal_retreat_max = retreat_max_mm * 0.5 if retreat_max_mm > _EPS else float("inf")

    for index in range(1, len(positions) - 1):
        prev_pos = positions[index - 1]
        corner = positions[index]
        next_pos = positions[index + 1]
        in_len = _distance(prev_pos, corner)
        out_len = _distance(corner, next_pos)
        turn_angle = _turn_angle_deg(prev_pos, corner, next_pos)
        if (
            in_len > _EPS
            and out_len > _EPS
            and turn_angle >= angle_threshold_deg
            and retreat_ratio > _EPS
        ):
            shape_limit = min(in_len, out_len) * retreat_ratio
            half_angle = math.radians(min(179.0, max(0.0, turn_angle))) * 0.5
            radius_retreat = min_fillet_radius * math.tan(half_angle)
            preferred_retreat = min(shape_limit, nominal_retreat_max)
            retreats[index] = min(shape_limit, max(preferred_retreat, radius_retreat))

    for seg_index, (start, end) in enumerate(zip(positions, positions[1:])):
        seg_len = _distance(start, end)
        if seg_len <= _EPS:
            continue
        left = retreats[seg_index] if 0 < seg_index < len(positions) - 1 else 0.0
        right_index = seg_index + 1
        right = retreats[right_index] if 0 < right_index < len(positions) - 1 else 0.0
        total = left + right
        limit = seg_len * 0.9
        if total > limit and total > _EPS:
            scale = limit / total
            if left > 0.0:
                retreats[seg_index] *= scale
            if right > 0.0:
                retreats[right_index] *= scale

    out = [positions[0]]
    arc_step_deg = max(5.0, min(30.0, 90.0 / max(2, blend_segments)))
    for index in range(1, len(positions) - 1):
        prev_pos = positions[index - 1]
        corner = positions[index]
        next_pos = positions[index + 1]
        retreat = retreats[index]
        if retreat <= _EPS:
            _append_distinct_position(out, corner)
            continue

        in_len = _distance(prev_pos, corner)
        out_len = _distance(corner, next_pos)
        entry = _position_along(corner, prev_pos, retreat / in_len)
        exit_pos = _position_along(corner, next_pos, retreat / out_len)
        _append_distinct_position(out, entry)
        if not _append_circular_fillet_positions(
            out,
            entry,
            corner,
            exit_pos,
            prev_pos,
            next_pos,
            step_angle_deg=arc_step_deg,
        ):
            _append_midpoint_fillet_positions(out, entry, corner, exit_pos, blend_segments)
        _append_distinct_position(out, exit_pos)

    _append_distinct_position(out, positions[-1])
    return out


def _append_circular_fillet_positions(
    out: list[Position],
    entry: Position,
    corner: Position,
    exit_pos: Position,
    prev_pos: Position,
    next_pos: Position,
    *,
    step_angle_deg: float,
) -> bool:
    in_dx = corner.x - prev_pos.x
    in_dy = corner.y - prev_pos.y
    out_dx = next_pos.x - corner.x
    out_dy = next_pos.y - corner.y
    in_len = math.hypot(in_dx, in_dy)
    out_len = math.hypot(out_dx, out_dy)
    if in_len <= _EPS or out_len <= _EPS:
        return False

    ux = in_dx / in_len
    uy = in_dy / in_len
    vx = out_dx / out_len
    vy = out_dy / out_len
    n1x = -uy
    n1y = ux
    n2x = -vy
    n2y = vx
    rhs_x = exit_pos.x - entry.x
    rhs_y = exit_pos.y - entry.y
    det = n2x * n1y - n1x * n2y
    if abs(det) <= _EPS:
        return False

    scale = (rhs_x * (-n2y) - (-n2x) * rhs_y) / det
    center_x = entry.x + scale * n1x
    center_y = entry.y + scale * n1y
    radius = math.hypot(entry.x - center_x, entry.y - center_y)
    if radius <= _EPS:
        return False

    start_angle = math.atan2(entry.y - center_y, entry.x - center_x)
    end_angle = math.atan2(exit_pos.y - center_y, exit_pos.x - center_x)
    cross = ux * vy - uy * vx
    if cross > 0.0:
        while end_angle < start_angle:
            end_angle += 2.0 * math.pi
    else:
        while end_angle > start_angle:
            end_angle -= 2.0 * math.pi

    sweep = end_angle - start_angle
    if abs(sweep) <= _EPS:
        return False
    segment_count = max(2, int(math.ceil(abs(math.degrees(sweep)) / step_angle_deg)))
    for step in range(1, segment_count):
        t = step / segment_count
        angle = start_angle + sweep * t
        _append_distinct_position(
            out,
            Position(
                x=center_x + radius * math.cos(angle),
                y=center_y + radius * math.sin(angle),
                z=entry.z + (exit_pos.z - entry.z) * t,
                a=entry.a + (exit_pos.a - entry.a) * t,
                b=entry.b + (exit_pos.b - entry.b) * t,
                c=entry.c + (exit_pos.c - entry.c) * t,
            ),
        )
    return True


def _append_midpoint_fillet_positions(
    out: list[Position],
    entry: Position,
    corner: Position,
    exit_pos: Position,
    blend_segments: int,
) -> None:
    control1 = _position_along(entry, corner, 0.25)
    control2 = _position_along(exit_pos, corner, 0.25)
    for step in range(1, max(2, blend_segments)):
        t = step / max(2, blend_segments)
        _append_distinct_position(out, _cubic_position(entry, control1, control2, exit_pos, t))


def _polyline_max_turn_angle_deg(points: list[Position], *, min_segment_mm: float) -> float:
    max_angle = 0.0
    for prev_pos, corner, next_pos in zip(points, points[1:], points[2:]):
        if (
            _distance(prev_pos, corner) < min_segment_mm
            or _distance(corner, next_pos) < min_segment_mm
        ):
            continue
        max_angle = max(max_angle, _turn_angle_deg(prev_pos, corner, next_pos))
    return max_angle


def _turn_angle_deg(prev_pos: Position, corner: Position, next_pos: Position) -> float:
    ax = corner.x - prev_pos.x
    ay = corner.y - prev_pos.y
    az = corner.z - prev_pos.z
    bx = next_pos.x - corner.x
    by = next_pos.y - corner.y
    bz = next_pos.z - corner.z
    a_len = math.sqrt(ax * ax + ay * ay + az * az)
    b_len = math.sqrt(bx * bx + by * by + bz * bz)
    if a_len <= _EPS or b_len <= _EPS:
        return 0.0
    dot = max(-1.0, min(1.0, (ax * bx + ay * by + az * bz) / (a_len * b_len)))
    return math.degrees(math.acos(dot))


def _position_along(start: Position, end: Position, ratio: float) -> Position:
    return Position(
        x=start.x + (end.x - start.x) * ratio,
        y=start.y + (end.y - start.y) * ratio,
        z=start.z + (end.z - start.z) * ratio,
        a=start.a + (end.a - start.a) * ratio,
        b=start.b + (end.b - start.b) * ratio,
        c=start.c + (end.c - start.c) * ratio,
    )


def _cubic_position(
    start: Position,
    control1: Position,
    control2: Position,
    end: Position,
    t: float,
) -> Position:
    omt = 1.0 - t
    b0 = omt * omt * omt
    b1 = 3.0 * omt * omt * t
    b2 = 3.0 * omt * t * t
    b3 = t * t * t
    return Position(
        x=b0 * start.x + b1 * control1.x + b2 * control2.x + b3 * end.x,
        y=b0 * start.y + b1 * control1.y + b2 * control2.y + b3 * end.y,
        z=b0 * start.z + b1 * control1.z + b2 * control2.z + b3 * end.z,
        a=b0 * start.a + b1 * control1.a + b2 * control2.a + b3 * end.a,
        b=b0 * start.b + b1 * control1.b + b2 * control2.b + b3 * end.b,
        c=b0 * start.c + b1 * control1.c + b2 * control2.c + b3 * end.c,
    )


def _append_distinct_position(points: list[Position], point: Position) -> None:
    if not points or _distance(points[-1], point) > _EPS:
        points.append(point)


def _positions_changed(before: list[Position], after: list[Position]) -> bool:
    if len(before) != len(after):
        return True
    return any(_distance(left, right) > _EPS for left, right in zip(before, after))


def _e_per_mm_from_moves(moves: list[MoveCommand]) -> float:
    for move in moves:
        length = _distance(move.start_pos, move.pos)
        if length > _EPS:
            return move.delta_e / length
    return 0.0


def _sample_spline_positions(
    curve: GlobalCurveCommand,
    source_positions: list[Position],
    sample_count: int | None = None,
) -> list[Position]:
    ctrl = [curve.start_pos] + list(curve.control_points)
    degree = min(3, len(ctrl) - 1)
    if len(ctrl) < 2 or degree < 1:
        return ctrl

    if sample_count is None:
        sample_count = max(64, min(800, len(source_positions) * 2))
    knots = _make_open_uniform_knots(len(ctrl), degree)
    u_min = knots[degree]
    u_max = knots[len(ctrl)]
    ctrl_xyzabc = _split_ctrl_components(ctrl)
    span = degree
    samples: list[Position] = []
    for idx in range(sample_count + 1):
        u = u_min + (u_max - u_min) * idx / sample_count
        point, span = _eval_bspline_point(u, degree, knots, ctrl_xyzabc, len(ctrl), span)
        samples.append(point)
    return samples


def _point_to_polyline_distance(point: Position, polyline: list[Position]) -> float:
    if len(polyline) == 1:
        return _distance(point, polyline[0])
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in zip(polyline, polyline[1:])
    )


def _point_to_segment_distance(point: Position, start: Position, end: Position) -> float:
    vx = end.x - start.x
    vy = end.y - start.y
    vz = end.z - start.z
    wx = point.x - start.x
    wy = point.y - start.y
    wz = point.z - start.z
    denom = vx * vx + vy * vy + vz * vz
    if denom <= _EPS:
        return _distance(point, start)
    t = max(0.0, min(1.0, (wx * vx + wy * vy + wz * vz) / denom))
    projection = Position(
        x=start.x + vx * t,
        y=start.y + vy * t,
        z=start.z + vz * t,
        a=0.0,
        b=0.0,
        c=0.0,
    )
    return _distance(point, projection)


def _polyline_length(points: list[Position]) -> float:
    return sum(_distance(start, end) for start, end in zip(points, points[1:]))


def _job_source_xy_min(job: SourceJob) -> tuple[float, float]:
    min_x: float | None = None
    min_y: float | None = None
    for layer in job.layers:
        for material_path in [*layer.resin_paths, *layer.fiber_paths]:
            points = np.asarray(material_path.points, dtype=np.float32)
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
        points = np.asarray(material_path.points, dtype=np.float32)
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


def _time_acc_s_for_material(material: str, params: ProcessParams) -> float | None:
    if material == "F":
        value = float(params.fiber.start_accel_s)
        return value if value > 0.0 else None
    if material == "R":
        return None
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
