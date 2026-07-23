"""
gcode_planner 的 npz 导出器（分片）.

- 解析后的指令 + 插值采样点，输出 npz 分片（二进制列存），与 RSI/uart 消费逻辑对齐。
- 事件来自 ToolChangeCommand、指定的 MCommand，以及 G92 重置挤出（ResetECommand）。
- 事件行复用事件发生前的上一帧位姿（保持不动），event_flag=1。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional
import os
import time
import json
import re
import math

from .types import (
    ParsedCommandList,
    MoveCommand,
    GlobalCurveCommand,
    ToolChangeCommand,
    MCommand,
    ResetECommand,
    ExtrudeWait,
    Position,
)
from .bspline_approximation import GlobalSplinePlanner
from .polynomial_interpolator import sample_global_curve_iter
from .rsi_timing import RsiTimingAccumulator


@dataclass
class CsvRow:
    seq: int
    x: float
    y: float
    z: float
    a: float
    b: float
    c: float
    e: float
    tool_id: int
    move_type: str
    src_line: str
    event_flag: int
    event_type: str
    payload: str
    trigger_seq: Optional[int]
    layer_index: int = 0
    total_layers: int = 0
    preview_layer_index: int = 0
    path_id: int = 0
    path_end_flag: int = 0
    planned_time_s: float = 0.0


@dataclass
class _PendingEvent:
    event_type: str
    payload: str
    src_line: int
    tool_id: int


def export_npz(
    parsed_commands: ParsedCommandList,
    output_path: str,
    dt: float = 0.004,
    chunk_size: int = 100000,
    corner_angle_deg: float = 10.0,
    corner_retreat_ratio: float = 0.2,
    density: int = 0,
    degree: int = 3,
    max_fit_points_per_segment: int = 20000,
    default_feed_mm_s: float = 10.0,
    export_sleep_ms: int = 0,
    export_yield_every: int = 0,
    split_by_layer_type: bool = False,
    plot_layer_xy: bool = False,
    plot_stride: int = 5,
    tool_offset: tuple = (0.0, 0.0, 0.0),
    progress_callback=None,
    enable_extrude_wait: bool = False,
    enable_travel_extrude_overlap: bool = True,
    resin_z_print_compensation_mm: float = 0.0,
    initial_tool_id: int = 2,
    tool_change_safe_lift_mm: float = 20.0,
    cut_lift_mm: float = 20.0,
    cut_wait_s: float = 15.0,
    fiber_retract_length_mm: float | None = None,
    external_npz_cut_absolute_e: bool = False,
) -> dict:
    """
    导出 npz（分片）.

    - 按 4ms 采样（要求上游或本函数已将运动转换为 GlobalCurveCommand）。
    - 事件对齐：事件指令出现后，标记落在随后的第一个采样点行。
    - 速度规划由 sample_global_curve 内部的七阶多项式完成，此处不做额外处理。
    返回耗时统计字典（秒），用于 CLI 打印。
    """
    t_total_start = time.perf_counter()
    timing = RsiTimingAccumulator(dt)
    timings = {
        "total_s": 0.0,
        "fit_s": 0.0,
        "fit_gen_points_s": 0.0,
        "fit_density_s": 0.0,
        "fit_prepare_data_s": 0.0,
        "fit_param_s": 0.0,
        "fit_knot_s": 0.0,
        "fit_lsq_s": 0.0,
        "fit_post_ctrl_s": 0.0,
        "fit_lsq_basis_build_s": 0.0,
        "fit_lsq_qk_build_s": 0.0,
        "fit_lsq_normal_mat_s": 0.0,
        "fit_lsq_solve_s": 0.0,
        "fit_lsq_total_s": 0.0,
        "sample_s": 0.0,
        "sample_arc_map_s": 0.0,
        "sample_lookup_s": 0.0,
        "sample_deboor_s": 0.0,
        "sample_pose_s": 0.0,
        "sample_extrude_s": 0.0,
        "write_s": 0.0,
        "manifest_s": 0.0,
        "plot_s": 0.0,
        "rows": 0,
        "parts": 0,
    }

    last_pose_map = {}
    current_tool = int(initial_tool_id)  # 1=纤维(T0), 2=树脂(T1)
    seq = 0
    planner = GlobalSplinePlanner()

    buffer: List[MoveCommand] = []
    current_type: Optional[str] = None
    current_layer: Optional[int] = None
    current_subtype: Optional[str] = None
    last_pose: Optional[CsvRow] = None
    last_feedrate_mm_min: Optional[float] = None
    resin_z_offset: float = 0.0

    def _command_layer(cmd) -> int:
        try:
            return max(0, int(getattr(cmd, "layer", 0) or 0))
        except (TypeError, ValueError):
            return 0

    total_layers = (
        max((_command_layer(cmd) for cmd in parsed_commands), default=0) + 1
        if parsed_commands else 0
    )

    # 预先定义 vocab，确保分片一致
    import numpy as np
    move_type_map = {
        "TRAVEL": 0,
        "PRINT": 1,
        "TRAVEL_FIT": 2,
        "PRINT_FIT": 3,
        "EVENT": 4,
    }
    event_type_map = {
        "": 0,
        "heat_cf": 1,
        "heat_resin": 2,
        "fan_cf": 3,
        "fan_resin": 4,
        "extrude_reset": 5,
        "tool_change_cf": 6,
        "tool_change_resin": 7,
        "cut": 8,
    }
    move_type_keys = np.array(list(move_type_map.keys()), dtype="S32")
    move_type_vals = np.array(list(move_type_map.values()), dtype=np.uint8)
    event_type_keys = np.array(list(event_type_map.keys()), dtype="S32")
    event_type_vals = np.array(list(event_type_map.values()), dtype=np.uint8)

    def _sanitize(s: str) -> str:
        out = []
        for ch in s.strip():
            if ch.isalnum():
                out.append(ch)
            elif ch in (" ", "-", "_"):
                out.append("_" if ch == " " else ch)
            elif ch == "/":
                out.append("-")
        return "".join(out) or "UNKNOWN"

    def _normalize_subtype(s: str) -> str:
        return "TRAVEL" if (not s or s == "UNKNOWN") else s

    base, ext = os.path.splitext(output_path)
    base_no_ext = base if ext.lower() == ".npz" else output_path
    base_dir = os.path.dirname(base_no_ext)
    base_name = os.path.basename(base_no_ext)
    base_name = re.sub(r"_layer_\\d{4}$", "", base_name)
    if base_dir and os.path.basename(os.path.normpath(base_dir)) == base_name:
        base_root = base_dir
    else:
        base_root = os.path.join(base_dir, base_name) if base_dir else base_name
    timing_sidecar_path = (
        os.path.join(base_root, f"{base_name}_timing.json")
        if split_by_layer_type else f"{base_no_ext}.timing.json"
    )

    class _Writer:
        def __init__(self, base_path: str):
            self.base_path = base_path
            self.part = 0
            self.wrote_any = False
            self.rows: List[CsvRow] = []
            self.last_seq: Optional[int] = None

        def add(self, row: CsvRow):
            self.rows.append(row)
            self.last_seq = row.seq
            if len(self.rows) >= chunk_size:
                self.flush()

        def flush(self):
            if not self.rows:
                return
            t0 = time.perf_counter()
            out_dir = os.path.dirname(self.base_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            out_path = f"{self.base_path}_part{self.part:04d}.npz"
            chunk = self.rows
            self.rows = []
            seq_arr = np.array([r.seq for r in chunk], dtype=np.uint32)
            x = np.array([r.x for r in chunk], dtype=np.float32)
            y = np.array([r.y for r in chunk], dtype=np.float32)
            z = np.array([r.z for r in chunk], dtype=np.float32)
            a = np.array([r.a for r in chunk], dtype=np.float32)
            b = np.array([r.b for r in chunk], dtype=np.float32)
            c = np.array([r.c for r in chunk], dtype=np.float32)
            e = np.array([r.e for r in chunk], dtype=np.float32)
            tool_id = np.array([r.tool_id for r in chunk], dtype=np.uint8)
            move_type = np.array([move_type_map.get(r.move_type, 255)
                                 for r in chunk], dtype=np.uint8)
            src_line = np.array([r.src_line for r in chunk], dtype="S32")
            event_flag = np.array([r.event_flag for r in chunk], dtype=np.uint8)
            event_type = np.array([event_type_map.get(r.event_type, 255)
                                  for r in chunk], dtype=np.uint8)
            payload = np.array([str(r.payload) for r in chunk], dtype="S32")
            trigger_seq = np.array(
                [r.trigger_seq if r.trigger_seq is not None else -1 for r in chunk],
                dtype=np.int32,
            )
            layer_index = np.array([r.layer_index for r in chunk], dtype=np.uint32)
            total_layers_arr = np.array([r.total_layers for r in chunk], dtype=np.uint32)
            preview_layer_index = np.array(
                [r.preview_layer_index for r in chunk],
                dtype=np.int32,
            )
            path_id = np.array([r.path_id for r in chunk], dtype=np.uint32)
            path_end_flag = np.array([r.path_end_flag for r in chunk], dtype=np.uint8)
            planned_time_s = np.array(
                [r.planned_time_s for r in chunk], dtype=np.float32)

            np.savez_compressed(
                out_path,
                seq=seq_arr,
                x=x,
                y=y,
                z=z,
                a=a,
                b=b,
                c=c,
                e=e,
                tool_id=tool_id,
                move_type=move_type,
                src_line=src_line,
                event_flag=event_flag,
                event_type=event_type,
                payload=payload,
                trigger_seq=trigger_seq,
                layer_index=layer_index,
                total_layers=total_layers_arr,
                preview_layer_index=preview_layer_index,
                path_id=path_id,
                path_end_flag=path_end_flag,
                planned_time_s=planned_time_s,
                move_type_vocab_keys=move_type_keys,
                move_type_vocab_vals=move_type_vals,
                event_type_vocab_keys=event_type_keys,
                event_type_vocab_vals=event_type_vals,
            )
            self.part += 1
            self.wrote_any = True
            timings["parts"] += 1
            timings["write_s"] += time.perf_counter() - t0

        def finalize(self):
            self.flush()
            if self.wrote_any and self.part == 1:
                only_part = f"{self.base_path}_part0000.npz"
                final_path = f"{self.base_path}.npz"
                if os.path.exists(only_part):
                    os.replace(only_part, final_path)

    writers = {}
    manifest = []
    manifest_by_key = {}
    occ_counters = {}
    path_id_by_segment = {}
    next_path_id = 1
    finalized_keys = set()
    plotted_layers = set()
    flat_preview_points = {}
    flat_preview_counts = {}
    flat_preview_last_e = {}
    flat_preview_needs_break = {}
    flat_preview_stride = max(1, int(plot_stride))

    def _writer_for(layer: int, subtype: str, occ: int) -> _Writer:
        if not split_by_layer_type:
            key = ("_all_", "_all_")
            if key not in writers:
                writers[key] = _Writer(base_no_ext)
            return writers[key]
        subtype = _normalize_subtype(subtype)
        key = (layer, subtype, occ)
        if key not in writers:
            safe_subtype = _sanitize(subtype)
            layer_dir = os.path.join(base_root, f"layer_{layer:04d}")
            base_path = os.path.join(
                layer_dir, f"{base_name}_layer_{layer:04d}_type_{safe_subtype}_occ_{occ:04d}")
            writers[key] = _Writer(base_path)
            entry = {
                "layer": layer,
                "type": subtype,
                "occ": occ,
                "base_path": base_path,
                "start_seq": seq,
                "end_seq": None,
            }
            manifest.append(entry)
            manifest_by_key[key] = entry
        return writers[key]

    def _ensure_segment(layer: int, subtype: str) -> int:
        nonlocal next_path_id
        key = (layer, subtype)
        occ = occ_counters.get(key, 0) + 1
        occ_counters[key] = occ
        _writer_for(layer, subtype, occ)
        path_id_by_segment[(layer, _normalize_subtype(subtype), occ)] = next_path_id
        next_path_id += 1
        return occ

    def _path_id_for(layer: int, subtype: str, occ: int) -> int:
        return path_id_by_segment.get((layer, _normalize_subtype(subtype), occ), 0)

    def _mark_path_end(layer: int, subtype: str, occ: int):
        path_id = _path_id_for(layer, subtype, occ)
        if path_id <= 0:
            return
        writer = _writer_for(layer, subtype, occ)
        last_index = None
        for idx, row in enumerate(writer.rows):
            if row.path_id == path_id:
                row.path_end_flag = 0
                last_index = idx
        if last_index is not None:
            writer.rows[last_index].path_end_flag = 1

    processed_rows = 0

    def _finalize_writer(key):
        writer = writers.get(key)
        if writer is None or key in finalized_keys:
            return
        writer.finalize()
        finalized_keys.add(key)
        entry = manifest_by_key.get(key)
        if entry is not None and writer.last_seq is not None:
            entry["end_seq"] = writer.last_seq

    def _cleanup_state_before(layer_limit: int):
        for dct in (last_pose_map, occ_counters):
            stale = [k for k in dct.keys() if isinstance(k, tuple) and k and k[0] < layer_limit]
            for key in stale:
                dct.pop(key, None)

    def _finalize_layers_before(layer_limit: int):
        if not split_by_layer_type:
            return
        target_keys = sorted([key for key in writers.keys() if isinstance(key, tuple) and len(
            key) == 3 and key[0] < layer_limit], key=lambda item: (item[0], item[1], item[2]), )
        for key in target_keys:
            _finalize_writer(key)

        if plot_layer_xy:
            completed_layers = sorted({key[0] for key in target_keys})
            for layer in completed_layers:
                if layer in plotted_layers:
                    continue
                entries = [entry for entry in manifest if int(entry.get("layer", 0)) == layer]
                if entries:
                    t0_plot = time.perf_counter()
                    _plot_single_layer(entries, base_root, stride=max(1, int(plot_stride)))
                    timings["plot_s"] += time.perf_counter() - t0_plot
                    plotted_layers.add(layer)

        _cleanup_state_before(layer_limit)

    def _accumulate_fit_profile(profile: dict):
        for key in (
            "fit_gen_points_s",
            "fit_density_s",
            "fit_prepare_data_s",
            "fit_param_s",
            "fit_knot_s",
            "fit_lsq_s",
            "fit_post_ctrl_s",
            "fit_lsq_basis_build_s",
            "fit_lsq_qk_build_s",
            "fit_lsq_normal_mat_s",
            "fit_lsq_solve_s",
            "fit_lsq_total_s",
        ):
            timings[key] += float(profile.get(key, 0.0))

    def _maybe_yield():
        nonlocal processed_rows
        if export_yield_every <= 0 or export_sleep_ms <= 0:
            return
        if processed_rows % export_yield_every == 0:
            time.sleep(export_sleep_ms / 1000.0)

    def _with_layer_progress(row: CsvRow, layer: int) -> CsvRow:
        try:
            preview_layer = int(layer)
        except (TypeError, ValueError):
            preview_layer = 0
        row.preview_layer_index = preview_layer
        row.layer_index = max(0, preview_layer)
        row.total_layers = total_layers
        return row

    def _record_flat_preview_point(layer: int, row: CsvRow):
        if split_by_layer_type or not plot_layer_xy:
            return

        last_e = flat_preview_last_e.get(layer, row.e)
        flat_preview_last_e[layer] = row.e
        is_print = row.move_type in ("PRINT", "PRINT_FIT")
        is_deposit = is_print and (row.e - last_e) > 1e-6

        if not is_deposit:
            xs_ys = flat_preview_points.get(layer)
            if xs_ys is not None and xs_ys[0]:
                flat_preview_needs_break[layer] = True
            return

        count = flat_preview_counts.get(layer, 0)
        flat_preview_counts[layer] = count + 1
        if count % flat_preview_stride != 0:
            return

        xs, ys = flat_preview_points.setdefault(layer, ([], []))
        if flat_preview_needs_break.pop(layer, False):
            if xs and not (math.isnan(xs[-1]) and math.isnan(ys[-1])):
                xs.append(float("nan"))
                ys.append(float("nan"))
        xs.append(row.x)
        ys.append(row.y)

    def _append_sample(
            gc: GlobalCurveCommand, layer: int, subtype: str, occ: int, mark_path_end: bool = True):
        nonlocal seq, last_feedrate_mm_min, processed_rows, last_pose_map, last_pose
        t0 = time.perf_counter()
        sample_profile = {
            "sample_arc_map_s": 0.0,
            "sample_lookup_s": 0.0,
            "sample_deboor_s": 0.0,
            "sample_pose_s": 0.0,
            "sample_extrude_s": 0.0,
        }
        feed_mm_min = gc.feedrate if (
            gc.feedrate is not None and gc.feedrate > 0) else last_feedrate_mm_min
        if feed_mm_min is None or feed_mm_min <= 0:
            target_velocity = default_feed_mm_s
        else:
            target_velocity = feed_mm_min / 60.0
        if gc.feedrate is not None and gc.feedrate > 0:
            last_feedrate_mm_min = gc.feedrate
        has_any = False
        move_lines: List[int] = [
            m.line for m in gc.original_moves] if gc.original_moves else [
            gc.line]
        if len(move_lines) > 1:
            src_lines = f"{move_lines[0]}-{move_lines[-1]}"
        else:
            src_lines = str(move_lines[0])
        sampled_rows: List[CsvRow] = []
        path_id = _path_id_for(layer, subtype, occ)
        sample_kwargs = {
            "dt": dt,
            "target_velocity": target_velocity,
            "profile": sample_profile,
        }
        time_acc_s = getattr(gc, "time_acc_s", None)
        if time_acc_s is not None and float(time_acc_s) > 0.0:
            sample_kwargs["t_acc"] = float(time_acc_s)
        t_acc_value = float(time_acc_s) if time_acc_s is not None and float(time_acc_s) > 0.0 else 2.0
        for pt in sample_global_curve_iter(gc, **sample_kwargs):
            if not has_any:
                timing.start_segment(path_id=path_id, move_type=gc.type, start_seq=seq)
            has_any = True
            planned_time_s = timing.append_trajectory_time()
            row = CsvRow(
                seq=seq,
                x=pt.pos.x,
                y=pt.pos.y,
                z=pt.pos.z,
                a=pt.pos.a,
                b=pt.pos.b,
                c=pt.pos.c,
                e=pt.e,
                tool_id=current_tool,
                move_type=gc.type,
                src_line=src_lines,
                event_flag=0,
                event_type="",
                payload="",
                trigger_seq=None,
                path_id=path_id,
                planned_time_s=planned_time_s,
            )
            sampled_rows.append(_with_layer_progress(row, layer))
            seq += 1

        if sampled_rows:
            timing.finish_segment(
                t_acc_s=t_acc_value,
                t_flat_s=max(0.0, float(pt.t) - t_acc_value - 2.0),
                t_dec_s=2.0,
                end_seq=sampled_rows[-1].seq,
            )
            writer = _writer_for(layer, subtype, occ)
            for row in sampled_rows:
                writer.add(row)
                _record_flat_preview_point(layer, row)
                processed_rows += 1
                _maybe_yield()
                last_pose_map[(layer, subtype)] = row
                last_pose = row
            if mark_path_end:
                _mark_path_end(layer, subtype, occ)
        timings["sample_s"] += time.perf_counter() - t0
        timings["sample_arc_map_s"] += sample_profile["sample_arc_map_s"]
        timings["sample_lookup_s"] += sample_profile["sample_lookup_s"]
        timings["sample_deboor_s"] += sample_profile["sample_deboor_s"]
        timings["sample_pose_s"] += sample_profile["sample_pose_s"]
        timings["sample_extrude_s"] += sample_profile["sample_extrude_s"]
        if not has_any:
            return

    current_occ: Optional[int] = None
    resin_z_compensation_appended = False

    def _move_length(move: MoveCommand) -> float:
        dx = move.pos.x - move.start_pos.x
        dy = move.pos.y - move.start_pos.y
        dz = move.pos.z - move.start_pos.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _can_overlap_on_travel(cmd) -> bool:
        return (
            isinstance(cmd, MoveCommand)
            and cmd.type == "TRAVEL"
            and not cmd.is_pure_state_change
            and _move_length(cmd) > 1e-9
        )

    def _add_extrude_overlap_to_travel(
            move: MoveCommand, delta_e: float, label: str) -> MoveCommand:
        return replace(
            move,
            e_val=move.e_val + delta_e,
            delta_e=move.delta_e + delta_e,
            raw=(move.raw or "") + f" | overlap_{label}",
        )

    def _overlap_extrude_waits_on_travel(commands: ParsedCommandList) -> ParsedCommandList:
        out: ParsedCommandList = []
        total = len(commands)
        for idx, cmd in enumerate(commands):
            if not isinstance(cmd, ExtrudeWait) or abs(cmd.delta_e) <= 1e-9:
                out.append(cmd)
                continue

            prev = out[-1] if out else None
            next_cmd = commands[idx + 1] if idx + 1 < total else None
            if (
                cmd.delta_e < 0.0
                and _can_overlap_on_travel(prev)
                and isinstance(next_cmd, ResetECommand)
            ):
                out[-1] = _add_extrude_overlap_to_travel(prev, cmd.delta_e, "retract")
                continue

            out.append(cmd)
        return out

    def _collapse_moves_to_single(moves: List[MoveCommand]) -> MoveCommand:
        first = moves[0]
        last = moves[-1]
        return MoveCommand(
            type=first.type,
            cmd=first.cmd,
            start_pos=first.start_pos,
            pos=last.pos,
            e_val=last.e_val,
            delta_e=sum(
                m.delta_e for m in moves),
            feedrate=first.feedrate if (
                first.feedrate is not None and first.feedrate > 0) else last.feedrate,
            line=first.line,
            layer=first.layer,
            subtype=first.subtype,
            raw=(
                first.raw or "") +
            " | compact_endpoint_comp",
            target_v_in=first.target_v_in,
            target_v_out=last.target_v_out,
            is_pure_state_change=False,
        )

    def _make_linear_move_like(
            start_pos,
            end_pos,
            template: MoveCommand,
            delta_e: float,
            e_val: float,
            raw_suffix: str) -> MoveCommand:
        return MoveCommand(
            type=template.type,
            cmd=template.cmd,
            start_pos=start_pos,
            pos=end_pos,
            e_val=e_val,
            delta_e=delta_e,
            feedrate=template.feedrate,
            line=template.line,
            layer=template.layer,
            subtype=template.subtype,
            raw=(template.raw or "") + raw_suffix,
            target_v_in=template.target_v_in,
            target_v_out=template.target_v_out,
            is_pure_state_change=False,
        )

    def _moves_bbox_diag(moves: List[MoveCommand]) -> float:
        pts = [moves[0].start_pos] + [m.pos for m in moves]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        zs = [p.z for p in pts]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        dz = max(zs) - min(zs)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _short_move_threshold(lengths: List[float]) -> float:
        positive = sorted(v for v in lengths if v > 1e-9)
        if not positive:
            return 0.8
        mid = len(positive) // 2
        median = positive[mid] if len(positive) % 2 == 1 else 0.5 * \
            (positive[mid - 1] + positive[mid])
        return min(2.5, max(0.8, median * 0.05))

    def _should_force_linear_segment(moves: List[MoveCommand], short_threshold: float) -> bool:
        if len(moves) > 8:
            return False
        lengths = [_move_length(m) for m in moves]
        positive = [v for v in lengths if v > 1e-9]
        if len(positive) < 2:
            return True
        min_len = min(positive)
        max_len = max(positive)
        return min_len <= short_threshold and max_len >= min_len * 8.0

    def _partition_moves_for_export(moves: List[MoveCommand]):
        overlap_indices = [
            idx for idx, move in enumerate(moves)
            if move.type == "TRAVEL" and abs(move.delta_e) > 1e-9
        ]
        if overlap_indices:
            parts = []
            start = 0
            for idx in overlap_indices:
                if start < idx:
                    parts.extend(_partition_moves_for_export(moves[start:idx]))
                parts.append(([moves[idx]], True))
                start = idx + 1
            if start < len(moves):
                parts.extend(_partition_moves_for_export(moves[start:]))
            return parts

        if len(moves) <= 2:
            return [(moves, True)]

        lengths = [_move_length(m) for m in moves]
        short_threshold = _short_move_threshold(lengths)
        start_idx = 0
        end_idx = len(moves)
        parts = []

        start_short = 0
        while start_short < len(lengths) and lengths[start_short] <= short_threshold:
            start_short += 1
        if start_short >= 3:
            cluster = moves[:start_short]
            if _moves_bbox_diag(cluster) <= short_threshold * 4.0:
                parts.append((cluster, True))
                start_idx = start_short

        end_short = 0
        while end_short < (end_idx -
                           start_idx) and lengths[end_idx -
                                                  end_short -
                                                  1] <= short_threshold:
            end_short += 1
        tail_part = None
        if end_short >= 3:
            cluster = moves[end_idx - end_short:end_idx]
            if _moves_bbox_diag(cluster) <= short_threshold * 4.0:
                tail_part = (cluster, True)
                end_idx -= end_short

        if start_idx < end_idx:
            if start_idx < len(lengths) - 1:
                first_len = lengths[start_idx]
                next_len = lengths[start_idx + 1]
                if first_len <= short_threshold and next_len >= max(
                        first_len * 8.0, short_threshold * 3.0):
                    parts.append((moves[start_idx:start_idx + 1], True))
                    start_idx += 1

        if start_idx < end_idx:
            if end_idx - 1 > start_idx:
                last_len = lengths[end_idx - 1]
                prev_len = lengths[end_idx - 2]
                if last_len <= short_threshold and prev_len >= max(
                        last_len * 8.0, short_threshold * 3.0):
                    tail_part = (moves[end_idx - 1:end_idx],
                                 True) if tail_part is None else tail_part
                    end_idx -= 1

        middle = moves[start_idx:end_idx]
        if middle:
            parts.append((middle, _should_force_linear_segment(middle, short_threshold)))
        if tail_part is not None:
            parts.append(tail_part)
        return [part for part in parts if part[0]]

    def _sanitize_solid_infill_endpoints(moves: List[MoveCommand]) -> List[MoveCommand]:
        if not moves:
            return moves
        subtype = (moves[0].subtype or "").strip().lower()
        if subtype != "solid infill" or len(moves) < 6:
            return moves

        lengths = [_move_length(m) for m in moves]
        short_limit = 0.6
        long_limit = 2.5
        max_cluster = 6
        min_cluster = 3

        def _prefix_cluster_end(seq_lengths):
            idx = 0
            while idx < min(len(seq_lengths), max_cluster) and seq_lengths[idx] <= short_limit:
                idx += 1
            if idx < min_cluster or idx >= len(seq_lengths):
                return 0
            if seq_lengths[idx] < long_limit:
                return 0
            return idx

        def _suffix_cluster_start(seq_lengths):
            idx = len(seq_lengths) - 1
            count = 0
            while idx >= 0 and count < max_cluster and seq_lengths[idx] <= short_limit:
                idx -= 1
                count += 1
            if count < min_cluster or idx < 0:
                return len(seq_lengths)
            if seq_lengths[idx] < long_limit:
                return len(seq_lengths)
            return idx + 1

        prefix_end = _prefix_cluster_end(lengths)
        suffix_start = _suffix_cluster_start(lengths)
        if prefix_end == 0 and suffix_start == len(moves):
            return moves

        out: List[MoveCommand] = []
        left = 0
        right = len(moves)
        if prefix_end > 0:
            out.append(_collapse_moves_to_single(moves[:prefix_end]))
            left = prefix_end
        if suffix_start < len(moves) and suffix_start > left:
            right = suffix_start
        out.extend(moves[left:right])
        if suffix_start < len(moves) and suffix_start >= left:
            out.append(_collapse_moves_to_single(moves[suffix_start:]))
        return out

    def _is_wall_outline_subtype(subtype: str) -> bool:
        normalized = (subtype or "").strip().lower().replace("_", "-")
        return normalized in {
            "wall",
            "wall-outer",
            "wall-inner",
            "outer wall",
            "inner wall",
            "perimeter",
            "external perimeter",
            "internal perimeter",
        }

    def _should_disable_spline_for_subtype(subtype: str) -> bool:
        return (subtype or "").strip().lower() == "solid infill"

    def _can_merge_collinear_moves(
            moves: List[MoveCommand],
            candidate: MoveCommand,
            tolerance: float = 0.05,
            min_cos: float = math.cos(math.radians(5.0))) -> bool:
        if not moves:
            return False
        first = moves[0]
        if (
            candidate.type != first.type
            or candidate.layer != first.layer
            or candidate.subtype != first.subtype
        ):
            return False
        if math.dist((moves[-1].pos.x, moves[-1].pos.y, moves[-1].pos.z),
                     (candidate.start_pos.x, candidate.start_pos.y, candidate.start_pos.z)) > 1e-6:
            return False

        prev = moves[-1]
        prev_vec = (
            prev.pos.x - prev.start_pos.x,
            prev.pos.y - prev.start_pos.y,
            prev.pos.z - prev.start_pos.z,
        )
        cand_vec = (
            candidate.pos.x - candidate.start_pos.x,
            candidate.pos.y - candidate.start_pos.y,
            candidate.pos.z - candidate.start_pos.z,
        )
        prev_len = math.sqrt(sum(v * v for v in prev_vec))
        cand_len = math.sqrt(sum(v * v for v in cand_vec))
        if prev_len <= 1e-9 or cand_len <= 1e-9:
            return True
        dot = sum(a * b for a, b in zip(prev_vec, cand_vec)) / (prev_len * cand_len)
        if dot < min_cos:
            return False

        start = first.start_pos
        end = candidate.pos
        axis = (end.x - start.x, end.y - start.y, end.z - start.z)
        axis_len = math.sqrt(sum(v * v for v in axis))
        if axis_len <= 1e-9:
            return False

        for point in [m.pos for m in moves] + [candidate.pos]:
            rel = (point.x - start.x, point.y - start.y, point.z - start.z)
            projection = sum(a * b for a, b in zip(rel, axis)) / axis_len
            if projection < -tolerance or projection > axis_len + tolerance:
                return False
            cross = (
                rel[1] * axis[2] - rel[2] * axis[1],
                rel[2] * axis[0] - rel[0] * axis[2],
                rel[0] * axis[1] - rel[1] * axis[0],
            )
            distance = math.sqrt(sum(v * v for v in cross)) / axis_len
            if distance > tolerance:
                return False
        return True

    def _merge_collinear_wall_moves(moves: List[MoveCommand]) -> List[MoveCommand]:
        if not moves or not _is_wall_outline_subtype(moves[0].subtype):
            return moves
        merged: List[MoveCommand] = []
        group: List[MoveCommand] = []

        def flush_group():
            nonlocal group
            if not group:
                return
            merged.append(_collapse_moves_to_single(group) if len(group) > 1 else group[0])
            group = []

        for move in moves:
            if not group:
                group = [move]
                continue
            if _can_merge_collinear_moves(group, move):
                group.append(move)
            else:
                flush_group()
                group = [move]
        flush_group()
        return merged

    def _rebuild_solid_infill_core(moves: List[MoveCommand]) -> List[MoveCommand]:
        if not moves:
            return moves
        subtype = (moves[0].subtype or "").strip().lower()
        if subtype != "solid infill":
            return moves

        def _is_main_diag(move: MoveCommand) -> bool:
            dx = move.pos.x - move.start_pos.x
            dy = move.pos.y - move.start_pos.y
            length = math.hypot(dx, dy)
            if length < 5.0:
                return False
            return abs(abs(dx) - abs(dy)) <= max(0.8, length * 0.18)

        diag_idx = [i for i, m in enumerate(moves) if _is_main_diag(m)]
        if len(diag_idx) < 4:
            return moves

        rebuilt: List[MoveCommand] = []
        carry_delta = sum(m.delta_e for m in moves[:diag_idx[0]])
        prev_diag = None

        for idx_pos, idx in enumerate(diag_idx):
            diag = moves[idx]
            if prev_diag is None:
                first_delta = diag.delta_e + carry_delta
                rebuilt.append(
                    _make_linear_move_like(
                        diag.start_pos,
                        diag.pos,
                        diag,
                        first_delta,
                        diag.e_val + carry_delta,
                        " | rebuilt_infill_diag",
                    )
                )
                prev_diag = diag
                carry_delta = 0.0
                continue

            between = moves[diag_idx[idx_pos - 1] + 1:idx]
            bridge_delta = sum(m.delta_e for m in between)
            if between:
                bridge_template = between[0]
                bridge_start = rebuilt[-1].pos
                bridge_end = diag.start_pos
                if math.hypot(bridge_end.x - bridge_start.x, bridge_end.y - bridge_start.y) > 1e-9:
                    rebuilt.append(
                        _make_linear_move_like(
                            bridge_start,
                            bridge_end,
                            bridge_template,
                            bridge_delta,
                            rebuilt[-1].e_val + bridge_delta,
                            " | rebuilt_infill_bridge",
                        )
                    )
                else:
                    rebuilt[-1].delta_e += bridge_delta
                    rebuilt[-1].e_val += bridge_delta

            rebuilt.append(
                _make_linear_move_like(
                    diag.start_pos,
                    diag.pos,
                    diag,
                    diag.delta_e,
                    (rebuilt[-1].e_val if rebuilt else 0.0) + diag.delta_e,
                    " | rebuilt_infill_diag",
                )
            )
            prev_diag = diag

        tail_moves = moves[diag_idx[-1] + 1:]
        tail_delta = sum(m.delta_e for m in tail_moves)
        if rebuilt and tail_delta:
            rebuilt[-1].delta_e += tail_delta
            rebuilt[-1].e_val += tail_delta
        return rebuilt if len(rebuilt) >= 2 else moves

    def _curve_is_pathological(gc: GlobalCurveCommand, moves: List[MoveCommand]) -> bool:
        ctrl = [gc.start_pos] + gc.control_points
        if len(ctrl) < 2 or not moves:
            return False

        orig_pts = [moves[0].start_pos] + [m.pos for m in moves]
        orig_x = [p.x for p in orig_pts]
        orig_y = [p.y for p in orig_pts]
        orig_z = [p.z for p in orig_pts]
        ctrl_x = [p.x for p in ctrl]
        ctrl_y = [p.y for p in ctrl]
        ctrl_z = [p.z for p in ctrl]

        span_x = max(orig_x) - min(orig_x)
        span_y = max(orig_y) - min(orig_y)
        span_z = max(orig_z) - min(orig_z)
        margin = max(5.0, 0.5 * max(span_x, span_y, span_z, 1.0))

        bbox_bad = (
            min(ctrl_x) < min(orig_x) - margin or
            max(ctrl_x) > max(orig_x) + margin or
            min(ctrl_y) < min(orig_y) - margin or
            max(ctrl_y) > max(orig_y) + margin or
            min(ctrl_z) < min(orig_z) - margin or
            max(ctrl_z) > max(orig_z) + margin
        )
        if bbox_bad:
            return True

        orig_len = 0.0
        for a, b in zip(orig_pts, orig_pts[1:]):
            orig_len += math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)
        if orig_len <= 1e-9:
            return False

        ctrl_len = 0.0
        for a, b in zip(ctrl, ctrl[1:]):
            ctrl_len += math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)
        return ctrl_len > orig_len * 4.0

    def flush_moves():
        nonlocal buffer, current_type, current_layer, current_subtype, current_occ
        if not buffer:
            return

        def _make_gc(move):
            return GlobalCurveCommand(
                type=move.type,
                cmd="SPLINE",
                start_pos=move.start_pos,
                control_points=[move.pos, move.pos, move.pos],  # 起点+重复终点，表示直线
                e_val=move.e_val,
                delta_e=move.delta_e,
                feedrate=move.feedrate,
                line=move.line,
                raw=move.raw or "fallback_linear",
                constraints=[],
                original_moves=[move],
            )

        def _make_polyline_gc(moves, raw_suffix):
            first = moves[0]
            last = moves[-1]
            return GlobalCurveCommand(
                type=first.type,
                cmd="POLYLINE",
                start_pos=first.start_pos,
                control_points=[move.pos for move in moves],
                e_val=last.e_val,
                delta_e=sum(move.delta_e for move in moves),
                feedrate=first.feedrate,
                line=first.line,
                raw=(first.raw or "") + raw_suffix,
                constraints=[],
                original_moves=list(moves),
            )

        work_buffer = _merge_collinear_wall_moves(
            _rebuild_solid_infill_core(_sanitize_solid_infill_endpoints(buffer))
        )
        if work_buffer and _is_wall_outline_subtype(work_buffer[0].subtype):
            gc_list = [_make_polyline_gc(work_buffer, " | wall_polyline")]
        elif work_buffer and _should_disable_spline_for_subtype(work_buffer[0].subtype):
            t0 = time.perf_counter()
            gc = planner.fit_global_curve(
                work_buffer,
                corner_angle_deg=corner_angle_deg,
                corner_retreat_ratio=corner_retreat_ratio,
                density=density,
                degree=degree,
                max_fit_points=max_fit_points_per_segment,
            )
            timings["fit_s"] += time.perf_counter() - t0
            _accumulate_fit_profile(planner.last_fit_profile)
            gc_list = [_make_gc(move) for move in work_buffer] if (
                gc is None or _curve_is_pathological(gc, work_buffer)) else [gc]
        else:
            gc_list = []
            for segment_moves, force_linear in _partition_moves_for_export(work_buffer):
                if force_linear and len(segment_moves) >= 3:
                    gc_list.append(
                        _make_polyline_gc(segment_moves, " | short_cluster_polyline")
                    )
                    continue
                if force_linear or len(segment_moves) <= 2:
                    gc_list.extend(_make_gc(move) for move in segment_moves)
                    continue

                t0 = time.perf_counter()
                gc = planner.fit_global_curve(
                    segment_moves,
                    corner_angle_deg=corner_angle_deg,
                    corner_retreat_ratio=corner_retreat_ratio,
                    density=density,
                    degree=degree,
                    max_fit_points=max_fit_points_per_segment,
                )
                timings["fit_s"] += time.perf_counter() - t0
                _accumulate_fit_profile(planner.last_fit_profile)
                if gc is None or _curve_is_pathological(gc, segment_moves):
                    gc_list.extend(_make_gc(move) for move in segment_moves)
                else:
                    gc_list.append(gc)

        layer = buffer[0].layer if buffer else 0
        subtype = buffer[0].subtype if buffer else "UNKNOWN"
        occ = current_occ if current_occ is not None else _ensure_segment(layer, subtype)
        for idx, gc in enumerate(gc_list):
            _append_sample(gc, layer, subtype, occ, mark_path_end=(idx == len(gc_list) - 1))
        buffer = []
        current_type = None
        current_layer = None
        current_subtype = None
        current_occ = None

    def _append_resin_z_print_compensation(layer: int, line: int):
        nonlocal resin_z_compensation_appended, resin_z_offset
        if resin_z_compensation_appended:
            return
        resin_z_compensation_appended = True
        if abs(resin_z_print_compensation_mm) <= 1e-9:
            return
        base = last_pose
        if base is None:
            base = CsvRow(seq=0, x=0.0, y=0.0, z=0.0, a=0.0, b=0.0, c=0.0,
                          e=0.0, tool_id=0, move_type="TRAVEL", src_line="0",
                          event_flag=0, event_type="", payload="", trigger_seq=None)
        start_p = Position(base.x, base.y, base.z, base.a, base.b, base.c)
        end_p = Position(base.x, base.y, base.z + resin_z_print_compensation_mm,
                         base.a, base.b, base.c)
        occ = _ensure_segment(layer, "TRAVEL")
        gc = GlobalCurveCommand(
            type="TRAVEL",
            cmd="SPLINE",
            start_pos=start_p,
            control_points=[end_p, end_p, end_p],
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=line,
            raw="resin_z_print_compensation",
            constraints=[],
            original_moves=[],
        )
        _append_sample(gc, layer, "TRAVEL", occ)
        resin_z_offset = resin_z_print_compensation_mm

    def _append_extrude_wait(
            cmd: ExtrudeWait, layer: int, subtype: str, occ: int, mark_path_end: bool = True):
        nonlocal seq, processed_rows, last_pose, last_feedrate_mm_min
        hold_row = last_pose or CsvRow(
            seq=seq,
            x=0.0,
            y=0.0,
            z=0.0,
            a=0.0,
            b=0.0,
            c=0.0,
            e=0.0,
            tool_id=current_tool,
            move_type="PRINT",
            src_line=str(cmd.line),
            event_flag=0,
            event_type="",
            payload="",
            trigger_seq=None,
        )
        start_e = 0.0 if (cmd.raw or "") == "external_npz_reset_anchor" else hold_row.e
        steps = max(1, int(math.ceil(max(float(cmd.wait_sec), dt) / dt)))
        writer = _writer_for(layer, subtype, occ)
        timing.start_segment(
            path_id=_path_id_for(layer, subtype, occ),
            move_type="PRINT",
            start_seq=seq,
        )
        for i in range(1, steps + 1):
            ratio = i / steps
            planned_time_s = timing.append_trajectory_time()
            row = CsvRow(
                seq=seq,
                x=hold_row.x,
                y=hold_row.y,
                z=hold_row.z,
                a=hold_row.a,
                b=hold_row.b,
                c=hold_row.c,
                e=start_e + cmd.delta_e * ratio,
                tool_id=current_tool,
                move_type="PRINT",
                src_line=str(cmd.line),
                event_flag=0,
                event_type="",
                payload="",
                trigger_seq=None,
                path_id=_path_id_for(layer, subtype, occ),
                planned_time_s=planned_time_s,
            )
            row = _with_layer_progress(row, layer)
            writer.add(row)
            processed_rows += 1
            _maybe_yield()
            seq += 1
            last_pose = row
        timing.finish_segment(
            t_acc_s=0.0,
            t_flat_s=steps * dt,
            t_dec_s=0.0,
            end_seq=seq - 1,
        )
        if mark_path_end:
            _mark_path_end(layer, subtype, occ)
        if cmd.feedrate > 0:
            last_feedrate_mm_min = cmd.feedrate

    def _feed_mm_s_from_feedrate(feedrate_mm_min: Optional[float]) -> float:
        if feedrate_mm_min is not None and feedrate_mm_min > 0:
            return feedrate_mm_min / 60.0
        if last_feedrate_mm_min is not None and last_feedrate_mm_min > 0:
            return last_feedrate_mm_min / 60.0
        return float(default_feed_mm_s)

    def _next_retract_feedrate(start_idx: int) -> float:
        for next_cmd in parsed_commands[start_idx + 1:]:
            if isinstance(next_cmd, ExtrudeWait) and next_cmd.delta_e < 0.0 and next_cmd.feedrate > 0:
                return next_cmd.feedrate
            if isinstance(next_cmd, MoveCommand) and not next_cmd.is_pure_state_change:
                break
            if isinstance(next_cmd, (ToolChangeCommand, MCommand, ResetECommand)):
                break
        return max(float(default_feed_mm_s), 1e-9) * 60.0

    def _append_cut_sequence(cmd: MCommand, layer: int, subtype: str, occ: int, command_index: int):
        nonlocal last_pose
        ev = _mcommand_to_event(cmd, current_tool)
        if ev is None:
            return

        is_external_npz_fiber_cut = (
            current_tool == 1
            and external_npz_cut_absolute_e
            and (cmd.raw or "") == "external_npz_cut"
        )
        if is_external_npz_fiber_cut:
            # Keep converter-generated fiber CUT extrusion phases independent
            # from path E and from each other:
            # reset -> CUT -> lift 0..L -> settle -> reset ->
            # retract 0..-L -> settle -> reset -> remaining high hold.
            # CUT stays nonblocking; putting the first reset before it prevents
            # RSI from advancing to an E=0 anchor before the reset handshake.
            settle_s = 3.0
            lift_mm = max(0.0, float(cut_lift_mm))
            wait_s = max(0.0, float(cut_wait_s))
            lift_feedrate = max(float(default_feed_mm_s), 1e-9) * 60.0
            lift_speed = max(_feed_mm_s_from_feedrate(lift_feedrate), 1e-9)
            retract_feedrate = lift_feedrate
            retract_speed = lift_speed

            def _append_cut_reset_anchor(*, mark_path_end: bool = False):
                _emit_event(_PendingEvent(
                    event_type="extrude_reset",
                    payload=str(current_tool),
                    src_line=cmd.line or 0,
                    tool_id=current_tool,
                ), layer, subtype, occ)
                _append_extrude_wait(ExtrudeWait(
                    type="EXTRUDE_WAIT",
                    wait_sec=dt,
                    delta_e=0.0,
                    feedrate=lift_feedrate,
                    line=cmd.line or 0,
                    layer=layer,
                    subtype=subtype,
                    raw="external_npz_reset_anchor",
                ), layer, subtype, occ, mark_path_end=mark_path_end)

            _append_cut_reset_anchor()
            _emit_event(ev, layer, subtype, occ)

            if lift_mm > 1e-9:
                hold_row = last_pose or CsvRow(
                    seq=seq,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    a=0.0,
                    b=0.0,
                    c=0.0,
                    e=0.0,
                    tool_id=current_tool,
                    move_type="TRAVEL",
                    src_line=str(cmd.line),
                    event_flag=0,
                    event_type="",
                    payload="",
                    trigger_seq=None,
                    path_id=_path_id_for(layer, subtype, occ),
                )
                start_p = Position(
                    hold_row.x, hold_row.y, hold_row.z,
                    hold_row.a, hold_row.b, hold_row.c,
                )
                end_p = Position(
                    hold_row.x, hold_row.y, hold_row.z + lift_mm,
                    hold_row.a, hold_row.b, hold_row.c,
                )
                lift_gc = GlobalCurveCommand(
                    type="TRAVEL",
                    cmd="SPLINE",
                    start_pos=start_p,
                    control_points=[end_p, end_p, end_p],
                    e_val=lift_mm,
                    delta_e=lift_mm,
                    feedrate=lift_feedrate,
                    line=cmd.line or 0,
                    raw="cut_lift_feed",
                    constraints=[],
                    original_moves=[],
                )
                _append_sample(
                    lift_gc, layer, subtype, occ, mark_path_end=False
                )

            _append_extrude_wait(ExtrudeWait(
                type="EXTRUDE_WAIT",
                wait_sec=settle_s,
                delta_e=0.0,
                feedrate=lift_feedrate,
                line=cmd.line or 0,
                layer=layer,
                subtype=subtype,
                raw="external_npz_cut_lift_settle",
            ), layer, subtype, occ, mark_path_end=False)

            # The reset at +L makes the equal retract a separate absolute
            # interval 0..-L instead of allowing it to inherit lift E.
            _append_cut_reset_anchor()
            if lift_mm > 1e-9:
                _append_extrude_wait(ExtrudeWait(
                    type="EXTRUDE_WAIT",
                    wait_sec=lift_mm / retract_speed,
                    delta_e=-lift_mm,
                    feedrate=retract_feedrate,
                    line=cmd.line or 0,
                    layer=layer,
                    subtype=subtype,
                    raw="cut_safety_retract",
                ), layer, subtype, occ, mark_path_end=False)

            _append_extrude_wait(ExtrudeWait(
                type="EXTRUDE_WAIT",
                wait_sec=settle_s,
                delta_e=0.0,
                feedrate=retract_feedrate,
                line=cmd.line or 0,
                layer=layer,
                subtype=subtype,
                raw="external_npz_cut_retract_settle",
            ), layer, subtype, occ, mark_path_end=False)

            lift_duration_s = lift_mm / lift_speed
            retract_duration_s = lift_mm / retract_speed
            remaining_wait_s = max(
                0.0,
                wait_s
                - lift_duration_s
                - settle_s
                - retract_duration_s
                - settle_s,
            )
            _append_cut_reset_anchor(mark_path_end=remaining_wait_s <= 1e-9)
            if remaining_wait_s > 1e-9:
                _append_extrude_wait(ExtrudeWait(
                    type="EXTRUDE_WAIT",
                    wait_sec=remaining_wait_s,
                    delta_e=0.0,
                    feedrate=lift_feedrate,
                    line=cmd.line or 0,
                    layer=layer,
                    subtype=subtype,
                    raw="cut_wait_remaining",
                ), layer, subtype, occ, mark_path_end=True)
            return

        # Generic GCode CUT behavior remains unchanged.
        _emit_event(ev, layer, subtype, occ)

        lift_mm = max(0.0, float(cut_lift_mm))
        wait_s = max(0.0, float(cut_wait_s))
        fiber_retract_mm = (
            max(0.0, float(fiber_retract_length_mm))
            if current_tool == 1 and fiber_retract_length_mm is not None else 0.0
        )
        total_retract_mm = lift_mm + fiber_retract_mm
        if lift_mm <= 1e-9:
            if wait_s > 1e-9:
                _append_extrude_wait(ExtrudeWait(
                    type="EXTRUDE_WAIT",
                    wait_sec=wait_s,
                    delta_e=0.0,
                    feedrate=max(float(default_feed_mm_s), 1e-9) * 60.0,
                    line=cmd.line or 0,
                    layer=layer,
                    subtype=subtype,
                    raw="cut_wait",
                ), layer, subtype, occ, mark_path_end=False)
            if total_retract_mm > 1e-9:
                retract_feedrate = _next_retract_feedrate(command_index)
                retract_speed = max(_feed_mm_s_from_feedrate(retract_feedrate), 1e-9)
                _append_extrude_wait(ExtrudeWait(
                    type="EXTRUDE_WAIT",
                    wait_sec=total_retract_mm / retract_speed,
                    delta_e=-total_retract_mm,
                    feedrate=retract_feedrate,
                    line=cmd.line or 0,
                    layer=layer,
                    subtype=subtype,
                    raw="cut_safety_retract",
                ), layer, subtype, occ, mark_path_end=True)
            return

        hold_row = last_pose or CsvRow(
            seq=seq,
            x=0.0,
            y=0.0,
            z=0.0,
            a=0.0,
            b=0.0,
            c=0.0,
            e=0.0,
            tool_id=current_tool,
            move_type="TRAVEL",
            src_line=str(cmd.line),
            event_flag=0,
            event_type="",
            payload="",
            trigger_seq=None,
            path_id=_path_id_for(layer, subtype, occ),
        )
        start_p = Position(hold_row.x, hold_row.y, hold_row.z, hold_row.a, hold_row.b, hold_row.c)
        end_p = Position(hold_row.x, hold_row.y, hold_row.z + lift_mm, hold_row.a, hold_row.b, hold_row.c)
        lift_feedrate = max(float(default_feed_mm_s), 1e-9) * 60.0
        lift_gc = GlobalCurveCommand(
            type="TRAVEL",
            cmd="SPLINE",
            start_pos=start_p,
            control_points=[end_p, end_p, end_p],
            e_val=hold_row.e + lift_mm,
            delta_e=lift_mm,
            feedrate=lift_feedrate,
            line=cmd.line or 0,
            raw="cut_lift_feed",
            constraints=[],
            original_moves=[],
        )
        _append_sample(lift_gc, layer, subtype, occ, mark_path_end=False)

        lift_duration_s = lift_mm / max(_feed_mm_s_from_feedrate(lift_feedrate), 1e-9)
        remaining_wait_s = max(0.0, wait_s - lift_duration_s)
        if remaining_wait_s > 1e-9:
            _append_extrude_wait(ExtrudeWait(
                type="EXTRUDE_WAIT",
                wait_sec=remaining_wait_s,
                delta_e=0.0,
                feedrate=lift_feedrate,
                line=cmd.line or 0,
                layer=layer,
                subtype=subtype,
                raw="cut_wait_remaining",
            ), layer, subtype, occ, mark_path_end=False)

        retract_feedrate = _next_retract_feedrate(command_index)
        retract_speed = max(_feed_mm_s_from_feedrate(retract_feedrate), 1e-9)
        _append_extrude_wait(ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=total_retract_mm / retract_speed,
            delta_e=-total_retract_mm,
            feedrate=retract_feedrate,
            line=cmd.line or 0,
            layer=layer,
            subtype=subtype,
            raw="cut_safety_retract",
        ), layer, subtype, occ, mark_path_end=True)

    def _emit_event(ev: _PendingEvent, layer: int, subtype: str, occ: int):
        nonlocal seq, processed_rows, last_pose_map, last_pose
        hold_row = last_pose or CsvRow(
            seq=seq,
            x=0.0,
            y=0.0,
            z=0.0,
            a=0.0,
            b=0.0,
            c=0.0,
            e=0.0,
            tool_id=ev.tool_id,
            move_type="TRAVEL",
            src_line=str(ev.src_line),
            event_flag=0,
            event_type="",
            payload="",
            trigger_seq=None,
            path_id=_path_id_for(layer, subtype, occ),
        )
        planned_time_s = timing.append_event_time()
        row = CsvRow(
            seq=seq,
            x=hold_row.x,
            y=hold_row.y,
            z=hold_row.z,
            a=hold_row.a,
            b=hold_row.b,
            c=hold_row.c,
            e=hold_row.e,
            tool_id=ev.tool_id,
            move_type="EVENT",
            src_line=str(ev.src_line),
            event_flag=1,
            event_type=ev.event_type,
            payload=ev.payload,
            trigger_seq=seq,
            path_id=_path_id_for(layer, subtype, occ),
            planned_time_s=planned_time_s,
        )
        _writer_for(layer, subtype, occ).add(_with_layer_progress(row, layer))
        processed_rows += 1
        _maybe_yield()
        seq += 1
        last_pose_map[(layer, subtype)] = hold_row
        last_pose = hold_row

    if enable_extrude_wait and enable_travel_extrude_overlap:
        parsed_commands = _overlap_extrude_waits_on_travel(parsed_commands)

    total_cmds = len(parsed_commands)

    def _is_external_npz_start_xy_travel(cmd) -> bool:
        return (
            isinstance(cmd, MoveCommand)
            and cmd.type == "TRAVEL"
            and not cmd.is_pure_state_change
            and (cmd.raw or "") == "external_npz_start_xy_travel"
        )

    delay_resin_z_compensation = any(
        _is_external_npz_start_xy_travel(cmd)
        for cmd in parsed_commands
    )

    if not delay_resin_z_compensation:
        _append_resin_z_print_compensation(0, 0)

    for idx, cmd in enumerate(parsed_commands):
        if progress_callback and total_cmds > 0 and idx % 200 == 0:
            progress_callback(idx / total_cmds)

        # ---- 偏置补偿：对 Tool 1 的所有轨迹应用坐标偏置 ----
        ox = oy = oz = 0.0
        if current_tool == 1 and (
                abs(tool_offset[0]) > 1e-9
                or abs(tool_offset[1]) > 1e-9
                or abs(tool_offset[2]) > 1e-9
        ):
            ox, oy, oz = tool_offset
        if abs(resin_z_offset) > 1e-9:
            oz += resin_z_offset
        if abs(ox) > 1e-9 or abs(oy) > 1e-9 or abs(oz) > 1e-9:
            from .types import Position as _Pos
            if isinstance(cmd, MoveCommand):
                cmd.start_pos = _Pos(
                    cmd.start_pos.x + ox,
                    cmd.start_pos.y + oy,
                    cmd.start_pos.z + oz,
                    cmd.start_pos.a,
                    cmd.start_pos.b,
                    cmd.start_pos.c)
                cmd.pos = _Pos(cmd.pos.x + ox, cmd.pos.y + oy, cmd.pos.z + oz,
                               cmd.pos.a, cmd.pos.b, cmd.pos.c)
            elif isinstance(cmd, GlobalCurveCommand):
                cmd.start_pos = _Pos(
                    cmd.start_pos.x + ox,
                    cmd.start_pos.y + oy,
                    cmd.start_pos.z + oz,
                    cmd.start_pos.a,
                    cmd.start_pos.b,
                    cmd.start_pos.c)
                cmd.control_points = [
                    _Pos(cp.x + ox, cp.y + oy, cp.z + oz, cp.a, cp.b, cp.c)
                    for cp in cmd.control_points
                ]
        # ---- 偏置补偿结束 ----

        cmd_layer = getattr(cmd, "layer", None)
        if split_by_layer_type and isinstance(cmd_layer, int):
            if current_layer is not None and cmd_layer > current_layer:
                flush_moves()
            _finalize_layers_before(cmd_layer)

        # 事件收集：遇到事件前先冲掉当前轨迹段，保证事件贴在后续采样点
        if isinstance(cmd, (ToolChangeCommand, MCommand, ResetECommand)):
            flush_moves()
            if isinstance(cmd, ToolChangeCommand):
                mapped_tool = _map_gcode_tool(cmd.tool)
                occ = occ_counters.get((cmd.layer, cmd.subtype), 0)
                if occ == 0:
                    occ = _ensure_segment(cmd.layer, cmd.subtype)

                if mapped_tool != current_tool:
                    # ---- 偏置补偿：在 tool_change 事件前注入安全抬升和 TRAVEL 段 ----
                    ox, oy, oz = tool_offset
                    has_offset = abs(ox) > 1e-9 or abs(oy) > 1e-9 or abs(oz) > 1e-9
                    if has_offset:
                        last_row = last_pose
                        if last_row is not None:
                            from .types import Position as _Pos
                            start_p = _Pos(last_row.x, last_row.y, last_row.z,
                                           last_row.a, last_row.b, last_row.c)
                            safe_lift = max(0.0, float(tool_change_safe_lift_mm))
                            offset_start_p = start_p
                            if safe_lift > 1e-9:
                                lifted_p = _Pos(last_row.x, last_row.y, last_row.z + safe_lift,
                                                last_row.a, last_row.b, last_row.c)
                                lift_gc = GlobalCurveCommand(
                                    type="TRAVEL",
                                    cmd="SPLINE",
                                    start_pos=start_p,
                                    control_points=[lifted_p, lifted_p, lifted_p],
                                    e_val=last_row.e,
                                    delta_e=0.0,
                                    feedrate=default_feed_mm_s * 60.0,
                                    line=cmd.line,
                                    raw="tool_change_safe_lift",
                                    constraints=[],
                                    original_moves=[],
                                )
                                _append_sample(lift_gc, cmd.layer, cmd.subtype, occ)
                                offset_start_p = lifted_p
                            if mapped_tool == 1:
                                end_p = _Pos(offset_start_p.x + ox, offset_start_p.y + oy,
                                             offset_start_p.z + oz,
                                             offset_start_p.a, offset_start_p.b, offset_start_p.c)
                            else:
                                end_p = _Pos(offset_start_p.x - ox, offset_start_p.y - oy,
                                             offset_start_p.z - oz,
                                             offset_start_p.a, offset_start_p.b, offset_start_p.c)
                            offset_gc = GlobalCurveCommand(
                                type="TRAVEL",
                                cmd="SPLINE",
                                start_pos=offset_start_p,
                                control_points=[end_p, end_p, end_p],
                                e_val=last_row.e,
                                delta_e=0.0,
                                feedrate=default_feed_mm_s * 60.0,
                                line=cmd.line,
                                raw="fallback_linear",
                                constraints=[],
                                original_moves=[],
                            )
                            _append_sample(offset_gc, cmd.layer, cmd.subtype, occ)
                    # ---- 偏置补偿结束 ----

                    current_tool = mapped_tool
                    _emit_event(_PendingEvent(
                        event_type="tool_change_cf" if mapped_tool == 1 else "tool_change_resin",
                        payload=str(mapped_tool),
                        src_line=cmd.line,
                        tool_id=mapped_tool,
                    ), cmd.layer, cmd.subtype, occ)
            elif isinstance(cmd, ResetECommand):
                if last_pose is None and cmd.pose is not None:
                    last_pose = CsvRow(
                        seq=seq,
                        x=cmd.pose.x,
                        y=cmd.pose.y,
                        z=cmd.pose.z,
                        a=cmd.pose.a,
                        b=cmd.pose.b,
                        c=cmd.pose.c,
                        e=cmd.val,
                        tool_id=current_tool,
                        move_type="TRAVEL",
                        src_line=str(cmd.line),
                        event_flag=0,
                        event_type="",
                        payload="",
                        trigger_seq=None,
                    )
                occ = occ_counters.get((cmd.layer, cmd.subtype), 0)
                if occ == 0:
                    occ = _ensure_segment(cmd.layer, cmd.subtype)
                _emit_event(_PendingEvent(
                    event_type="extrude_reset",
                    payload=str(current_tool),
                    src_line=cmd.line,
                    tool_id=current_tool,
                ), cmd.layer, cmd.subtype, occ)
            else:
                occ = occ_counters.get((cmd.layer, cmd.subtype), 0)
                if occ == 0:
                    occ = _ensure_segment(cmd.layer, cmd.subtype)
                if isinstance(cmd, MCommand) and cmd.code.upper() == "CUT":
                    _append_cut_sequence(cmd, cmd.layer, cmd.subtype, occ, idx)
                else:
                    ev = _mcommand_to_event(cmd, current_tool)
                    if ev:
                        _emit_event(ev, cmd.layer, cmd.subtype, occ)
            continue

        if isinstance(cmd, ExtrudeWait):
            if not enable_extrude_wait:
                continue
            flush_moves()
            layer = getattr(cmd, "layer", 0)
            subtype = getattr(cmd, "subtype", "UNKNOWN") or "UNKNOWN"
            occ = occ_counters.get((layer, subtype), 0)
            if occ == 0:
                occ = _ensure_segment(layer, subtype)
            _append_extrude_wait(cmd, layer, subtype, occ)
            continue

        # 轨迹分段收集
        if isinstance(cmd, MoveCommand):
            if cmd.is_pure_state_change:
                continue
            if current_type is None and last_pose is not None:
                from .types import Position as _Pos
                if (
                    abs(cmd.start_pos.x - last_pose.x) > 1e-9
                    or abs(cmd.start_pos.y - last_pose.y) > 1e-9
                    or abs(cmd.start_pos.z - last_pose.z) > 1e-9
                    or abs(cmd.start_pos.a - last_pose.a) > 1e-9
                    or abs(cmd.start_pos.b - last_pose.b) > 1e-9
                    or abs(cmd.start_pos.c - last_pose.c) > 1e-9
                ):
                    cmd.start_pos = _Pos(last_pose.x, last_pose.y, last_pose.z,
                                         last_pose.a, last_pose.b, last_pose.c)
            if current_type is None:
                current_type = cmd.type
                current_layer = cmd.layer
                current_subtype = cmd.subtype
                current_occ = _ensure_segment(cmd.layer, cmd.subtype)
            if (cmd.type != current_type) or (cmd.layer !=
                                              current_layer) or (cmd.subtype != current_subtype):
                flush_moves()
                current_type = cmd.type
                current_layer = cmd.layer
                current_subtype = cmd.subtype
                current_occ = _ensure_segment(cmd.layer, cmd.subtype)
            buffer.append(cmd)
            if (
                delay_resin_z_compensation
                and not resin_z_compensation_appended
                and _is_external_npz_start_xy_travel(cmd)
            ):
                flush_moves()
                _append_resin_z_print_compensation(cmd.layer, cmd.line)
            continue

        if isinstance(cmd, GlobalCurveCommand):
            # 如果上游已提供曲线，先冲掉当前 Move 段，再直接采样
            flush_moves()
            layer = getattr(cmd, "layer", 0)
            subtype = getattr(cmd, "subtype", "UNKNOWN")
            occ = occ_counters.get((layer, subtype), 0)
            if occ == 0:
                occ = _ensure_segment(layer, subtype)
            _append_sample(cmd, layer, subtype, occ)
            continue

    # 文件末尾冲掉残余 Move 段
    flush_moves()

    if split_by_layer_type:
        remaining_layers = sorted(
            {key[0] for key in writers.keys() if isinstance(key, tuple) and len(key) == 3},
        )
        if remaining_layers:
            _finalize_layers_before(max(remaining_layers) + 1)
    else:
        for key in list(writers.keys()):
            _finalize_writer(key)

    if plot_layer_xy and not split_by_layer_type and flat_preview_points:
        t0_plot = time.perf_counter()
        _plot_flat_layer_previews(flat_preview_points, base_root)
        timings["plot_s"] += time.perf_counter() - t0_plot

    if split_by_layer_type and manifest:
        t0 = time.perf_counter()
        manifest_path = os.path.join(base_root, f"{base_name}_manifest.json")
        manifest.sort(
            key=lambda item: (
                int(item.get("start_seq", -1) if item.get("start_seq") is not None else -1),
                int(item.get("end_seq", -1) if item.get("end_seq") is not None else -1),
                int(item.get("layer", 0)),
                str(item.get("type", "")),
                int(item.get("occ", 0)),
            )
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        timings["manifest_s"] += time.perf_counter() - t0

    # Write tool_offset sidecar JSON file
    try:
        offset_file = base_no_ext + ".offset.json"
        with open(offset_file, "w", encoding="utf-8") as f:
            json.dump({"tool_offset": list(tool_offset), "resin_z_print_compensation_mm": float(
                resin_z_print_compensation_mm)}, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[Warning] Failed to write offset sidecar json: {exc}")

    try:
        timing_parent = os.path.dirname(timing_sidecar_path)
        if timing_parent:
            os.makedirs(timing_parent, exist_ok=True)
        with open(timing_sidecar_path, "w", encoding="utf-8") as f:
            json.dump(timing.summary(), f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[Warning] Failed to write timing sidecar json: {exc}")

    timings["timing_sidecar"] = timing_sidecar_path
    timings["planned_total_time_s"] = timing.trajectory_time()
    timings["rows"] = processed_rows
    timings["total_s"] = time.perf_counter() - t_total_start
    return timings


def _plot_flat_layer_previews(layer_points: dict, base_root: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return

    from pathlib import Path

    out_dir = Path(base_root) / "layer_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer, points in sorted(layer_points.items()):
        xs, ys = points
        if not xs:
            continue
        layer_num = int(layer)
        fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
        ax.plot(xs, ys, linewidth=0.8, color="#2b2b2b")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Layer {layer_num:04d} XY Path")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.savefig(str(out_dir / f"layer_{layer_num:04d}.png"), bbox_inches="tight")
        plt.close(fig)


def _plot_single_layer(entries, base_root: str, stride: int = 5) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return

    import numpy as np
    from pathlib import Path

    if not entries:
        return

    def _resolve_npz_files(base_path: Path):
        if base_path.suffix != ".npz":
            base_path = base_path.with_suffix(".npz")
        if base_path.exists():
            return [base_path]
        prefix = base_path.stem + "_part"
        return sorted([p for p in base_path.parent.glob("*.npz") if p.stem.startswith(prefix)])

    layer = int(entries[0].get("layer", 0))
    xs_all = []
    ys_all = []
    first_seg = True
    for seg in entries:
        base = Path(seg["base_path"]).expanduser().resolve()
        files = _resolve_npz_files(base)
        if not files:
            continue
        for f in files:
            z = np.load(str(f))
            if "x" not in z or "y" not in z or "e" not in z or "move_type" not in z:
                continue
            x = z["x"]
            y = z["y"]
            e = z["e"]
            mt = z["move_type"]
            is_print = (mt == 1) | (mt == 3)
            de = np.diff(e, prepend=e[0])
            is_deposit = is_print & (de > 1e-6)
            x = np.where(is_deposit, x, np.nan)
            y = np.where(is_deposit, y, np.nan)
            if stride > 1:
                x = x[::stride]
                y = y[::stride]
            if not first_seg:
                xs_all.append(np.array([np.nan], dtype=np.float32))
                ys_all.append(np.array([np.nan], dtype=np.float32))
            first_seg = False
            xs_all.append(x)
            ys_all.append(y)

    if not xs_all:
        return

    x = np.concatenate(xs_all)
    y = np.concatenate(ys_all)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    ax.plot(x, y, linewidth=0.8, color="#2b2b2b")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Layer {layer:04d} XY Path")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    out_dir = Path(base_root) / "layer_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer_{layer:04d}.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def _npz_exporter(output_path: str, rows: List[CsvRow], chunk_size: int) -> None:
    import numpy as np
    base, ext = os.path.splitext(output_path)
    single_path = output_path if ext.lower() == ".npz" else base + ".npz"
    part_base = base if ext.lower() == ".npz" else output_path

    move_type_map = {
        "TRAVEL": 0,
        "PRINT": 1,
        "TRAVEL_FIT": 2,
        "PRINT_FIT": 3,
        "EVENT": 4,
    }
    event_type_map = {
        "": 0,
        "heat_cf": 1,
        "heat_resin": 2,
        "fan_cf": 3,
        "fan_resin": 4,
        "extrude_reset": 5,
        "tool_change_cf": 6,
        "tool_change_resin": 7,
        "cut": 8,
    }
    # 使用定长字节串，便于 C++ 侧 cnpy 读取
    move_type_keys = np.array(list(move_type_map.keys()), dtype="S32")
    move_type_vals = np.array(list(move_type_map.values()), dtype=np.uint8)
    event_type_keys = np.array(list(event_type_map.keys()), dtype="S32")
    event_type_vals = np.array(list(event_type_map.values()), dtype=np.uint8)

    n = len(rows)
    start = 0
    part = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = rows[start:end]
        seq = np.array([r.seq for r in chunk], dtype=np.uint32)
        x = np.array([r.x for r in chunk], dtype=np.float32)
        y = np.array([r.y for r in chunk], dtype=np.float32)
        z = np.array([r.z for r in chunk], dtype=np.float32)
        a = np.array([r.a for r in chunk], dtype=np.float32)
        b = np.array([r.b for r in chunk], dtype=np.float32)
        c = np.array([r.c for r in chunk], dtype=np.float32)
        e = np.array([r.e for r in chunk], dtype=np.float32)
        tool_id = np.array([r.tool_id for r in chunk], dtype=np.uint8)
        move_type = np.array([move_type_map.get(r.move_type, 255) for r in chunk], dtype=np.uint8)
        src_line = np.array([r.src_line for r in chunk], dtype="S32")
        event_flag = np.array([r.event_flag for r in chunk], dtype=np.uint8)
        event_type = np.array([event_type_map.get(r.event_type, 255)
                              for r in chunk], dtype=np.uint8)
        payload = np.array([str(r.payload) for r in chunk], dtype="S32")
        trigger_seq = np.array(
            [r.trigger_seq if r.trigger_seq is not None else -1 for r in chunk],
            dtype=np.int32,
        )
        layer_index = np.array([r.layer_index for r in chunk], dtype=np.uint32)
        total_layers_arr = np.array([r.total_layers for r in chunk], dtype=np.uint32)
        preview_layer_index = np.array(
            [r.preview_layer_index for r in chunk],
            dtype=np.int32,
        )

        out_path = (
            f"{part_base}_part{part:04d}.npz"
            if n > chunk_size
            else single_path
        )
        np.savez_compressed(
            out_path,
            seq=seq,
            x=x,
            y=y,
            z=z,
            a=a,
            b=b,
            c=c,
            e=e,
            tool_id=tool_id,
            move_type=move_type,
            src_line=src_line,
            event_flag=event_flag,
            event_type=event_type,
            payload=payload,
            trigger_seq=trigger_seq,
            layer_index=layer_index,
            total_layers=total_layers_arr,
            preview_layer_index=preview_layer_index,
            move_type_vocab_keys=move_type_keys,
            move_type_vocab_vals=move_type_vals,
            event_type_vocab_keys=event_type_keys,
            event_type_vocab_vals=event_type_vals,
        )
        start = end
        part += 1


def _map_gcode_tool(gcode_tool: int) -> int:
    """将 GCode 中的 T0/T1 映射到内部工具号：1=纤维(T0)，2=树脂(T1)."""
    if gcode_tool == 0:
        return 1
    if gcode_tool == 1:
        return 2
    return gcode_tool


def _mcommand_to_event(cmd: MCommand, current_tool: int) -> Optional[_PendingEvent]:
    """M 指令映射到事件名/负载；未覆盖的返回 None."""
    code = cmd.code.upper()
    params = cmd.params or {}

    def _get_tool_from_params():
        if "T" in params:
            try:
                return _map_gcode_tool(int(params["T"]))
            except Exception:
                return current_tool
        return current_tool

    tool_id = _get_tool_from_params()
    src_line = cmd.line if cmd.line is not None else -1

    if code in ("M104", "M109"):  # 加热
        if "S" not in params:
            return None
        temp_val = params["S"]
        ev_type = "heat_cf" if tool_id == 1 else "heat_resin"
        return _PendingEvent(ev_type, str(temp_val), src_line, tool_id)

    if code == "M106":  # 风扇
        ev_type = "fan_cf" if tool_id == 1 else "fan_resin"
        return _PendingEvent(ev_type, "1", src_line, tool_id)

    if code == "M107":  # 关风扇
        ev_type = "fan_cf" if tool_id == 1 else "fan_resin"
        return _PendingEvent(ev_type, "0", src_line, tool_id)

    if code == "CUT":
        payload = str(int(params.get("P", 1)))
        return _PendingEvent("cut", payload, src_line, tool_id)

    # 其他 M 指令：忽略（热床等）
    return None
