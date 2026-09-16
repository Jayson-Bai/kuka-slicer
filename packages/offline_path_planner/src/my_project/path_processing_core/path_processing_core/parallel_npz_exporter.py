"""Deterministic layer-parallel wrapper around the serial Core exporter.

Each logical layer is an independent command interval for the external NPZ
contract.  Workers run the unchanged fitter/sampler, while the parent restores
the global seq, path-id, RSI clock, injection catalog, and sidecars in original
layer order.  No trajectory math is recomputed during the merge.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any

import numpy as np

from .npz_exporter import _map_gcode_tool, export_npz
from .types import MCommand, ResetECommand, ToolChangeCommand


_ROW_FIELDS = (
    "x", "y", "z", "a", "b", "c", "e", "tool_id", "move_type",
    "src_line", "event_flag", "event_type", "payload", "layer_index",
    "preview_layer_index", "path_end_flag", "core_injection_role",
)
_VOCAB_FIELDS = (
    "move_type_vocab_keys", "move_type_vocab_vals",
    "event_type_vocab_keys", "event_type_vocab_vals",
    "core_injection_role_vocab_keys", "core_injection_role_vocab_vals",
)
_TIMING_SUM_KEYS = (
    "fit_s", "fit_gen_points_s", "fit_density_s", "fit_prepare_data_s",
    "fit_param_s", "fit_knot_s", "fit_lsq_s", "fit_post_ctrl_s",
    "fit_lsq_basis_build_s", "fit_lsq_qk_build_s",
    "fit_lsq_normal_mat_s", "fit_lsq_solve_s", "fit_lsq_total_s",
    "sample_s", "sample_arc_map_s", "sample_lookup_s", "sample_deboor_s",
    "sample_pose_s", "sample_extrude_s", "write_s", "manifest_s", "plot_s",
)

_WORKER_NUMERIC_THREAD_ENVS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_PERSISTENT_POOL_LOCK = threading.Lock()
_PERSISTENT_POOL: ProcessPoolExecutor | None = None
_PERSISTENT_POOL_WORKERS = 0
_PERSISTENT_POOL_READY_PIDS: set[int] = set()


@contextmanager
def _single_thread_worker_numeric_environment():
    """Let each layer process own one numerical-library thread.

    Layer export is process-parallel.  Inheriting the UI's workstation-wide
    numeric thread limit in every child turns ``N`` workers into roughly
    ``N²`` competing BLAS/OpenMP threads.  The variables are set only while
    children are created: on Windows ``spawn`` reads them before NumPy/SciPy
    import, and the parent environment is restored before the executor's
    results are consumed.
    """

    previous = {name: os.environ.get(name) for name in _WORKER_NUMERIC_THREAD_ENVS}
    try:
        for name in _WORKER_NUMERIC_THREAD_ENVS:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _export_layer_worker(payload: tuple) -> dict[str, Any]:
    (
        command_start,
        target_layer,
        target_command_count,
        commands,
        output_path,
        export_kwargs,
        dispatch_order,
        submitted_at,
    ) = payload
    worker_started_at = time.perf_counter()
    # The parent establishes one BLAS/OpenMP thread before this spawned
    # process imports the numerical stack.  Export itself remains the
    # unchanged serial algorithm so the merged archive stays deterministic.
    stats = export_npz(commands, output_path, **export_kwargs)
    worker_finished_at = time.perf_counter()
    return {
        "command_start": int(command_start),
        "target_layer": int(target_layer),
        "command_count": len(commands),
        "target_command_count": int(target_command_count),
        "output_path": output_path,
        "stats": stats,
        "worker_pid": os.getpid(),
        "worker_started_at": worker_started_at,
        "worker_finished_at": worker_finished_at,
        "queue_delay_s": worker_started_at - float(submitted_at),
        "dispatch_order": int(dispatch_order),
    }


def _worker_ready() -> int:
    """Force a spawned worker to finish importing this module before a job."""

    # A short delay keeps all submitted warm-up jobs pending long enough for
    # ``ProcessPoolExecutor`` to spawn every requested worker, not just one.
    time.sleep(0.05)
    return os.getpid()


def _persistent_worker_pool(max_workers: int) -> tuple[ProcessPoolExecutor, bool]:
    """Return the UI-owned Core pool and whether it was just created."""

    global _PERSISTENT_POOL, _PERSISTENT_POOL_WORKERS, _PERSISTENT_POOL_READY_PIDS
    requested = max(1, int(max_workers))
    with _PERSISTENT_POOL_LOCK:
        if _PERSISTENT_POOL is None or _PERSISTENT_POOL_WORKERS != requested:
            if _PERSISTENT_POOL is not None:
                _PERSISTENT_POOL.shutdown(wait=True, cancel_futures=True)
            _PERSISTENT_POOL = ProcessPoolExecutor(max_workers=requested)
            _PERSISTENT_POOL_WORKERS = requested
            _PERSISTENT_POOL_READY_PIDS = set()
            return _PERSISTENT_POOL, True
        return _PERSISTENT_POOL, False


def warm_parallel_worker_pool(max_workers: int, progress_callback=None) -> dict[str, int]:
    """Pre-spawn the UI Core pool without changing any export output."""

    global _PERSISTENT_POOL_READY_PIDS
    requested = max(1, int(max_workers))
    pool, created = _persistent_worker_pool(max_workers)
    if not created:
        ready = min(requested, len(_PERSISTENT_POOL_READY_PIDS) or requested)
        if progress_callback is not None:
            progress_callback(ready, requested)
        return {"ready_workers": ready, "total_workers": requested}
    with _single_thread_worker_numeric_environment():
        ready_pids: set[int] = set()
        for _attempt in range(4):
            futures = [pool.submit(_worker_ready) for _ in range(requested)]
            for future in as_completed(futures):
                ready_pids.add(int(future.result()))
                if progress_callback is not None:
                    progress_callback(min(len(ready_pids), requested), requested)
            if len(ready_pids) >= requested:
                break
        if len(ready_pids) < requested:
            raise RuntimeError(
                f"Core worker warm-up reached {len(ready_pids)}/{requested} processes"
            )
    with _PERSISTENT_POOL_LOCK:
        _PERSISTENT_POOL_READY_PIDS = ready_pids
    return {"ready_workers": len(ready_pids), "total_workers": requested}


def shutdown_parallel_worker_pool() -> None:
    """Release UI-owned Core workers during server shutdown."""

    global _PERSISTENT_POOL, _PERSISTENT_POOL_WORKERS, _PERSISTENT_POOL_READY_PIDS
    with _PERSISTENT_POOL_LOCK:
        if _PERSISTENT_POOL is not None:
            _PERSISTENT_POOL.shutdown(wait=True, cancel_futures=True)
        _PERSISTENT_POOL = None
        _PERSISTENT_POOL_WORKERS = 0
        _PERSISTENT_POOL_READY_PIDS = set()


def _command_layers(parsed_commands) -> list[tuple[int, int, int, list]]:
    chunks: list[tuple[int, int, int, list]] = []
    start = 0
    current_layer = None
    current: list = []
    for index, command in enumerate(parsed_commands):
        layer = int(getattr(command, "layer", 0) or 0)
        if current_layer is None:
            current_layer = layer
            start = index
        elif layer != current_layer:
            if layer < current_layer:
                raise ValueError("parallel Core export requires non-decreasing command layers")
            chunks.append((current_layer, start, index, current))
            current_layer = layer
            start = index
            current = []
        current.append(command)
    if current_layer is not None:
        chunks.append((current_layer, start, len(parsed_commands), current))
    return chunks


def _initial_tools(chunks, initial_tool_id: int) -> list[int]:
    current_tool = int(initial_tool_id)
    values: list[int] = []
    for _layer, _start, _end, commands in chunks:
        values.append(current_tool)
        for command in commands:
            if isinstance(command, ToolChangeCommand):
                current_tool = _map_gcode_tool(command.tool)
    return values


def _boundary_context(commands: list) -> tuple[int, list]:
    """Return the smallest suffix that reconstructs a layer's final state."""

    last_cut = next(
        (
            index
            for index in range(len(commands) - 1, -1, -1)
            if isinstance(commands[index], MCommand)
            and commands[index].code.upper() == "CUT"
        ),
        None,
    )
    anchor_limit = len(commands) if last_cut is None else last_cut
    anchor = next(
        (
            index
            for index in range(anchor_limit - 1, -1, -1)
            if isinstance(commands[index], ResetECommand)
        ),
        0,
    )
    return anchor, commands[anchor:]


def _tool_before_indices(parsed_commands, initial_tool_id: int) -> dict[int, int]:
    current_tool = int(initial_tool_id)
    values: dict[int, int] = {}
    for index, command in enumerate(parsed_commands):
        values[index] = current_tool
        if isinstance(command, ToolChangeCommand):
            current_tool = _map_gcode_tool(command.tool)
    return values


def _read_manifest(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["core_injection_manifest"].item()))


def _merge_manifests(results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[int, int]]]:
    manifests = [_read_manifest(Path(result["output_path"])) for result in results]
    merged = deepcopy(manifests[0])
    base_block = deepcopy(manifests[0]["blocks"][0])
    base_block["id"] = 1
    blocks_by_command: dict[int, dict[str, Any]] = {}
    for result, manifest in zip(results, manifests):
        for raw_block in manifest.get("blocks", [])[1:]:
            if raw_block.get("command_index") is None:
                continue
            command_index = int(raw_block["command_index"]) + int(result["command_start"])
            block = deepcopy(raw_block)
            block["command_index"] = command_index
            blocks_by_command.setdefault(command_index, block)
    global_id_by_command = {
        command_index: index + 2
        for index, command_index in enumerate(sorted(blocks_by_command))
    }
    blocks = [base_block]
    for command_index in sorted(blocks_by_command):
        block = blocks_by_command[command_index]
        block["id"] = global_id_by_command[command_index]
        blocks.append(block)

    mappings: list[dict[int, int]] = []
    for result, manifest in zip(results, manifests):
        mapping = {1: 1}
        for raw_block in manifest.get("blocks", [])[1:]:
            if raw_block.get("command_index") is not None:
                command_index = int(raw_block["command_index"]) + int(result["command_start"])
                mapping[int(raw_block["id"])] = global_id_by_command[command_index]
        mappings.append(mapping)
    merged["blocks"] = blocks
    return merged, mappings


def _load_row_arrays(
    results: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Load every row field once per worker archive, in layer order.

    A worker archive is a compressed ``.npz``. Loading it separately for
    each field turns a 13-layer job into hundreds of archive opens and member
    decompressions. Copies are retained exactly as before, but the archive is
    opened only once per layer.
    """

    field_parts: dict[str, list[np.ndarray]] = {field: [] for field in _ROW_FIELDS}
    row_counts: list[int] = []
    for result in results:
        with np.load(result["output_path"], allow_pickle=False) as data:
            selected = data["preview_layer_index"] == result["target_layer"]
            row_counts.append(int(np.count_nonzero(selected)))
            for field in _ROW_FIELDS:
                field_parts[field].append(np.array(data[field][selected], copy=True))
    arrays = {
        field: np.concatenate(parts) if len(parts) > 1 else parts[0]
        for field, parts in field_parts.items()
    }
    return arrays, row_counts


def _write_offset_sidecar(
    output_path: Path,
    manifest: dict[str, Any],
    tool_offset: tuple[float, float, float],
    resin_z_print_compensation_mm: float,
) -> None:
    sidecar = {
        "format": manifest["format"],
        "schema_version": manifest["schema_version"],
        "tool_offset": list(tool_offset),
        "resin_z_print_compensation_mm": float(resin_z_print_compensation_mm),
    }
    if manifest.get("injection_state") == "base":
        for key in (
            "injection_state", "offset_kind", "offset_frame", "abc_convention",
            "abc_semantics", "offset_application", "calibration_id", "injected_at",
        ):
            sidecar[key] = manifest.get(key)
    output_path.with_suffix(".offset.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def export_npz_parallel_by_layer(
    parsed_commands,
    output_path: str,
    *,
    max_workers: int,
    progress_callback=None,
    reuse_workers: bool = False,
    **export_kwargs,
) -> dict[str, Any]:
    """Export independent logical layers concurrently and merge deterministically."""

    started = time.perf_counter()
    chunks = _command_layers(parsed_commands)
    if len(chunks) < 2 or max_workers < 2:
        return export_npz(
            parsed_commands,
            output_path,
            progress_callback=progress_callback,
            **export_kwargs,
        )
    if export_kwargs.get("split_by_layer_type") or export_kwargs.get("plot_layer_xy"):
        raise ValueError("parallel layer export does not support split/plot outputs")
    tool_offset = tuple(float(value) for value in export_kwargs.get("tool_offset", (0, 0, 0)))
    resin_z = float(export_kwargs.get("resin_z_print_compensation_mm", 0.0))
    if any(abs(value) > 1e-12 for value in tool_offset) or abs(resin_z) > 1e-12:
        raise ValueError("parallel layer export requires the offline zero-offset base contract")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    worker_root = Path(tempfile.mkdtemp(prefix="core-layers-", dir=destination.parent))
    initial_tool_id = int(export_kwargs.get("initial_tool_id", 2))
    tools = _initial_tools(chunks, initial_tool_id)
    tools_before_command = _tool_before_indices(parsed_commands, initial_tool_id)
    worker_kwargs = dict(export_kwargs)
    worker_kwargs.pop("progress_callback", None)
    worker_kwargs["chunk_size"] = max(1_000_000, int(worker_kwargs.get("chunk_size", 100_000)))
    worker_kwargs["split_by_layer_type"] = False
    worker_kwargs["plot_layer_xy"] = False

    payloads = []
    for index, ((layer, start, _end, commands), tool) in enumerate(zip(chunks, tools)):
        # Replaying only the preceding layer's final state-bearing suffix gives
        # the unchanged serial exporter the exact boundary pose, tool, E and
        # pending CUT state without paying to sample the whole layer twice.
        if index:
            _previous_layer, previous_start, _previous_end, previous_commands = chunks[index - 1]
            context_offset, context_commands = _boundary_context(previous_commands)
            worker_start = previous_start + context_offset
            worker_commands = [*context_commands, *commands]
            worker_tool = tools_before_command[worker_start]
        else:
            worker_start = start
            worker_commands = commands
            worker_tool = tool
        kwargs = dict(worker_kwargs)
        kwargs["initial_tool_id"] = worker_tool
        payloads.append((
            worker_start,
            layer,
            len(commands),
            worker_commands,
            str(worker_root / f"layer_{layer:04d}.npz"),
            kwargs,
        ))

    # Keep logical layer order. With more layers than workers the symmetric
    # conformal stack naturally pairs its light outer layers with the heavy
    # centre layers. Sorting the centre layers first made all workers enter
    # the memory-bandwidth-heavy phase together and created a slow second
    # wave on Windows. Merge order is still resolved explicitly below.
    payloads.sort(key=lambda payload: int(payload[1]))

    results_by_start: dict[int, dict[str, Any]] = {}
    completed_commands = 0
    total_commands = max(1, len(parsed_commands))
    try:
        # Signal that Core has accepted the job before waiting for the first
        # whole layer. Otherwise a long first worker leaves the UI displaying
        # the preceding SourceJob phase even though all Core workers run.
        if progress_callback is not None:
            progress_callback(0.0)
        worker_count = min(int(max_workers), len(payloads))
        worker_pool_started = time.perf_counter()

        def collect_worker_results(executor: ProcessPoolExecutor) -> None:
            nonlocal completed_commands
            future_to_payload = {}
            for dispatch_order, payload in enumerate(payloads):
                submitted_payload = (*payload, dispatch_order, time.perf_counter())
                future_to_payload[
                    executor.submit(_export_layer_worker, submitted_payload)
                ] = submitted_payload
            for future in as_completed(future_to_payload):
                result = future.result()
                results_by_start[int(result["target_layer"])] = result
                completed_commands += int(result["target_command_count"])
                if progress_callback is not None:
                    progress_callback(min(0.98, completed_commands / total_commands))

        if reuse_workers:
            executor, created = _persistent_worker_pool(worker_count)
            environment = _single_thread_worker_numeric_environment() if created else nullcontext()
            with environment:
                collect_worker_results(executor)
        else:
            with _single_thread_worker_numeric_environment():
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    collect_worker_results(executor)

        worker_pool_s = time.perf_counter() - worker_pool_started
        merge_started = time.perf_counter()
        results = sorted(results_by_start.values(), key=lambda result: result["target_layer"])
        arrays, row_counts = _load_row_arrays(results)
        total_rows = sum(row_counts)
        if total_rows <= 0:
            raise ValueError("parallel Core workers produced no rows")

        manifest, injection_mappings = _merge_manifests(results)
        manifest_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        arrays["seq"] = np.arange(total_rows, dtype=np.uint32)
        arrays["total_layers"] = np.full(
            total_rows,
            max(layer for layer, _start, _end, _commands in chunks) + 1,
            dtype=np.uint32,
        )

        path_parts = []
        path_mappings: list[dict[int, int]] = []
        block_parts = []
        role_parts = []
        for result_index, (result, mapping) in enumerate(zip(results, injection_mappings)):
            with np.load(result["output_path"], allow_pickle=False) as data:
                selected = data["preview_layer_index"] == result["target_layer"]
                local_paths = np.array(data["path_id"][selected], copy=True)
                positive_ids = np.unique(local_paths[local_paths > 0])
                if result_index == 0:
                    path_id_offset = 0
                else:
                    previous_layer = int(results[result_index - 1]["target_layer"])
                    prefix = data["preview_layer_index"] == previous_layer
                    prefix_indices = np.flatnonzero(prefix)
                    if not prefix_indices.size:
                        raise ValueError("parallel Core overlap produced no preceding-layer path ids")
                    previous_result = results[result_index - 1]
                    with np.load(previous_result["output_path"], allow_pickle=False) as previous_data:
                        previous_selected = (
                            previous_data["preview_layer_index"]
                            == previous_result["target_layer"]
                        )
                        previous_keys = zip(
                            previous_data["src_line"][previous_selected],
                            previous_data["move_type"][previous_selected],
                            previous_data["event_flag"][previous_selected],
                            previous_data["event_type"][previous_selected],
                            path_parts[result_index - 1],
                        )
                        global_path_by_row_key = {
                            (src, int(move), int(event), int(event_type)): int(path_id)
                            for src, move, event, event_type, path_id in previous_keys
                            if int(path_id) > 0
                        }
                    path_id_offset = None
                    for row_index in reversed(prefix_indices):
                        local_path_id = int(data["path_id"][row_index])
                        key = (
                            data["src_line"][row_index],
                            int(data["move_type"][row_index]),
                            int(data["event_flag"][row_index]),
                            int(data["event_type"][row_index]),
                        )
                        global_path_id = global_path_by_row_key.get(key)
                        if local_path_id > 0 and global_path_id is not None:
                            path_id_offset = global_path_id - local_path_id
                            break
                    if path_id_offset is None:
                        raise ValueError("parallel Core could not align a layer boundary path id")
                path_mapping = {
                    int(local_path_id): int(local_path_id) + path_id_offset
                    for local_path_id in positive_ids
                }
                # ``positive_ids`` is sorted by ``np.unique``.  The former
                # per-id boolean assignment scanned every trajectory row once
                # for every path in the layer.  One lookup maps all positive
                # IDs while preserving the exact integer mapping.
                positive_mask = local_paths > 0
                if np.any(positive_mask):
                    mapped_ids = np.asarray(
                        [path_mapping[int(path_id)] for path_id in positive_ids],
                        dtype=local_paths.dtype,
                    )
                    local_paths[positive_mask] = mapped_ids[
                        np.searchsorted(positive_ids, local_paths[positive_mask])
                    ]
                path_mappings.append(path_mapping)
                path_parts.append(local_paths)
                local_blocks = np.array(data["core_injection_block_id"][selected], copy=True)
                remapped = np.full(local_blocks.shape, -1, dtype=np.int32)
                for old_id, new_id in mapping.items():
                    remapped[local_blocks == old_id] = new_id
                local_roles = np.array(data["core_injection_role"][selected], copy=True)
                block_parts.append(remapped)
                role_parts.append(local_roles)
        arrays["path_id"] = np.concatenate(path_parts)
        arrays["core_injection_block_id"] = np.concatenate(block_parts)
        arrays["core_injection_role"] = np.concatenate(role_parts)

        event_rows = arrays["event_flag"] != 0
        arrays["trigger_seq"] = np.where(
            event_rows,
            arrays["seq"].astype(np.int32),
            np.int32(-1),
        )
        dt = float(export_kwargs.get("dt", 0.004))
        planned = np.empty(total_rows, dtype=np.float32)
        planned_exact = np.empty(total_rows, dtype=np.float64)
        clock_before_row = np.empty(total_rows, dtype=np.float64)
        clock = 0.0
        trajectory_rows = 0
        for index, is_event in enumerate(event_rows):
            clock_before_row[index] = clock
            if not is_event:
                if trajectory_rows:
                    clock += dt
                trajectory_rows += 1
            planned[index] = clock
            planned_exact[index] = clock
        arrays["planned_time_s"] = planned

        with np.load(results[0]["output_path"], allow_pickle=False) as first:
            vocabs = {field: np.array(first[field], copy=True) for field in _VOCAB_FIELDS}
        write_started = time.perf_counter()
        np.savez_compressed(
            destination,
            seq=arrays["seq"], x=arrays["x"], y=arrays["y"], z=arrays["z"],
            a=arrays["a"], b=arrays["b"], c=arrays["c"], e=arrays["e"],
            tool_id=arrays["tool_id"], move_type=arrays["move_type"],
            src_line=arrays["src_line"], event_flag=arrays["event_flag"],
            event_type=arrays["event_type"], payload=arrays["payload"],
            trigger_seq=arrays["trigger_seq"], layer_index=arrays["layer_index"],
            total_layers=arrays["total_layers"],
            preview_layer_index=arrays["preview_layer_index"],
            path_id=arrays["path_id"], path_end_flag=arrays["path_end_flag"],
            planned_time_s=arrays["planned_time_s"],
            move_type_vocab_keys=vocabs["move_type_vocab_keys"],
            move_type_vocab_vals=vocabs["move_type_vocab_vals"],
            event_type_vocab_keys=vocabs["event_type_vocab_keys"],
            event_type_vocab_vals=vocabs["event_type_vocab_vals"],
            core_injection_manifest=np.asarray(manifest_json, dtype="U"),
            core_injection_block_id=arrays["core_injection_block_id"],
            core_injection_role=arrays["core_injection_role"],
            core_injection_role_vocab_keys=vocabs["core_injection_role_vocab_keys"],
            core_injection_role_vocab_vals=vocabs["core_injection_role_vocab_vals"],
        )
        merge_write_s = time.perf_counter() - write_started

        segments = []
        event_count = int(np.count_nonzero(event_rows))
        seq_offset = 0
        for result, path_mapping, row_count in zip(results, path_mappings, row_counts):
            timing_path = Path(result["output_path"]).with_suffix(".timing.json")
            local_timing = json.loads(timing_path.read_text(encoding="utf-8"))
            with np.load(result["output_path"], allow_pickle=False) as data:
                selected_indices = np.flatnonzero(
                    data["preview_layer_index"] == result["target_layer"]
                )
            local_seq_start = int(selected_indices[0])
            for raw_segment in local_timing.get("segments", []):
                local_path_id = int(raw_segment["path_id"])
                if local_path_id not in path_mapping:
                    continue
                segment = dict(raw_segment)
                segment["path_id"] = path_mapping[local_path_id]
                segment["start_seq"] = int(segment["start_seq"]) - local_seq_start + seq_offset
                segment["end_seq"] = int(segment["end_seq"]) - local_seq_start + seq_offset
                segment["duration_s"] = (
                    float(planned_exact[segment["end_seq"]])
                    - float(clock_before_row[segment["start_seq"]])
                )
                segments.append(segment)
            seq_offset += row_count
        timing_payload = {
            "format": "rsi_print_timing",
            "version": 1,
            "sample_period_s": dt,
            "total_planned_time_s": clock,
            "trajectory_rows": trajectory_rows,
            "event_rows_ignored": event_count,
            "segments": segments,
        }
        timing_path = destination.with_suffix(".timing.json")
        timing_path.write_text(
            json.dumps(timing_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_offset_sidecar(destination, manifest, tool_offset, resin_z)

        stats: dict[str, Any] = {key: 0.0 for key in _TIMING_SUM_KEYS}
        for result in results:
            for key in _TIMING_SUM_KEYS:
                stats[key] += float(result["stats"].get(key, 0.0))
        stats["write_s"] += merge_write_s
        worker_sum_s = sum(
            float(result["stats"].get("total_s", 0.0)) for result in results
        )
        worker_max_s = max(
            float(result["stats"].get("total_s", 0.0)) for result in results
        )
        theoretical_worker_floor_s = max(worker_sum_s / worker_count, worker_max_s)
        first_wave = [
            result for result in results if int(result["dispatch_order"]) < worker_count
        ]
        first_wave_start_span_s = (
            max(float(result["worker_started_at"]) for result in first_wave)
            - min(float(result["worker_started_at"]) for result in first_wave)
            if first_wave
            else 0.0
        )
        stats.update({
            "total_s": time.perf_counter() - started,
            "rows": total_rows,
            "parts": 1,
            "detailed_sampling_timing": bool(
                export_kwargs.get("collect_detailed_timing", False)
            ),
            "timing_sidecar": str(timing_path),
            "planned_total_time_s": clock,
            "parallel_workers": worker_count,
            "parallel_layers": len(payloads),
            # Diagnostics only: returned to the UI response, never embedded
            # in the final NPZ or its sidecars.
            "parallel_worker_pool_s": worker_pool_s,
            "parallel_worker_sum_s": worker_sum_s,
            "parallel_worker_max_s": worker_max_s,
            "parallel_worker_floor_s": theoretical_worker_floor_s,
            "parallel_schedule_gap_s": max(0.0, worker_pool_s - theoretical_worker_floor_s),
            "parallel_worker_utilization": min(
                1.0, worker_sum_s / max(worker_count * worker_pool_s, 1e-9)
            ),
            "parallel_first_wave_start_span_s": first_wave_start_span_s,
            "parallel_max_queue_delay_s": max(
                float(result["queue_delay_s"]) for result in results
            ),
            "parallel_layer_timings": [
                {
                    "layer": int(result["target_layer"]),
                    "worker_pid": int(result["worker_pid"]),
                    "dispatch_order": int(result["dispatch_order"]),
                    "queue_delay_s": float(result["queue_delay_s"]),
                    "worker_s": float(result["stats"].get("total_s", 0.0)),
                    "rows": int(result["stats"].get("rows", 0)),
                    "fit_s": float(result["stats"].get("fit_s", 0.0)),
                    "sample_s": float(result["stats"].get("sample_s", 0.0)),
                    "write_s": float(result["stats"].get("write_s", 0.0)),
                    "sample_arc_map_s": float(
                        result["stats"].get("sample_arc_map_s", 0.0)
                    ),
                    "sample_lookup_s": float(
                        result["stats"].get("sample_lookup_s", 0.0)
                    ),
                    "sample_deboor_s": float(
                        result["stats"].get("sample_deboor_s", 0.0)
                    ),
                    "sample_pose_s": float(
                        result["stats"].get("sample_pose_s", 0.0)
                    ),
                    "sample_extrude_s": float(
                        result["stats"].get("sample_extrude_s", 0.0)
                    ),
                }
                for result in results
            ],
            "parallel_merge_s": time.perf_counter() - merge_started,
        })
        if progress_callback is not None:
            progress_callback(1.0)
        return stats
    finally:
        shutil.rmtree(worker_root, ignore_errors=True)


__all__ = ["export_npz_parallel_by_layer"]
