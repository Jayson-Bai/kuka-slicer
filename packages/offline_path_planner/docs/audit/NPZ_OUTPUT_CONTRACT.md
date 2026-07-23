# System NPZ output contract

## Contract authority

The active writer is the nested streaming writer in
`path_processing_core.npz_exporter`. It builds typed arrays and passes them to
`numpy.savez_compressed`
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:230`,
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:257`).
The frozen external-template output is the current value-level golden
(`handoff/GOLDEN_SHA256SUMS:1`).

There is no schema/version field inside the NPZ. Therefore, the exact field
names, dtypes, vocabulary values, and sidecar behavior below form the de facto
version-0 contract.

## NPZ arrays

All row arrays have length `N`. Vocabulary arrays have the vocabulary length.

| # | Name | Dtype | Meaning | C++ consumer |
|---:|---|---|---|---|
| 1 | `seq` | `uint32` | Global exported row sequence | Required |
| 2 | `x` | `float32` | Cartesian X | Required |
| 3 | `y` | `float32` | Cartesian Y | Required |
| 4 | `z` | `float32` | Cartesian Z | Required |
| 5 | `a` | `float32` | Euler/orientation A | Required |
| 6 | `b` | `float32` | Euler/orientation B | Required |
| 7 | `c` | `float32` | Euler/orientation C | Required |
| 8 | `e` | `float32` | Cumulative extrusion state | Required |
| 9 | `tool_id` | `uint8` | System tool ID | Required, widened to `int32` |
| 10 | `move_type` | `uint8` | Vocabulary-coded row movement type | Required |
| 11 | `src_line` | `S32` | Fixed-width source-line text | Required |
| 12 | `event_flag` | `uint8` | `1` identifies an event row | Required |
| 13 | `event_type` | `uint8` | Vocabulary-coded event type | Required |
| 14 | `payload` | `S32` | Fixed-width event payload | Required |
| 15 | `trigger_seq` | `int32` | Sequence threshold for event execution; `-1` is fallback/none | Required |
| 16 | `layer_index` | `uint32` | Logical layer for runtime progress | Optional, defaults to zero |
| 17 | `total_layers` | `uint32` | Total logical layers | Optional, defaults to zero |
| 18 | `preview_layer_index` | `int32` | Preview grouping layer; may be distinct from runtime layer | Not read by supplied C++ loader |
| 19 | `path_id` | `uint32` | Stable exported path/segment ID; zero means none | Optional, defaults to zero |
| 20 | `path_end_flag` | `uint8` | `1` on the last row of a path | Optional, defaults to zero |
| 21 | `planned_time_s` | `float32` | Cumulative planned trajectory time at this row | Optional; valid only with timing sidecar |
| 22 | `move_type_vocab_keys` | `S32` | Movement vocabulary labels | Read once when present |
| 23 | `move_type_vocab_vals` | `uint8` | Movement vocabulary IDs | Read once when present |
| 24 | `event_type_vocab_keys` | `S32` | Event vocabulary labels | Read once when present |
| 25 | `event_type_vocab_vals` | `uint8` | Event vocabulary IDs | Read once when present |

The writer's dtype construction is at
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:226`
through `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:255`;
the persisted field order is at
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:259`
through `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:283`.

The supplied C++ loader treats fields 1–15 as mandatory through `npz.at`
(`src/my_project/control_center/src/npz_loader.cpp:406` through
`src/my_project/control_center/src/npz_loader.cpp:426`). It gives compatibility
defaults only to planned time and fields 16, 17, 19, and 20
(`src/my_project/control_center/src/npz_loader.cpp:429` through
`src/my_project/control_center/src/npz_loader.cpp:467`).

## Vocabularies

Movement vocabulary
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:151`):

| Label | ID |
|---|---:|
| `TRAVEL` | 0 |
| `PRINT` | 1 |
| `TRAVEL_FIT` | 2 |
| `PRINT_FIT` | 3 |
| `EVENT` | 4 |

Event vocabulary
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:158`):

| Label | ID |
|---|---:|
| empty | 0 |
| `heat_cf` | 1 |
| `heat_resin` | 2 |
| `fan_cf` | 3 |
| `fan_resin` | 4 |
| `extrude_reset` | 5 |
| `tool_change_cf` | 6 |
| `tool_change_resin` | 7 |
| `cut` | 8 |

Unknown producer labels are serialized as ID `255`
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:235`,
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:239`).

## Row semantics

### Trajectory rows

Trajectory rows have `event_flag=0` and are converted to
`TrajectoryPoint`. The C++ queue copies XYZABCE, tool, sequence, layer, path,
and planned-time fields
(`src/my_project/control_center/src/queue_manager.cpp:52` through
`src/my_project/control_center/src/queue_manager.cpp:69`).

`path_end_flag=1` marks the last emitted row for a nonzero `path_id`; the writer
clears earlier candidates before setting the final row
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:356`
through `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:370`).

### Event rows

Event rows:

- reuse the current held pose;
- have `move_type=EVENT` and `event_flag=1`;
- set `trigger_seq` to their own `seq`;
- do not advance planned trajectory time.

Producer evidence:
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1517`
through `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1539`.

The C++ queue creates `PlannedEvent`, falls back to row `seq` only when
`trigger_seq < 0`, does not enqueue the event as a trajectory row, and waits
when the next trajectory sequence reaches the trigger
(`src/my_project/control_center/src/queue_manager.cpp:39` through
`src/my_project/control_center/src/queue_manager.cpp:49`,
`src/my_project/control_center/src/queue_manager.cpp:111`).
The message-level event contract is event type, payload, source line, and
trigger sequence
(`src/my_project/my_project_interfaces/msg/PlannedEvent.msg:3`).

### Planned time

The first trajectory row is time zero. Each later trajectory row increments by
`dt`; events return the current time without incrementing it
(`src/my_project/path_processing_core/path_processing_core/rsi_timing.py:23`,
`src/my_project/path_processing_core/path_processing_core/rsi_timing.py:32`).

The C++ loader reports planned time as valid only if all row values are finite
and the timing sidecar contains a valid nonnegative total
(`src/my_project/control_center/src/npz_loader.cpp:143`,
`src/my_project/control_center/src/npz_loader.cpp:203`).

## File topology

### Flat output

- Requested base `<name>.npz`.
- Writer initially creates `<name>_partNNNN.npz`.
- A one-part output is renamed to `<name>.npz`; multi-part output retains the
  numbered parts
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:290`).
- `<name>.offset.json` stores `tool_offset` and
  `resin_z_print_compensation_mm`
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1822`).
- `<name>.timing.json` stores the timing summary.

The C++ loader first accepts an exact `.npz`, otherwise discovers and
lexicographically sorts `<stem>_part*.npz`
(`src/my_project/control_center/src/npz_loader.cpp:291` through
`src/my_project/control_center/src/npz_loader.cpp:328`).

### Split-by-layer/type output

Each occurrence receives a base under `layer_NNNN/`; the manifest records
layer, subtype, occurrence, `base_path`, and sequence bounds
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:312`
through `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:335`).
The C++ loader extracts each `base_path` and supports relative paths plus a
limited moved-manifest fallback
(`src/my_project/control_center/src/npz_loader.cpp:331` through
`src/my_project/control_center/src/npz_loader.cpp:391`).

## Consumer compatibility matrix

| Feature | Python preview | C++ loader/queue |
|---|---|---|
| Flat `.npz` | yes | yes |
| Numbered parts | yes | yes |
| Split manifest | yes | yes |
| Event rows | filtered/represented for preview logic | yes, separate event queue |
| `layer_index` | yes | yes |
| `preview_layer_index` | preferred when present (`src/my_project/gcode_planner/gcode_planner/path_preview.py:339`) | no |
| `path_id` / end flag | yes (`src/my_project/gcode_planner/gcode_planner/path_preview.py:530`) | yes |
| `planned_time_s` | array can be read as NPZ data | yes, sidecar-gated |
| Vocabularies | yes | yes |

## Frozen compatibility rules

Until a versioned schema and dual-repository tests are approved:

1. Do not rename, remove, reorder, or change the dtype of the 25 arrays.
2. Do not change vocabulary IDs.
3. Preserve `seq`, event-row, and `trigger_seq` semantics.
4. Preserve flat and multipart base-path resolution.
5. Preserve offset and timing sidecars.
6. Treat `preview_layer_index` as Python-preview metadata; do not assume the
   supplied C++ runtime consumes it.
7. Validate value-level array equality across platforms; do not require
   generated NPZ ZIP bytes to be identical across NumPy/platform versions.

No contract field or behavior was changed during phase one.
