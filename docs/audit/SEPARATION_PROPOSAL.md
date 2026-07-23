# Offline/realtime separation proposal

## Status

This is a proposal only. It authorizes no source move, deletion, schema change,
package split, or upper-computer modification. Phase two begins only after the
repository owner reviews the complete audit and explicitly approves it
(`EXTERNAL_CODEX_HANDOFF.md:193`).

## Current coupling

The offline producers and realtime upper-computer code are coupled in three
ways:

1. The RQt panel imports and calls the external-NPZ producer directly
   (`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5120`).
2. The same panel imports the GCode parser, primeline, and shared exporter
   directly (`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5136`).
3. After export, the panel launches the realtime system through
   `startup.launch.py` with `npz_path`
   (`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5903`).

The file boundary already exists operationally: both producers create a system
NPZ, and the realtime loader resolves a base path or manifest
(`src/my_project/control_center/src/npz_loader.cpp:277`). The recommended
separation formalizes that boundary instead of sharing Python internals.

## Proposed ownership

### Offline planner repository

Retain:

- `path_processing_core`;
- GCode parser, primeline, offline CLI, preview, and test-data generators;
- external source-NPZ loader, parameter model, converter, CLI, and optional
  standalone frontend;
- source templates and representative GCode/source-NPZ fixtures;
- NPZ writer, schema/golden contract, sidecar definitions, and producer tests;
- developer estimation/plot/split scripts.

These components are the active producer scope shown in
`handoff/SOURCE_TREE.tsv:1` through `handoff/SOURCE_TREE.tsv:35` and
`handoff/SOURCE_TREE.tsv:40` through `handoff/SOURCE_TREE.tsv:123`, excluding
the read-only consumer excerpts.

### Realtime upper-computer repository

Retain:

- RQt/operator UI;
- ROS launch files and parameters;
- `control_center`, queue management, realtime timing/RSI nodes;
- UART, hardware, and startup behavior;
- ROS messages and live status reporting;
- consumer-side NPZ validation and compatibility tests.

The supplied reference files demonstrate this ownership:
`src/my_project/control_center/src/queue_manager.cpp:24`,
`src/my_project/my_project_interfaces/msg/TrajectoryPoint.msg:1`, and
`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5903`.

### Shared contract, not shared implementation

Both repositories own tests for:

- the 25-array NPZ schema;
- vocabulary IDs;
- event/trigger sequence semantics;
- flat, multipart, and manifest path resolution;
- offset and timing sidecars;
- one or more frozen golden jobs.

The producer owns serialization. The consumer owns loading and runtime
validation. Neither repository imports the other's private source modules.

## Proposed interface

The first supported interface should remain file/job based:

```text
Upper-computer request
  -> invoke offline planner CLI or task runner
       inputs: source file + explicit parameter/calibration snapshot
       outputs: result directory + machine-readable job result
  -> validate declared contract version
  -> hand NPZ base/manifest path to realtime loader
```

Minimum job result:

- success/failure and diagnostic text;
- planner/build version;
- NPZ contract version;
- source and parameter fingerprints;
- output base/manifest path;
- offset/timing sidecar paths;
- row/part/layer counts and total planned time.

This avoids moving ROS launch, RSI, UART, or hardware code into the offline
repository and avoids importing planner internals into the upper-computer UI.

## Required phase-two order

### 1. Freeze executable consumer contracts

Before refactoring:

- add producer tests against the existing frozen golden;
- add C++ consumer tests that load the same golden;
- make event and timing assertions explicit;
- decide whether `preview_layer_index` is UI-only;
- record current backward-compatible defaults.

The present C++ mandatory/optional split is visible at
`src/my_project/control_center/src/npz_loader.cpp:406` and
`src/my_project/control_center/src/npz_loader.cpp:429`.

### 2. Introduce NPZ contract versioning without breaking version 0

Add a version in a backward-compatible form and continue emitting/accepting the
current 25 fields. Do not repurpose vocabulary IDs. Add migrations/readers
before any producer starts emitting a changed schema.

### 3. Make offline packaging standalone

- separate pure GCode parsing from the `rclpy` node adapter;
- keep `path_processing_core` ROS-free;
- express NumPy, test, Qt, and plot dependencies accurately;
- make calibration/parameter snapshots explicit inputs;
- retain current console entry points or provide compatibility aliases.

The current ROS import boundary is
`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:8`.

### 4. Expose one stable planner invocation

Prefer a CLI plus machine-readable result first. A local service/task API can
follow if cancellation, progress, or concurrency requires it. Both GCode and
external NPZ should use the same job/result envelope while retaining their
input-specific parameter models.

### 5. Migrate upper-computer calls

Replace the direct imports at
`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5120` and
`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5136` with the stable
invocation. Keep realtime launch and runtime status in the upper-computer
repository.

### 6. Deprecate compatibility wrappers deliberately

The `gcode_planner` re-export modules are retained until:

- both repositories have been searched at pinned commits;
- downstream imports have migrated;
- a deprecation window has elapsed;
- contract and functional tests pass.

Do not delete them merely because they are thin; examples include
`src/my_project/gcode_planner/gcode_planner/npz_exporter.py:1` and
`src/my_project/gcode_planner/gcode_planner/types.py:1`.

### 7. Resolve dormant implementations separately

Decide whether external source paths should remain smoothed polylines or
activate validated spline fitting. This is an algorithm decision, not a
packaging cleanup. The current polyline path is at
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:566`;
the dormant spline helper is at
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:685`.

Likewise, decide whether the unused private `_npz_exporter` can be removed only
after the active writer's contract is fully covered
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1946`).

## Explicit non-goals

- No realtime RSI/UART/hardware/startup implementation enters this repository.
- No NPZ schema or vocabulary changes as part of directory cleanup.
- No copying missing upper-computer files into the handoff.
- No change to source trajectory Z semantics.
- No activation of dormant spline code without algorithm-specific tests.
- No claim that hardware or full upper-computer integration has been tested.

## Approval gates

Owner approval should answer:

1. Is the proposed repository ownership boundary correct?
2. Is the system NPZ the long-term inter-repository boundary?
3. Is `preview_layer_index` intentionally preview-only?
4. Should external paths remain polyline-based or eventually use validated
   spline fitting?
5. Is standalone non-ROS GCode CLI support required?
6. Which repository owns the contract-version definition and golden-release
   process?
7. Can phase two include isolated style cleanup, or must it remain a separate
   change?

## Phase-two acceptance criteria

- Both repositories test the same versioned golden contract.
- Existing version-0 golden output remains loadable.
- GCode and external-NPZ functional baselines remain green.
- Upper-computer UI invokes a stable external interface, not planner internals.
- Realtime launch/hardware behavior remains in the realtime repository.
- Compatibility wrappers are either retained or removed with demonstrated
  downstream evidence.
- No handoff/golden history or tag is rewritten.

Until these gates are approved, this proposal remains documentation only.
