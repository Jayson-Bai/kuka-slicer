# Dependency audit

## Current package graph

```text
external_npz_preprocessor ─┐
                           ├─> path_processing_core ─> numpy
gcode_planner ─────────────┘
     └─> rclpy

my_project_ui (read-only reference)
     ├─> both offline producers
     ├─> path_processing_core.head_calibration
     └─> ROS 2, RQt, Qt, messages, launch files, realtime nodes

control_center (read-only reference)
     └─> cnpy + ROS messages + system NPZ contract
```

The three installable Python packages declare the following:

| Package | Declared Python dependencies | ROS package dependencies | Direct runtime imports |
|---|---|---|---|
| `path_processing_core` | `setuptools`, `numpy` (`src/my_project/path_processing_core/setup.py:13`) | `python3-numpy` (`src/my_project/path_processing_core/package.xml:10`) | Standard library and NumPy; no `rclpy` |
| `gcode_planner` | `setuptools`, `numpy`, `rclpy`, `path_processing_core` (`src/my_project/gcode_planner/setup.py:14`) | Same logical set (`src/my_project/gcode_planner/package.xml:10`) | Parser imports `rclpy` at module load (`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:8`) |
| `external_npz_preprocessor` | `setuptools`, `numpy`, `path_processing_core` (`src/my_project/external_npz_preprocessor/setup.py:14`) | NumPy and shared core (`src/my_project/external_npz_preprocessor/package.xml:10`) | CLI path is standard library + NumPy + core; UI imports `python_qt_binding` (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py:7`) |

All three packages use version `0.0.0`
(`src/my_project/path_processing_core/setup.py:7`,
`src/my_project/gcode_planner/setup.py:7`,
`src/my_project/external_npz_preprocessor/setup.py:8`). There is therefore no
package-version signal with which to negotiate an NPZ contract revision.

## Runtime findings

### Shared core is already mostly ROS-independent

`path_processing_core` imports no ROS package. Its API consists of command
dataclasses, fitting/sampling code, calibration JSON, timing, and NumPy export
(`src/my_project/path_processing_core/path_processing_core/types.py:6`,
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:19`).
This is the correct lowest dependency layer.

### GCode CLI has an avoidable ROS import boundary

The offline CLI imports parser functions
(`src/my_project/gcode_planner/gcode_planner/cli.py:10`), while the parser
module imports `rclpy` and subclasses `Node`
(`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:8`,
`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:92`).
Consequently, a nominally offline parse/export command needs a ROS Python
environment even though `parse_gcode_lines` itself is a pure function
(`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:50`).

This is not a phase-one defect to fix. It is a phase-two packaging separation
candidate.

### External NPZ CLI is ROS-independent, but its optional UI dependency is undeclared

The CLI calls `convert_external_npz`
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/cli.py:92`),
which uses only the shared core and NumPy
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/export_runner.py:7`).

The separately exposed UI entry point imports `python_qt_binding`
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/app.py:7`),
but neither `setup.py` nor `package.xml` declares a Qt/Python binding
(`src/my_project/external_npz_preprocessor/setup.py:14`,
`src/my_project/external_npz_preprocessor/package.xml:10`).
**Finding:** CLI installation can be valid while the advertised UI entry point
fails at import time.

### Utility dependencies are not packaged

Plotting utilities import NumPy and/or Matplotlib and live under `scripts/`,
outside package metadata. The exporter also conditionally imports Matplotlib
only for previews
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1834`).
**Inference:** a minimal export installation does not need Matplotlib, but
plotting and `--plot-layer-xy` do. No optional dependency group documents that
distinction.

### Read-only consumer fragments are intentionally incomplete

The C++ loader uses `cnpy` in its implementation
(`src/my_project/control_center/src/npz_loader.cpp:1`) and the queue manager
uses ROS message types (`src/my_project/control_center/include/control_center/queue_manager.hpp:1`).
No `CMakeLists.txt` or `package.xml` for `control_center` is present in the
handoff index (`handoff/SOURCE_TREE.tsv:36` through
`handoff/SOURCE_TREE.tsv:39`). These files are contract evidence, not a
standalone build target.

The RQt panel imports `UiStatus` and `ExtruderLatencyStatus` in addition to
`TrajectoryPoint`
(`src/my_project/my_project_ui/my_project_ui/ui_panel.py:14`), while only
`PlannedEvent.msg` and `TrajectoryPoint.msg` were handed off
(`handoff/SOURCE_TREE.tsv:100` through `handoff/SOURCE_TREE.tsv:101`).
It also references a VTK preview module
(`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5231`) and
`startup.launch.py` (`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5908`)
that are absent. This confirms that the UI is a read-only integration excerpt.

## Test dependencies

The package manifests declare `pytest`, `ament_copyright`, `ament_flake8`, and
`ament_pep257` as test dependencies
(`src/my_project/gcode_planner/package.xml:14`,
`src/my_project/external_npz_preprocessor/package.xml:13`,
`src/my_project/path_processing_core/package.xml:12`).

Observed test environments:

| Environment | Versions/capability | Result |
|---|---|---|
| Windows | Python 3.13.1, NumPy 2.2.0, pytest 9.1.1; no `rclpy`/ament | 155 functional tests passed with repository test shims; static launchers excluded |
| WSL `Ubuntu2204_ros2` | Ubuntu/ROS 2 Humble, Python 3.10.12, NumPy 1.21.5, pytest 6.2.5, ROS/ament under `/opt/ros/humble` | Three static launchers ran when ROS site-packages were explicitly preserved; baseline is 1 skipped and 2 failed |

The static-test Python modules import ament directly
(`src/my_project/gcode_planner/test/test_flake8.py:15`,
`src/my_project/gcode_planner/test/test_pep257.py:15`,
`src/my_project/gcode_planner/test/test_copyright.py:15`).

## Dependency risks

| Risk | Impact | Phase-one status |
|---|---|---|
| Offline GCode CLI imports ROS at module load | Prevents a plain-Python standalone install | Documented; no source change |
| External UI dependency absent from metadata | UI entry point may fail after otherwise successful installation | Documented; no source change |
| No package/contract version | Consumers cannot negotiate schema changes | Documented; schema frozen |
| Plot dependencies not expressed | Preview/plot commands may fail late | Documented |
| C++/UI excerpts lack their complete build/runtime trees | Cannot run end-to-end upper-computer integration from this repo | Expected handoff boundary; not a phase-one blocker |

## Phase-two dependency direction

After owner approval, preserve this dependency rule:

1. `path_processing_core` remains ROS-free.
2. Pure GCode parsing is separated from the optional ROS node adapter.
3. External CLI dependencies remain minimal; Qt is an explicit optional extra
   or separate frontend package.
4. Upper-computer code consumes a versioned file/task interface and does not
   import internal planner implementation modules.
5. Realtime ROS packages and hardware protocols do not enter the offline
   planner repository.

This is a proposal only; no package metadata or import was changed in phase one.
