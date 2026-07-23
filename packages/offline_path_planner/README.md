# KUKA Offline Planner

`kuka-offline-planner` is the independently installable offline half of the
printing pipeline. It provides two inputs that converge on the existing system
NPZ exporter:

```text
GCode                         -> gcode_planner
external_layer_paths_v1 NPZ   -> external_npz_preprocessor
                                  |
                                  v
                           path_processing_core
                                  |
                                  v
                              system NPZ
```

The distribution contains three existing Python namespaces so the imported
planner source and legacy import paths remain compatible:

- `gcode_planner`: GCode input adapter and compatibility wrappers.
- `external_npz_preprocessor`: source NPZ input adapter.
- `path_processing_core`: shared command types, planning, interpolation,
  calibration, timing, and system NPZ export.

It does not install the retained upper-computer UI/C++ consumer excerpts and
does not depend on `kuka-slicer`. The only supported connection to
`kuka-slicer` is an `external_layer_paths_v1` NPZ file on disk.

## Standalone installation

From the repository root:

```powershell
python -m pip install .\packages\offline_path_planner
```

The standalone commands are:

```powershell
kuka-offline-gcode --help
kuka-offline-npz --help
```

Example source NPZ conversion:

```powershell
kuka-offline-npz `
  --source .\input\part.npz `
  --out .\output\part-system.npz
```

The pure GCode CLI does not require ROS 2. The legacy
`gcode_planner.gcode_parser:main` node remains available to a ROS/colcon build
and still requires `rclpy`.

## Development tests

Run the functional baseline from this directory:

```powershell
$env:PYTHONPATH = @(
  "src\my_project\path_processing_core"
  "src\my_project\gcode_planner"
  "src\my_project\external_npz_preprocessor"
) -join [IO.Path]::PathSeparator

python -m pytest -q `
  src\my_project\path_processing_core\test `
  src\my_project\gcode_planner\test `
  src\my_project\external_npz_preprocessor\test `
  test\scripts\test_plot_npz_xy.py `
  --ignore=src\my_project\gcode_planner\test\test_copyright.py `
  --ignore=src\my_project\gcode_planner\test\test_flake8.py `
  --ignore=src\my_project\gcode_planner\test\test_pep257.py
```

The handoff checksums and golden outputs are historical evidence. They must not
be regenerated after approved phase-two source changes.
