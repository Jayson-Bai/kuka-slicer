# Test and reproducibility baseline

## Baseline identity

| Item | Value |
|---|---|
| Audit date | 2026-07-23 |
| Branch | `audit/offline-planner-baseline` |
| Handoff commit | `444c420d26b5cf8071b681006f015f18ac5dd60f` |
| Handoff tag | `offline-planner-handoff-v1` |
| Source count | 124 (`handoff/SOURCE_TREE.tsv:1`–`handoff/SOURCE_TREE.tsv:124`) |
| Frozen golden count | 3 (`handoff/GOLDEN_SHA256SUMS:1`–`handoff/GOLDEN_SHA256SUMS:3`) |

No command below wrote generated artifacts inside the repository. Export
outputs went to Windows system temp or WSL `/tmp`.

## Integrity checks

The per-file SHA256 lists were recomputed against the working tree.

| Check | Passed | Failed |
|---|---:|---:|
| `handoff/SOURCE_SHA256SUMS` | 124 | 0 |
| `handoff/GOLDEN_SHA256SUMS` | 3 | 0 |

The source archive checksum authority is
`handoff/SOURCE_ARCHIVE_SHA256.txt:1`.

## Functional tests

Environment:

- Windows Python 3.13.1
- NumPy 2.2.0
- pytest 9.1.1
- no native `rclpy` or ament Python modules

Command:

```powershell
$env:PYTHONPATH = "src/my_project/gcode_planner;" +
  "src/my_project/external_npz_preprocessor;" +
  "src/my_project/path_processing_core"
python -m pytest -q `
  src/my_project/gcode_planner/test `
  src/my_project/external_npz_preprocessor/test `
  src/my_project/path_processing_core/test `
  test/scripts/test_plot_npz_xy.py `
  --ignore=src/my_project/gcode_planner/test/test_copyright.py `
  --ignore=src/my_project/gcode_planner/test/test_flake8.py `
  --ignore=src/my_project/gcode_planner/test/test_pep257.py
```

Result: **155 passed, 0 failed, 0 skipped in 2.97 s**.

The excluded files are ament static-test launchers, not functional tests; each
imports its ament runner directly
(`src/my_project/gcode_planner/test/test_copyright.py:15`,
`src/my_project/gcode_planner/test/test_flake8.py:15`,
`src/my_project/gcode_planner/test/test_pep257.py:15`).

## ROS 2 static tests

Environment:

- WSL distribution `Ubuntu2204_ros2`
- ROS 2 Humble under `/opt/ros/humble`
- Python 3.10.12
- NumPy 1.21.5
- pytest 6.2.5
- `ament_copyright`, `ament_flake8`, and `ament_pep257` 0.12.14

The test run explicitly included both planner packages and ROS
site-packages in `PYTHONPATH`. Result:

| Test | Result | Detail |
|---|---|---|
| `test_copyright.py` | skipped | Repository has no generated copyright headers; the ament test deliberately skips (`src/my_project/gcode_planner/test/test_copyright.py:20`) |
| `test_flake8.py` | failed | 44 existing errors across 69 checked Python files |
| `test_pep257.py` | failed | 3 existing D213 errors |

Aggregate: **0 passed, 2 failed, 1 skipped**. These tests were run; they are not
claimed as passing.

### Complete flake8 baseline

| Code | Count | Locations |
|---|---:|---|
| E303 | 3 | `src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py:158`, `src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py:221`, `src/my_project/gcode_planner/test/test_extrude_reset_payload.py:1139` |
| E402 | 4 | `scripts/estimate_npz_rows.py:16`, `scripts/estimate_npz_rows.py:17`, `scripts/estimate_npz_rows.py:24`, `scripts/estimate_npz_rows.py:25` |
| E501 | 34 | `scripts/estimate_npz_rows.py:63`, `scripts/estimate_npz_rows.py:133`; `scripts/plot_layers_from_manifest.py:25`, `scripts/plot_layers_from_manifest.py:67`; `src/my_project/external_npz_preprocessor/external_npz_preprocessor/cli.py:15`; `src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:185`, `src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:320`; `src/my_project/external_npz_preprocessor/external_npz_preprocessor/param_config.py:100`, `src/my_project/external_npz_preprocessor/external_npz_preprocessor/param_config.py:101`, `src/my_project/external_npz_preprocessor/external_npz_preprocessor/param_config.py:102`; `src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:37`, `src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:110`; `src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_template.py:33`; `src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py:307`, `src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py:308`; `src/my_project/external_npz_preprocessor/test/test_converter.py:317`, `src/my_project/external_npz_preprocessor/test/test_converter.py:318`, `src/my_project/external_npz_preprocessor/test/test_converter.py:961`, `src/my_project/external_npz_preprocessor/test/test_converter.py:962`, `src/my_project/external_npz_preprocessor/test/test_converter.py:973`, `src/my_project/external_npz_preprocessor/test/test_converter.py:1115`, `src/my_project/external_npz_preprocessor/test/test_converter.py:1287`, `src/my_project/external_npz_preprocessor/test/test_converter.py:1313`; `src/my_project/external_npz_preprocessor/test/test_export_runner.py:38`; `src/my_project/external_npz_preprocessor/test/test_param_config.py:11`, `src/my_project/external_npz_preprocessor/test/test_param_config.py:121`; `src/my_project/gcode_planner/gcode_planner/cli.py:71`; `src/my_project/gcode_planner/gcode_planner/primeline.py:124`; `src/my_project/my_project_ui/my_project_ui/ui_panel.py:1761`; `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:475`, `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:511`, `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1241`, `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1249`, `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1457` |
| F401 | 1 | `src/my_project/external_npz_preprocessor/test/test_param_config.py:1` |
| W391 | 2 | `src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:131`, `src/my_project/external_npz_preprocessor/external_npz_preprocessor/__init__.py:2` |

Paths shortened to package roots in the long E501 row remain anchored by the
full tracked package ranges at `handoff/SOURCE_TREE.tsv:46` through
`handoff/SOURCE_TREE.tsv:123`.

### Complete pep257 baseline

All three are D213:

- `src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:38`
- `src/my_project/gcode_planner/gcode_planner/primeline.py:27`
- `src/my_project/path_processing_core/path_processing_core/rsi_timing.py:1`

Static style failures were not repaired because phase one forbids source
formatting or cleanup.

## External-NPZ golden reproduction

Input:
`data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz`
(`handoff/SOURCE_TREE.tsv:2`).

Command shape:

```powershell
python -m external_npz_preprocessor.cli `
  --source data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz `
  --out <system-temp>\external-template-system.npz
```

Result:

| Measure | Observed |
|---|---:|
| Exit status | 0 |
| Rows | 43,231 |
| Parts | 1 |
| Sequence | 0–43,230 |
| Final planned time | 172.77200317382812 s |
| Planned time monotonic | yes |
| Field order/dtypes/shapes vs golden | equal |
| Every NPZ array vs golden | `numpy.array_equal` |
| Offset/timing JSON vs golden | semantic equality |

The active writer defines the 25 fields at
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:257`.

## GCode CLI representative export

Environment: ROS 2 Humble in WSL. Input:
`data/input_gcode/split_test0210/test0210_layer_0001.gcode`
(`handoff/SOURCE_TREE.tsv:16`).

```bash
python3 -m gcode_planner.cli \
  --gcode data/input_gcode/split_test0210/test0210_layer_0001.gcode \
  --out /tmp/offline-planner-gcode-audit-20260723-1/gcode-system.npz
```

Result:

| Measure | Observed |
|---|---:|
| Exit status | 0 |
| Rows | 247,616 |
| Parts | 1 |
| Sequence | 0–247,615 |
| Field count | 25 |
| Final planned time | 990.4199829101562 s |
| Planned time monotonic | yes |
| CLI export stage | 2.996 s |

The generated files were one NPZ plus offset and timing sidecars in `/tmp`.

## Baseline interpretation

The functional and real-export baseline is green. The static-style baseline is
known red/skip and is explicitly frozen as such. No unrun end-to-end hardware,
realtime RSI, launch, or upper-computer test is claimed as passing, because the
handoff does not contain the complete runtime system.
