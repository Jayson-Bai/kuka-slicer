# 正式打印预计剩余时间 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** 在系统 NPZ 导出时保存 RSI 轨迹时间轴，并在正式打印 UI 中按实际 `seq_used` 低频显示总时长、已用时长和预计剩余时长。

**Architecture:** 新增 Python `rsi_timing` 小模块负责按实际输出行累计时间和生成 timing sidecar；`npz_exporter` 只调用该模块，不在运行时重复路径计算。C++ `NpzLoader` 读取可选 `planned_time_s` 和 sidecar 总时长，`QueueManager` 转发到 `TrajectoryPoint`，`SystemManagerNode` 在非 RSI 实时线程的低频 UI 定时器中做 O(1) 剩余时间计算，RQT UI 只负责格式化显示。

**Tech Stack:** Python 3 dataclasses/NumPy/pytest；ROS 2 IDL；C++17/rclcpp/cnpy；PyQt5/RQT。

---

### Task 1: 建立可测试的 RSI 时间累计器

**Files:**
- Create: `src/my_project/path_processing_core/path_processing_core/rsi_timing.py`
- Create: `src/my_project/path_processing_core/test/test_rsi_timing.py`

- [ ] **Step 1: Write the failing tests**

```python
from path_processing_core.rsi_timing import RsiTimingAccumulator


def test_trajectory_rows_advance_by_dt_but_event_rows_do_not():
    timing = RsiTimingAccumulator(dt=0.004)
    assert timing.trajectory_time() == 0.0
    assert timing.append_trajectory_time() == 0.0
    assert timing.append_event_time() == 0.0
    assert timing.append_trajectory_time() == 0.004
    assert timing.append_trajectory_time() == 0.008
    assert timing.trajectory_rows == 3
    assert timing.event_rows_ignored == 1


def test_segment_metadata_preserves_acc_flat_dec_and_sequence_range():
    timing = RsiTimingAccumulator(dt=0.1)
    timing.start_segment(path_id=7, move_type="TRAVEL", start_seq=10)
    timing.append_trajectory_time()
    timing.append_trajectory_time()
    timing.finish_segment(t_acc_s=2.0, t_flat_s=3.5, t_dec_s=2.0, end_seq=11)

    assert timing.segments == [{
        "path_id": 7,
        "move_type": "TRAVEL",
        "start_seq": 10,
        "end_seq": 11,
        "duration_s": 0.1,
        "t_acc_s": 2.0,
        "t_flat_s": 3.5,
        "t_dec_s": 2.0,
    }]


def test_summary_is_json_serializable_and_reports_total_time():
    timing = RsiTimingAccumulator(dt=0.25)
    timing.append_trajectory_time()
    timing.append_trajectory_time()
    summary = timing.summary()

    assert summary["format"] == "rsi_print_timing"
    assert summary["version"] == 1
    assert summary["sample_period_s"] == 0.25
    assert summary["total_planned_time_s"] == 0.25
    assert summary["trajectory_rows"] == 2
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py src/my_project/gcode_planner/test/test_extrude_reset_payload.py -q
```

Expected: FAIL because `path_processing_core.rsi_timing` does not exist.

- [ ] **Step 3: Implement the minimal accumulator**

Implement `RsiTimingAccumulator` with:

```python
class RsiTimingAccumulator:
    def __init__(self, dt: float):
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.dt = float(dt)
        self._time_s = 0.0
        self.trajectory_rows = 0
        self.event_rows_ignored = 0
        self.segments = []
        self._open_segment = None

    def append_trajectory_time(self) -> float:
        value = self._time_s
        if self.trajectory_rows:
            self._time_s += self.dt
            value = self._time_s
        self.trajectory_rows += 1
        return value

    def append_event_time(self) -> float:
        self.event_rows_ignored += 1
        return self._time_s

    def trajectory_time(self) -> float:
        return self._time_s
```

`start_segment()` and `finish_segment()` must append the exact dictionary shape used by the tests, reject nested segments, and use `max(0.0, duration)` for duration safety. `summary()` must return the JSON-ready metadata dictionary from the test.

- [ ] **Step 4: Run the focused test and verify it passes**

Run the same pytest command. Expected: all three tests pass.

- [ ] **Step 5: Commit the isolated unit**

```bash
git add src/my_project/path_processing_core/path_processing_core/rsi_timing.py src/my_project/path_processing_core/test/test_rsi_timing.py
git commit -m "feat: add RSI timing accumulator"
```

### Task 2: Persist timing in system NPZ exports

**Files:**
- Modify: `src/my_project/path_processing_core/path_processing_core/npz_exporter.py`
- Modify: `src/my_project/gcode_planner/test/test_extrude_reset_payload.py`

- [ ] **Step 1: Add a failing exporter contract test**

Append a test using a temporary output and two commands that produce travel, print, and an event. Assert that the generated NPZ contains `planned_time_s`, has the same length as `seq`, that event rows repeat the previous timing value, and that `<base>.timing.json` contains the total and model fields:

```python
import json
import numpy as np


def test_export_npz_writes_rsi_timing_array_and_sidecar(tmp_path):
    out = tmp_path / "timed.npz"
    stats = export_npz(_commands_with_travel_print_and_event(), str(out), dt=0.1)

    with np.load(out) as data:
        assert "planned_time_s" in data
        assert len(data["planned_time_s"]) == len(data["seq"])
        assert np.all(np.isfinite(data["planned_time_s"]))
        event_rows = data["event_flag"] == 1
        assert all(data["planned_time_s"][i] == data["planned_time_s"][i - 1] for i in np.flatnonzero(event_rows))

    metadata = json.loads(
        (tmp_path / "timed.timing.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "rsi_print_timing"
    assert metadata["total_planned_time_s"] >= 0.0
    assert metadata["event_rows_ignored"] >= 1
    assert metadata["segments"]
    assert {"t_acc_s", "t_flat_s", "t_dec_s"}.issubset(metadata["segments"][0])
    assert stats["timing_sidecar"] == str(tmp_path / "timed.timing.json")
```

Place this test in `src/my_project/gcode_planner/test/test_extrude_reset_payload.py`; add `import json` to its existing imports. Add a test-only `_commands_with_travel_print_and_event()` using the existing command classes; it must return one `TRAVEL` MoveCommand, one `_move(...)` PRINT command, and one ResetECommand. Do not create a second exporter implementation.

- [ ] **Step 2: Run the exporter test and verify it fails**

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py src/my_project/gcode_planner/test/test_extrude_reset_payload.py -q
```

Expected: FAIL because exporter rows and sidecar do not contain timing data.

- [ ] **Step 3: Add timing fields and accumulation to the exporter**

Make these changes:

1. Add `planned_time_s: float = 0.0` to `CsvRow`.
2. Create `RsiTimingAccumulator(dt)` once per `export_npz()` call.
3. Before every `writer.add(row)` for a normal trajectory row, set `row.planned_time_s = timing.append_trajectory_time()`.
4. Before every `writer.add(row)` for an event row, set `row.planned_time_s = timing.append_event_time()`; do not advance time.
5. In `_Writer.flush()`, write `planned_time_s=np.array([...], dtype=np.float32)` alongside `seq`.
6. In `_append_sample()`, open and close a segment after the generator has produced its rows. Use the final `InterpolatedPoint.t` as `duration_s`, `gc.time_acc_s` when positive otherwise `2.0`, `t_dec_s=2.0`, and `max(0.0, duration_s - t_acc_s - t_dec_s)` for `t_flat_s`. The sequence range comes from the sampled rows and the segment type/path id already assigned by the exporter.
7. In `_append_extrude_wait()`, record a segment with `duration_s=steps * dt`, `t_acc_s=0.0`, `t_flat_s=steps * dt`, and `t_dec_s=0.0`.
8. Write a timing sidecar after all writers finish. For non-split output use `<base>.timing.json`; for split output use `<base_root>/<base_name>_timing.json`. Add `timing_sidecar` to returned stats. Ensure JSON writing failure logs a warning and does not abort an otherwise valid NPZ export.
9. Do not alter `planned_time_s` for pure event rows, and do not include printing-head wait time or ABORT time.

- [ ] **Step 4: Run the focused exporter tests**

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py src/my_project/gcode_planner/test/test_extrude_reset_payload.py -q
```

Expected: all focused tests pass and existing exporter event tests remain green.

- [ ] **Step 5: Commit the NPZ format change**

```bash
git add src/my_project/path_processing_core/path_processing_core/npz_exporter.py src/my_project/gcode_planner/test/test_extrude_reset_payload.py
git commit -m "feat: persist RSI timing in NPZ exports"
```

### Task 3: Read timing safely in the C++ trajectory pipeline

**Files:**
- Modify: `src/my_project/control_center/include/control_center/npz_loader.hpp`
- Modify: `src/my_project/control_center/src/npz_loader.cpp`
- Modify: `src/my_project/control_center/src/queue_manager.cpp`
- Modify: `src/my_project/my_project_interfaces/msg/TrajectoryPoint.msg`
- Modify: `src/my_project/control_center/test/test_staged_pause_contract.py`

- [ ] **Step 1: Add failing source-contract tests**

Extend the existing runtime contract test with assertions for:

```python
assert "planned_time_s" in msg
assert "planned_total_time_s" in msg
assert "bool planned_time_valid" in msg
assert "planned_time_s" in header
assert "planned_total_time_s" in header
assert "planned_time_valid" in queue
```

Add a loader contract test requiring a timing metadata API:

```python
assert "bool timing_valid() const" in header
assert "double total_planned_time_s() const" in header
assert 'npz.count("planned_time_s")' in loader_cpp
assert "timing.json" in loader_cpp
```

- [ ] **Step 2: Run the contract test and verify it fails**

```bash
python3 -m pytest src/my_project/control_center/test/test_staged_pause_contract.py -q
```

Expected: FAIL because the message and loader APIs are absent.

- [ ] **Step 3: Add optional timing fields to the runtime contract**

Append the three fields to `TrajectoryPoint.msg` exactly as specified:

```text
float32 planned_time_s
float32 planned_total_time_s
bool planned_time_valid
```

Add to `NpzRow` and `NpzChunk`:

```cpp
float planned_time_s{0.0F};
std::vector<float> planned_time_s;
```

Add these public methods to `NpzLoader`:

```cpp
bool timing_valid() const {return timing_valid_;}
double total_planned_time_s() const {return total_planned_time_s_;}
```

Load the optional `planned_time_s` array in `load_chunk()`. If it is missing, mismatched with `seq`, or contains a non-finite value, set `timing_valid_ = false` and keep loading all normal trajectory fields. Read only `total_planned_time_s` from the inferred timing sidecar using a small string-based JSON number extractor; do not add a general JSON dependency. Accept direct NPZ, `_part` base, and `_manifest.json` inputs. A missing or malformed sidecar makes only the timing estimate invalid.

In `next_row()`, copy the timing value when valid, otherwise leave it at `0.0F`. In `QueueManager::fill()`, copy the timing fields to each `TrajectoryPoint` and set `planned_time_valid` only when both the row field and total metadata are valid. Existing trajectory/event behavior must remain unchanged.

- [ ] **Step 4: Build the C++ package and rerun the contract test**

```bash
colcon build --packages-select my_project_interfaces control_center --symlink-install
python3 -m pytest src/my_project/control_center/test/test_staged_pause_contract.py -q
```

Expected: build succeeds and the contract test passes.

- [ ] **Step 5: Commit the runtime contract change**

```bash
git add src/my_project/my_project_interfaces/msg/TrajectoryPoint.msg src/my_project/control_center/include/control_center/npz_loader.hpp src/my_project/control_center/src/npz_loader.cpp src/my_project/control_center/src/queue_manager.cpp src/my_project/control_center/test/test_staged_pause_contract.py
git commit -m "feat: carry NPZ timing through trajectory queue"
```

### Task 4: Publish low-frequency UI time status

**Files:**
- Modify: `src/my_project/my_project_interfaces/msg/UiStatus.msg`
- Modify: `src/my_project/control_center/src/system_manager_node.cpp`
- Modify: `src/my_project/my_project_startup/launch/startup.launch.py`
- Modify: `src/my_project/my_project_ui/my_project_ui/ui_panel.py`
- Modify: `src/my_project/my_project_ui/test/test_mode_selection_layout.py`
- Modify: `README.md`
- Modify: `src/my_project/my_project_ui/UI_ARCHITECTURE.md`

- [ ] **Step 1: Add failing UI/runtime contract tests**

Add assertions for:

```python
assert "planned_total_time_s" in ui_status_msg
assert "planned_elapsed_time_s" in ui_status_msg
assert "planned_remaining_time_s" in ui_status_msg
assert "bool print_time_valid" in ui_status_msg
assert '"print_time_update_period_ms"' in startup
assert "print_time_update_period_ms_" in system_manager
assert "planned_remaining_time_s" in ui_panel
assert "时间估计" in ui_panel
```

- [ ] **Step 2: Run the UI contract test and verify it fails**

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py -q
```

Expected: FAIL because the UI message fields and display code are absent.

- [ ] **Step 3: Add low-frequency aggregation fields and launch configuration**

Append to `UiStatus.msg`:

```text
float32 planned_total_time_s
float32 planned_elapsed_time_s
float32 planned_remaining_time_s
bool print_time_valid
```

In `SystemManagerNode`:

```cpp
print_time_update_period_ms_ = declare_parameter<int>("print_time_update_period_ms", 500);
```

Cache the last valid elapsed/total/remaining values and their update timestamp. During `publish_ui_status()`, update them only when the period has elapsed and `current_traj_valid && current_traj.planned_time_valid`. Compute:

```cpp
remaining = std::clamp(
  static_cast<double>(current_traj.planned_total_time_s) -
  static_cast<double>(current_traj.planned_time_s),
  0.0, static_cast<double>(current_traj.planned_total_time_s));
```

If the current trajectory is temporarily not aligned, reuse the last valid values. Reset the cache when a new valid run has a lower sequence or a different total duration. Populate `UiStatus` with zeroes and `print_time_valid=false` before the first valid value. This code stays in the existing system-manager timer and never runs in `rsi_node`'s UDP thread.

Add `print_time_update_period_ms` as a launch argument with default `500`, pass it to `system_manager_node`, and add it to `LAUNCH_PARAMS` under `系统管理器` in the UI settings dialog.

- [ ] **Step 4: Add the formal-print display and formatting tests**

Add a pure helper in `ui_panel.py`:

```python
def _format_print_duration(seconds):
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0.0:
        return "--"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
```

Add one label beside the existing formal-print layer progress bar. `_update_print_progress()` must keep the existing layer logic and set this label to either:

```text
总 01:23:45 | 已用 00:12:08 | 剩余约 01:11:37
```

or `时间估计 --` when `msg.print_time_valid` is false. Keep the widget visible only for `_MODE_PAGE_PRINT`; do not show it in test mode. Add tests for duration formatting, formal-mode visibility, and invalid-state text.

- [ ] **Step 5: Run message build and focused UI tests**

```bash
colcon build --packages-select my_project_interfaces control_center my_project_startup my_project_ui --symlink-install
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py -q
```

Expected: build succeeds and focused UI tests pass.

- [ ] **Step 6: Update architecture docs and commit**

Document the new data flow, timing sidecar, ignored event/ABORT scope, `UiStatus` fields, and `print_time_update_period_ms=500` in `README.md` and `src/my_project/my_project_ui/UI_ARCHITECTURE.md`. Do not commit generated NPZ/timing sidecars, build output, or changes to the repository ignore policy.

```bash
git add src/my_project/my_project_interfaces/msg/UiStatus.msg src/my_project/control_center/src/system_manager_node.cpp src/my_project/my_project_startup/launch/startup.launch.py src/my_project/my_project_ui/my_project_ui/ui_panel.py src/my_project/my_project_ui/test/test_mode_selection_layout.py README.md src/my_project/my_project_ui/UI_ARCHITECTURE.md
git commit -m "feat: show RSI print time in formal UI"
```

### Task 5: Full verification and handoff

**Files:**
- Modify only if verification exposes a defect in the files from Tasks 1-4.

- [ ] **Step 1: Run all affected Python tests**

```bash
python3 -m pytest src/my_project/path_processing_core/test src/my_project/gcode_planner/test src/my_project/control_center/test src/my_project/my_project_ui/test -q
```

Expected: zero failures.

- [ ] **Step 2: Build all affected ROS 2 packages**

```bash
colcon build --packages-select my_project_interfaces path_processing_core gcode_planner control_center my_project_startup my_project_ui --symlink-install
```

Expected: exit code 0.

- [ ] **Step 3: Inspect the final diff and repository boundaries**

```bash
git status --short --branch
git diff main...HEAD --stat
git diff main...HEAD --check
```

Expected: only the timing feature, tests, documentation, and the isolated branch commits are present; no generated build/NPZ artifacts are tracked.

- [ ] **Step 4: Run the final focused contract checks**

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py src/my_project/control_center/test/test_staged_pause_contract.py src/my_project/my_project_ui/test/test_mode_selection_layout.py -q
```

Expected: zero failures.

- [ ] **Step 5: Commit any verified fixes if needed**

If verification exposes a defect, add only the exact affected paths from Tasks 1–4 and commit the verified fix with a focused message. If no defect is found, leave the implementation commits unchanged.
