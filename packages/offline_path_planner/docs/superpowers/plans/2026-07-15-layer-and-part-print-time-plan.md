# 当前层与整体零件预计打印时间 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** 在正式打印页增加当前层和整个零件两组独立的进度条、已用时间和预计剩余时间，并保持 RSI 实时线程不受影响。

**Architecture:** 导出阶段由 `RsiTimingAccumulator` 按非事件轨迹行收集每层首行时间，在现有 timing sidecar 中写入层起止边界。`NpzLoader` 读取轻量层表并按 `layer_index` 注入轨迹消息；`system_manager_node` 在现有 UI 定时器中低频计算整体和当前层缓存；RQT 正式打印页上下显示两行进度。

**Tech Stack:** Python dataclasses/NumPy/pytest；ROS 2 IDL；C++17/rclcpp/cnpy；PyQt5/RQT。

---

### Task 1: 为 RSI timing accumulator 增加层边界

**Files:**
- Modify: `src/my_project/path_processing_core/path_processing_core/rsi_timing.py`
- Modify: `src/my_project/path_processing_core/test/test_rsi_timing.py`

- [ ] **Step 1: Write the failing layer-boundary tests**

在现有测试文件中加入：

```python
def test_summary_builds_layer_boundaries_from_trajectory_rows():
    timing = RsiTimingAccumulator(dt=0.1)
    timing.append_trajectory_time(layer_index=0)
    timing.append_trajectory_time(layer_index=0)
    timing.append_event_time()
    timing.append_trajectory_time(layer_index=1)
    timing.append_trajectory_time(layer_index=1)

    summary = timing.summary()

    assert summary["total_planned_time_s"] == 0.3
    assert summary["layers"] == [
        {
            "layer_index": 0,
            "start_time_s": 0.0,
            "end_time_s": 0.2,
            "duration_s": 0.2,
        },
        {
            "layer_index": 1,
            "start_time_s": 0.2,
            "end_time_s": 0.3,
            "duration_s": 0.1,
        },
    ]


def test_events_do_not_create_or_advance_layer_boundaries():
    timing = RsiTimingAccumulator(dt=0.25)
    timing.append_event_time()
    timing.append_event_time()
    timing.append_trajectory_time(layer_index=4)
    summary = timing.summary()

    assert summary["event_rows_ignored"] == 2
    assert summary["layers"] == [{
        "layer_index": 4,
        "start_time_s": 0.0,
        "end_time_s": 0.0,
        "duration_s": 0.0,
    }]
```

- [ ] **Step 2: Run the new tests and verify the red state**

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py -q
```

Expected: the existing three tests pass and the two new tests fail with `TypeError` because `append_trajectory_time()` has no `layer_index` argument.

- [ ] **Step 3: Implement layer collection and summary generation**

Update `RsiTimingAccumulator` with a `_layer_start_times` dictionary. Extend `append_trajectory_time(self, layer_index=None)` so it records the first returned timestamp for each non-`None` layer index. Add this summary helper:

```python
def _layer_summary(self):
    ordered = sorted(self._layer_start_times.items())
    layers = []
    for index, (layer_index, start_time_s) in enumerate(ordered):
        end_time_s = (
            ordered[index + 1][1]
            if index + 1 < len(ordered)
            else self._time_s
        )
        end_time_s = max(float(start_time_s), float(end_time_s))
        layers.append({
            "layer_index": int(layer_index),
            "start_time_s": float(start_time_s),
            "end_time_s": end_time_s,
            "duration_s": end_time_s - float(start_time_s),
        })
    return layers
```

Add `"layers": self._layer_summary()` to `summary()`. Keep `append_event_time()` unchanged so events do not create layer entries, and preserve all existing segment metadata.

- [ ] **Step 4: Run the accumulator tests**

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py -q
```

Expected: all five tests pass.

- [ ] **Step 5: Commit the accumulator change**

```bash
git add src/my_project/path_processing_core/path_processing_core/rsi_timing.py src/my_project/path_processing_core/test/test_rsi_timing.py
git commit -m "feat: add layer timing boundaries"
```

### Task 2: 写出 NPZ sidecar 层表并验证单层/多层输出

**Files:**
- Modify: `src/my_project/path_processing_core/path_processing_core/npz_exporter.py`
- Modify: `src/my_project/gcode_planner/test/test_extrude_reset_payload.py`

- [ ] **Step 1: Add failing exporter layer-sidecar tests**

Add `from dataclasses import replace` to the exporter test imports and add:

```python
def test_export_npz_writes_layer_timing_metadata(tmp_path):
    out = tmp_path / "layers.npz"
    parsed = _commands_with_travel_print_and_event()
    parsed[1] = replace(parsed[1], layer=1)
    export_npz(parsed, str(out), dt=0.1)

    metadata = json.loads(
        (tmp_path / "layers.timing.json").read_text(encoding="utf-8"))
    assert [item["layer_index"] for item in metadata["layers"]] == [0, 1]
    assert metadata["layers"][0]["start_time_s"] == 0.0
    assert metadata["layers"][0]["end_time_s"] == metadata["layers"][1]["start_time_s"]
    assert metadata["layers"][-1]["end_time_s"] == metadata["total_planned_time_s"]
    assert all(item["duration_s"] >= 0.0 for item in metadata["layers"])


def test_export_npz_split_writes_layer_timing_sidecar(tmp_path):
    out = tmp_path / "split.npz"
    parsed = _commands_with_travel_print_and_event()
    parsed[1] = replace(parsed[1], layer=1)
    export_npz(parsed, str(out), dt=0.1, split_by_layer_type=True)

    sidecar = tmp_path / "split" / "split_timing.json"
    assert sidecar.exists()
    assert len(json.loads(sidecar.read_text(encoding="utf-8"))["layers"]) == 2
```

- [ ] **Step 2: Run exporter tests and verify the red state**

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py src/my_project/gcode_planner/test/test_extrude_reset_payload.py -q
```

Expected: the existing tests pass and the new layer tests fail because exporter calls do not pass layer indices into the accumulator yet.

- [ ] **Step 3: Pass layer indices into timing accumulation**

In `_append_sample()`, change the normal row timestamp call to:

```python
planned_time_s = timing.append_trajectory_time(layer_index=layer)
```

In `_append_extrude_wait()`, use the function's `layer` argument in the same call. Leave `_emit_event()` on `timing.append_event_time()` without a layer index. Do not add a new NPZ array; the existing sidecar `timing.summary()` output carries the `layers` list and the existing single-file/split path logic remains unchanged.

- [ ] **Step 4: Run exporter and regression tests**

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py src/my_project/gcode_planner/test/test_extrude_reset_payload.py -q
```

Expected: all accumulator and exporter tests pass, including both sidecar path cases.

- [ ] **Step 5: Commit the sidecar layer metadata**

```bash
git add src/my_project/path_processing_core/path_processing_core/npz_exporter.py src/my_project/gcode_planner/test/test_extrude_reset_payload.py
git commit -m "feat: export per-layer timing metadata"
```

### Task 3: Read layer timing safely through the C++ trajectory queue

**Files:**
- Modify: `src/my_project/control_center/include/control_center/npz_loader.hpp`
- Modify: `src/my_project/control_center/src/npz_loader.cpp`
- Modify: `src/my_project/control_center/src/queue_manager.cpp`
- Modify: `src/my_project/my_project_interfaces/msg/TrajectoryPoint.msg`
- Modify: `src/my_project/control_center/test/test_staged_pause_contract.py`

- [ ] **Step 1: Add failing source-contract assertions**

Extend `test_staged_pause_contract.py` with:

```python
def test_runtime_trajectory_contract_carries_layer_timing():
    msg = _read(TRAJ)
    for field in (
        "planned_layer_start_time_s",
        "planned_layer_total_time_s",
        "bool planned_layer_time_valid",
    ):
        assert field in msg
    header = _read(NPZ_HEADER)
    assert "planned_layer_start_time_s" in header
    assert "planned_layer_total_time_s" in header
    queue = _read(QUEUE)
    assert "tp.planned_layer_time_valid" in queue


def test_npz_loader_parses_layer_timing_entries():
    loader_cpp = _read(NPZ_LOADER_CPP)
    assert "layers" in loader_cpp
    assert "start_time_s" in loader_cpp
    assert "duration_s" in loader_cpp
    assert "layer_timing_" in _read(NPZ_HEADER)
```

- [ ] **Step 2: Run the contract test and verify the red state**

```bash
python3 -m pytest src/my_project/control_center/test/test_staged_pause_contract.py -q
```

Expected: the two new tests fail because the message and loader layer map do not exist.

- [ ] **Step 3: Extend the ROS message and loader row contract**

Append to `TrajectoryPoint.msg`:

```text
float32 planned_layer_start_time_s
float32 planned_layer_total_time_s
bool planned_layer_time_valid
```

Add the matching fields to `NpzRow`:

```cpp
float planned_layer_start_time_s{};
float planned_layer_total_time_s{};
bool planned_layer_time_valid{};
```

Add this loader state:

```cpp
std::unordered_map<uint32_t, std::pair<float, float>> layer_timing_;
bool layer_timing_valid_{false};
```

Implement `load_layer_timing_metadata(const std::string & path)` beside the existing total-time sidecar reader. It must use the same direct/part/manifest sidecar path and scan each object containing `"layer_index"`, `"start_time_s"`, and `"duration_s"`; validate finite non-negative numbers; and store `{start_time_s, duration_s}`. Duplicate layer indices or a malformed entry clear the map and set `layer_timing_valid_ = false`. A missing `layers` array sets only `layer_timing_valid_ = false`; it does not invalidate the existing whole-part `timing_valid_` result.

In `next_row()`, initialize the three fields to zero/false. When `layer_timing_valid_` and the current `layer_index` exists in the map, fill the start and duration and set the row flag true. In `QueueManager::fill()`, copy all three row fields to `TrajectoryPoint`.

- [ ] **Step 4: Build and run the C++ contract test**

```bash
colcon build --packages-select my_project_interfaces control_center --symlink-install
python3 -m pytest src/my_project/control_center/test/test_staged_pause_contract.py -q
```

Expected: both packages build and all source-contract tests pass.

- [ ] **Step 5: Commit the layer trajectory contract**

```bash
git add src/my_project/my_project_interfaces/msg/TrajectoryPoint.msg src/my_project/control_center/include/control_center/npz_loader.hpp src/my_project/control_center/src/npz_loader.cpp src/my_project/control_center/src/queue_manager.cpp src/my_project/control_center/test/test_staged_pause_contract.py
git commit -m "feat: carry layer timing through trajectory queue"
```

### Task 4: Publish two low-frequency timing summaries

**Files:**
- Modify: `src/my_project/my_project_interfaces/msg/UiStatus.msg`
- Modify: `src/my_project/control_center/src/system_manager_node.cpp`
- Modify: `src/my_project/my_project_startup/launch/startup.launch.py`
- Modify: `src/my_project/my_project_ui/test/test_mode_selection_layout.py`

- [ ] **Step 1: Add failing UI/status contract assertions**

Extend the UI test file with:

```python
def test_ui_status_contains_layer_time_summary():
    ui_status = (
        PROJECT_SRC / "my_project_interfaces" / "msg" / "UiStatus.msg"
    ).read_text(encoding="utf-8")
    system = _read(SYSTEM_MANAGER)
    assert "planned_layer_total_time_s" in ui_status
    assert "planned_layer_elapsed_time_s" in ui_status
    assert "planned_layer_remaining_time_s" in ui_status
    assert "bool print_layer_time_valid" in ui_status
    assert "print_layer_last_layer_index_" in system


def test_layer_time_stays_out_of_rsi_node():
    system = _read(SYSTEM_MANAGER)
    rsi = _read(RSI)
    assert "update_print_time_status" in system
    assert "print_layer_remaining_time_s" in system
    assert "planned_layer" not in rsi
```

- [ ] **Step 2: Run UI contract tests and verify the red state**

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py -q
```

Expected: the new assertions fail because no layer fields or cache state exists.

- [ ] **Step 3: Add layer summary fields and cache logic**

Append to `UiStatus.msg`:

```text
float32 planned_layer_total_time_s
float32 planned_layer_elapsed_time_s
float32 planned_layer_remaining_time_s
bool print_layer_time_valid
```

Extend the existing `SystemManagerNode::update_print_time_status()` so each update starts with both summaries invalid. For a valid current trajectory, compute:

```cpp
const float part_total = std::max(0.0F, current.planned_total_time_s);
const float part_elapsed = std::clamp(current.planned_time_s, 0.0F, part_total);
const float layer_total = std::max(0.0F, current.planned_layer_total_time_s);
const float layer_elapsed = std::clamp(
  current.planned_time_s - current.planned_layer_start_time_s,
  0.0F, layer_total);
```

Update both caches immediately on first valid data, when 500 ms has elapsed, when sequence decreases, when part total changes, or when `current.layer_index` differs from `print_layer_last_layer_index_`. Publish cached values on every UI status tick; if no cache exists, publish zeroes and both validity flags false. Keep this method in `system_manager_node.cpp`; do not add it to `rsi_node`.

- [ ] **Step 4: Build and run message/UI tests**

Keep the existing `print_time_update_period_ms=500` declaration and pass-through. Run:

```bash
colcon build --packages-select my_project_interfaces control_center my_project_startup my_project_ui --symlink-install
python3 -m pytest src/my_project/control_center/test/test_staged_pause_contract.py src/my_project/my_project_ui/test/test_mode_selection_layout.py -q
```

Expected: the build succeeds and all timing/UI contract tests pass.

- [ ] **Step 5: Commit the two timing summaries**

```bash
git add src/my_project/my_project_interfaces/msg/UiStatus.msg src/my_project/control_center/src/system_manager_node.cpp src/my_project/my_project_startup/launch/startup.launch.py src/my_project/my_project_ui/test/test_mode_selection_layout.py
git commit -m "feat: publish layer and part time summaries"
```

### Task 5: 在正式打印页显示当前层与整体零件两组进度

**Files:**
- Modify: `src/my_project/my_project_ui/my_project_ui/ui_panel.py`
- Modify: `src/my_project/my_project_ui/test/test_mode_selection_layout.py`
- Modify: `README.md`
- Modify: `docs/UI_ARCHITECTURE.md`

- [ ] **Step 1: Add the failing UI layout contract test**

在现有 UI 源码契约测试中加入：

```python
def test_formal_print_has_layer_and_part_progress_rows():
    src = _source()
    assert "_print_progress_bar" in src
    assert "_print_total_progress_bar" in src
    assert "_print_layer_time_label" in src
    assert "_print_time_label" in src
    assert "当前层" in src
    assert "整体" in src
    assert "print_layer_time_valid" in src
```

- [ ] **Step 2: Run the UI test and verify the red state**

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py -q
```

Expected: the new assertion fails because the formal-print widget currently has only one progress row and no layer-time label.

- [ ] **Step 3: Implement the two-row formal-print widget**

将现有 `_print_progress_widget` 改为垂直布局，保留已有的正式打印显示/隐藏条件和样式，并放入两行：

1. 第一行使用 `_print_progress_label`、`_print_progress_bar`、`_print_layer_time_label`，分别显示 `当前层 xx / yy`、当前层进度条和 `本层剩余 HH:MM:SS`。
2. 第二行新增 `_print_total_progress_label`、`_print_total_progress_bar`，并复用 `_print_time_label`，分别显示 `整体 xx%`、整个零件进度条和 `整体剩余 HH:MM:SS`。

两条进度条都使用 0–100 的整数范围；当前层行继续使用现有层索引/层总数计算。新增整体行的更新逻辑放在 `_update_print_time()` 中，使用 `UiStatus` 的低频字段：

```python
def _update_print_time(self, msg: UiStatus):
    if not msg.print_layer_time_valid:
        self._print_layer_time_label.setText("本层剩余 --")
    else:
        self._print_layer_time_label.setText(
            "本层剩余 " + _format_print_duration(msg.planned_layer_remaining_time_s)
        )

    if not msg.print_time_valid:
        self._print_time_label.setText("整体剩余 --")
        self._print_total_progress_bar.setValue(0)
        self._print_total_progress_label.setText("整体 --")
        return

    total = max(0.0, float(msg.planned_total_time_s))
    elapsed = max(0.0, min(float(msg.planned_elapsed_time_s), total))
    progress = int(round(elapsed / total * 100.0)) if total > 0.0 else 100
    self._print_total_progress_bar.setValue(progress)
    self._print_total_progress_label.setText(f"整体 {progress}%")
    self._print_time_label.setText(
        "整体剩余 " + _format_print_duration(msg.planned_remaining_time_s)
    )
```

在 `_update_print_progress()` 中继续先处理无当前轨迹的重置，再调用 `_update_print_time(msg)`；当正式打印状态结束或没有有效时间时，两条时间显示和整体进度条都恢复到占位状态。此更新只发生在现有 UI 状态回调线程，不把时间计算放进 RSI 回调或实时控制循环。

- [ ] **Step 4: Run the UI tests and syntax check**

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py -q
python3 -m py_compile src/my_project/my_project_ui/my_project_ui/ui_panel.py
```

Expected: the layout contract and existing UI tests pass, and `py_compile` exits successfully.

- [ ] **Step 5: Document the two display meanings**

在 `README.md` 和 `docs/UI_ARCHITECTURE.md` 中说明：正式打印页第一行是当前层进度/当前层剩余时间，第二行是整个零件进度/整体剩余时间；层边界来自导出时生成的 timing sidecar，跨层空走归入前一层计划时间；事件等待、ABORT 和打印头事件不参与 RSI 预计时间；UI 默认以 500 ms 低频刷新，不影响 RSI 实时线程。明确“最后一层结束时间”表示计划总时间终点，而不是单独再增加一段等待时间。

- [ ] **Step 6: Commit the formal-print display**

```bash
git add src/my_project/my_project_ui/my_project_ui/ui_panel.py src/my_project/my_project_ui/test/test_mode_selection_layout.py README.md docs/UI_ARCHITECTURE.md
git commit -m "feat: show layer and part print estimates"
```

### Task 6: 完成跨包验证并检查仓库边界

**Files:**
- No source changes expected; only add a narrowly scoped verification fix if a test exposes a real implementation defect.

- [ ] **Step 1: Run focused timing, exporter, contract, and UI tests**

```bash
python3 -m pytest src/my_project/path_processing_core/test/test_rsi_timing.py src/my_project/gcode_planner/test/test_extrude_reset_payload.py src/my_project/control_center/test/test_staged_pause_contract.py src/my_project/my_project_ui/test/test_mode_selection_layout.py -q
```

Expected: all focused timing and display tests pass.

- [ ] **Step 2: Run the affected package regression suites**

```bash
python3 -m pytest src/my_project/path_processing_core/test -q -k 'not flake8 and not pep257'
python3 -m pytest src/my_project/gcode_planner/test -q -k 'not flake8 and not pep257'
python3 -m pytest src/my_project/control_center/test src/my_project/my_project_ui/test -q -k 'not flake8 and not pep257'
```

Expected: the functional suites pass. Repository-wide `flake8`/`pep257` scans remain excluded because the existing workspace test configuration also scans generated build/install artifacts and unrelated third-party sources.

- [ ] **Step 3: Build all affected ROS packages**

```bash
colcon build --packages-select path_processing_core gcode_planner my_project_interfaces control_center my_project_startup my_project_ui --symlink-install
```

Expected: all six affected packages build successfully.

- [ ] **Step 4: Inspect the final diff and repository scope**

```bash
git status --short
git diff main...HEAD --check
git diff main...HEAD --stat
git diff --name-only main...HEAD
```

Confirm that only the timing implementation, tests, message definitions, launch/UI wiring, and the two explicitly tracked design/plan documents changed; do not add generated `build/`, `install/`, `log/`, NPZ files, or `.gitignore` changes.

- [ ] **Step 5: Commit only a verified follow-up fix, if required**

If verification finds an implementation defect, add a regression test first, apply the smallest fix, rerun the relevant command, and commit it with a message describing the defect. Otherwise leave the implementation commits unchanged and report the exact verification results.
