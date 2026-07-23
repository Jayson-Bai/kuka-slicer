# 测试模式纤维打印 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在测试模式中加入树脂/纤维标定、纤维单独打印、复合打印和剪切按钮，并只调整正式打印的偏置 UI 布局。

**Architecture:** 保持现有测试模式架构：UI 生成临时 GCode/NPZ，通过 `/print_test/rsi_command` reset，再通过 `/print_test/load_npz` 下发给 `center_node`。新增标定 JSON helper 负责树脂/纤维补偿持久化；新增 GCode 生成 helper 负责纤维矩阵和复合矩阵，运行时实时链路不做队列插入。

**Tech Stack:** ROS 2 Python UI (`rqt_gui_py`, `python_qt_binding`), ROS 2 C++ nodes, `gcode_planner` Python GCode/NPZ exporter, pytest, colcon.

---

## 文件结构

- Create: `src/my_project/gcode_planner/gcode_planner/head_calibration.py`
  - 负责 `/home/jayson/kuka_ram_ws/data/head_calibration_offsets/head_offsets.json` 的默认值、加载、保存、相对补偿计算。
- Modify: `src/my_project/gcode_planner/gcode_planner/print_test_generator.py`
  - 增加纤维工具号、可配置线长/Y 间距、单喷头矩阵生成、复合矩阵生成、tool change 安全序列生成。
- Modify: `src/my_project/gcode_planner/test/test_print_test_generator.py`
  - 覆盖纤维矩阵、复合矩阵、补偿方向、安全抬升、工具号。
- Create: `src/my_project/gcode_planner/test/test_head_calibration.py`
  - 覆盖 JSON schema、默认值、时间戳、覆盖写入、相对补偿。
- Modify: `src/my_project/my_project_ui/my_project_ui/ui_panel.py`
  - 重组测试模式 UI、状态、按钮、参数解析、标定保存、纤维/复合 job 生成、剪切校验、正式打印偏置 UI 布局。
- Modify: `src/my_project/my_project_ui/test/test_mode_selection_layout.py`
  - 增加静态 UI/流程约束测试，确保按钮、JSON 路径、正式打印布局存在且正式打印不接入新导出补偿。
- Modify: `src/my_project/my_project_ui/test/test_print_test_matrix_export.py`
  - 增加 UI 生成临时 NPZ 时使用新 fiber/composite 参数的静态约束测试。
- Modify: `src/my_project/my_project_ui/UI_ARCHITECTURE.md`
  - 同步测试模式和正式打印偏置 UI 文档。

---

### Task 1: Head Calibration Helper

**Files:**
- Create: `src/my_project/gcode_planner/gcode_planner/head_calibration.py`
- Create: `src/my_project/gcode_planner/test/test_head_calibration.py`

- [ ] **Step 1: Write failing calibration tests**

Create `src/my_project/gcode_planner/test/test_head_calibration.py`:

```python
import json
from datetime import datetime

from gcode_planner.head_calibration import (
    DEFAULT_HEAD_CALIBRATION,
    HeadCalibration,
    calibration_relative_offsets,
    load_head_calibration,
    save_head_calibration,
)


def test_load_head_calibration_returns_defaults_when_file_missing(tmp_path):
    path = tmp_path / "head_offsets.json"

    cal = load_head_calibration(path)

    assert cal.resin_z_print_compensation_mm == 0.0
    assert cal.fiber_x_print_compensation_mm == 0.0
    assert cal.fiber_y_print_compensation_mm == 0.0
    assert cal.fiber_z_print_compensation_mm == 0.0
    assert DEFAULT_HEAD_CALIBRATION["resin"]["z_print_compensation_mm"] == 0.0


def test_save_head_calibration_overwrites_current_file_with_timestamp(tmp_path):
    path = tmp_path / "head_offsets.json"
    save_head_calibration(
        path,
        HeadCalibration(
            resin_z_print_compensation_mm=-20.0,
            fiber_x_print_compensation_mm=5.0,
            fiber_y_print_compensation_mm=4.0,
            fiber_z_print_compensation_mm=-25.0,
        ),
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert datetime.fromisoformat(data["updated_at"])
    assert data["resin"] == {"z_print_compensation_mm": -20.0}
    assert data["fiber"] == {
        "x_print_compensation_mm": 5.0,
        "y_print_compensation_mm": 4.0,
        "z_print_compensation_mm": -25.0,
    }

    save_head_calibration(path, HeadCalibration(resin_z_print_compensation_mm=-1.0))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["resin"]["z_print_compensation_mm"] == -1.0
    assert data["fiber"]["x_print_compensation_mm"] == 0.0


def test_calibration_relative_offsets_use_target_minus_current_head():
    cal = HeadCalibration(
        resin_z_print_compensation_mm=-20.0,
        fiber_x_print_compensation_mm=5.0,
        fiber_y_print_compensation_mm=4.0,
        fiber_z_print_compensation_mm=-25.0,
    )

    assert calibration_relative_offsets(cal, from_tool="resin", to_tool="fiber") == (
        5.0,
        4.0,
        -5.0,
    )
    assert calibration_relative_offsets(cal, from_tool="fiber", to_tool="resin") == (
        -5.0,
        -4.0,
        5.0,
    )
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python3 -m pytest src/my_project/gcode_planner/test/test_head_calibration.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gcode_planner.head_calibration'`.

- [ ] **Step 3: Implement calibration helper**

Create `src/my_project/gcode_planner/gcode_planner/head_calibration.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping


DEFAULT_HEAD_CALIBRATION_PATH = Path(
    "/home/jayson/kuka_ram_ws/data/head_calibration_offsets/head_offsets.json"
)

DEFAULT_HEAD_CALIBRATION = {
    "resin": {"z_print_compensation_mm": 0.0},
    "fiber": {
        "x_print_compensation_mm": 0.0,
        "y_print_compensation_mm": 0.0,
        "z_print_compensation_mm": 0.0,
    },
}


@dataclass(frozen=True)
class HeadCalibration:
    resin_z_print_compensation_mm: float = 0.0
    fiber_x_print_compensation_mm: float = 0.0
    fiber_y_print_compensation_mm: float = 0.0
    fiber_z_print_compensation_mm: float = 0.0


def _as_float(data: Mapping[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_head_calibration(path: str | Path = DEFAULT_HEAD_CALIBRATION_PATH) -> HeadCalibration:
    p = Path(path)
    if not p.is_file():
        return HeadCalibration()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return HeadCalibration()
    resin = data.get("resin", {}) if isinstance(data, dict) else {}
    fiber = data.get("fiber", {}) if isinstance(data, dict) else {}
    if not isinstance(resin, dict):
        resin = {}
    if not isinstance(fiber, dict):
        fiber = {}
    return HeadCalibration(
        resin_z_print_compensation_mm=_as_float(
            resin, "z_print_compensation_mm", 0.0
        ),
        fiber_x_print_compensation_mm=_as_float(
            fiber, "x_print_compensation_mm", 0.0
        ),
        fiber_y_print_compensation_mm=_as_float(
            fiber, "y_print_compensation_mm", 0.0
        ),
        fiber_z_print_compensation_mm=_as_float(
            fiber, "z_print_compensation_mm", 0.0
        ),
    )


def save_head_calibration(
    path: str | Path,
    calibration: HeadCalibration,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "resin": {
            "z_print_compensation_mm": float(
                calibration.resin_z_print_compensation_mm
            )
        },
        "fiber": {
            "x_print_compensation_mm": float(
                calibration.fiber_x_print_compensation_mm
            ),
            "y_print_compensation_mm": float(
                calibration.fiber_y_print_compensation_mm
            ),
            "z_print_compensation_mm": float(
                calibration.fiber_z_print_compensation_mm
            ),
        },
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def calibration_relative_offsets(
    calibration: HeadCalibration,
    *,
    from_tool: str,
    to_tool: str,
) -> tuple[float, float, float]:
    resin = (0.0, 0.0, float(calibration.resin_z_print_compensation_mm))
    fiber = (
        float(calibration.fiber_x_print_compensation_mm),
        float(calibration.fiber_y_print_compensation_mm),
        float(calibration.fiber_z_print_compensation_mm),
    )
    tools = {"resin": resin, "fiber": fiber}
    if from_tool not in tools or to_tool not in tools:
        raise ValueError("from_tool and to_tool must be 'resin' or 'fiber'")
    src = tools[from_tool]
    dst = tools[to_tool]
    return tuple(dst[i] - src[i] for i in range(3))
```

- [ ] **Step 4: Run calibration tests**

Run:

```bash
python3 -m pytest src/my_project/gcode_planner/test/test_head_calibration.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/my_project/gcode_planner/gcode_planner/head_calibration.py src/my_project/gcode_planner/test/test_head_calibration.py
git commit -m "feat: add head calibration persistence"
```

---

### Task 2: Fiber and Composite GCode Generation

**Files:**
- Modify: `src/my_project/gcode_planner/gcode_planner/print_test_generator.py`
- Modify: `src/my_project/gcode_planner/test/test_print_test_generator.py`

- [ ] **Step 1: Add failing generator tests**

Append to `src/my_project/gcode_planner/test/test_print_test_generator.py`:

```python
from gcode_planner.head_calibration import HeadCalibration
from gcode_planner.print_test_generator import (
    FIBER_TOOL_ID,
    generate_composite_test_matrix_gcode,
    generate_head_test_matrix_gcode,
    generate_pose_adjust_gcode,
)


def test_pose_adjust_gcode_moves_xyzabc_without_extrusion():
    lines = generate_pose_adjust_gcode(
        start_pose=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        target_pose=(7.0, 8.0, 9.0, 4.0, 5.0, 6.0),
        speed_mm_s=5.0,
    )

    moves = _moves(lines)

    assert len(moves) == 1
    assert moves[0].start_pos.x == 1.0
    assert moves[0].pos.x == 7.0
    assert moves[0].pos.y == 8.0
    assert moves[0].pos.z == 9.0
    assert moves[0].delta_e == 0.0


def test_fiber_matrix_uses_fiber_tool_and_serpentine_geometry():
    lines = generate_head_test_matrix_gcode(
        start_pose=(1.0, 2.0, 0.4, 0.0, 0.0, 0.0),
        tool="fiber",
        layer_heights_mm=[0.5],
        extrusion_scales=[0.8, 1.0],
        speed_mm_s=10.0,
        line_length_mm=300.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        prime_length_mm=5.0,
        retract_length_mm=3.0,
        prime_speed_mm_s=2.0,
        retract_speed_mm_s=8.0,
    )

    parsed = parse_gcode_lines(lines)
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    print_moves = [cmd for cmd in _moves(lines) if cmd.type == "PRINT"]

    assert FIBER_TOOL_ID == 1
    assert tools[0].tool_id == FIBER_TOOL_ID
    assert [(cmd.start_pos.x, cmd.start_pos.y) for cmd in print_moves] == [
        (1.0, 2.0),
        (301.0, 12.0),
    ]
    assert [cmd.pos.x for cmd in print_moves] == [301.0, 1.0]


def test_composite_matrix_inserts_safe_lift_compensation_and_tool_change():
    lines = generate_composite_test_matrix_gcode(
        start_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0],
        fiber_layer_heights_mm=[0.6],
        fiber_extrusion_scales=[1.1],
        speed_mm_s=10.0,
        line_length_mm=300.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        prime_length_mm=0.0,
        retract_length_mm=0.0,
        prime_speed_mm_s=2.0,
        retract_speed_mm_s=8.0,
        fiber_prime_length_mm=0.0,
        fiber_retract_length_mm=0.0,
        fiber_prime_speed_mm_s=2.0,
        fiber_retract_speed_mm_s=8.0,
        calibration=HeadCalibration(
            resin_z_print_compensation_mm=-20.0,
            fiber_x_print_compensation_mm=5.0,
            fiber_y_print_compensation_mm=4.0,
            fiber_z_print_compensation_mm=-25.0,
        ),
        tool_change_safe_lift_mm=10.0,
    )

    text = "\n".join(lines)

    assert ";TOOL_CHANGE_SAFE_LIFT:10.000000" in text
    assert ";TOOL_CHANGE_COMPENSATION:5.000000,4.000000,-5.000000" in text
    assert "T0" in lines
    assert lines.index("T0") > next(
        i for i, line in enumerate(lines) if line.startswith(";TOOL_CHANGE_COMPENSATION")
    )
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest src/my_project/gcode_planner/test/test_print_test_generator.py -q
```

Expected: FAIL with missing `FIBER_TOOL_ID` or missing generator functions.

- [ ] **Step 3: Implement generator helpers**

Modify `src/my_project/gcode_planner/gcode_planner/print_test_generator.py`:

```python
from gcode_planner.head_calibration import (
    HeadCalibration,
    calibration_relative_offsets,
)

FIBER_TOOL_ID = 1
FIBER_GCODE_TOOL = 0
```

Add helpers after `_header`:

```python
def _tool_header(start_pose: Sequence[float], *, tool: str) -> list[str]:
    lines = _header(start_pose)
    if tool == "fiber":
        lines.insert(4, f"T{FIBER_GCODE_TOOL}")
    elif tool == "resin":
        lines.insert(4, f"T{RESIN_GCODE_TOOL}")
    else:
        raise ValueError("tool must be 'resin' or 'fiber'")
    return lines
```

Add this arbitrary pose travel helper after `generate_z_adjust_gcode`:

```python
def generate_pose_adjust_gcode(
    *,
    start_pose: Sequence[float],
    target_pose: Sequence[float],
    speed_mm_s: float,
) -> list[str]:
    _pose_values(start_pose)
    x, y, z, a, b, c = _pose_values(target_pose)
    feed = _feed(speed_mm_s)
    lines = _header(start_pose, reset_e=False)
    lines.append(
        f"G1 X{x:.6f} Y{y:.6f} Z{z:.6f} "
        f"A{a:.6f} B{b:.6f} C{c:.6f} F{feed:.3f}"
    )
    return lines
```

Add public wrappers after `generate_test_matrix_gcode`:

```python
def generate_head_test_matrix_gcode(
    *,
    start_pose: Sequence[float],
    tool: str,
    layer_heights_mm: Sequence[float],
    extrusion_scales: Sequence[float],
    speed_mm_s: float,
    line_length_mm: float = 300.0,
    y_spacing_mm: float = 10.0,
    finish_lift_mm: float = 10.0,
    max_lines: int = TEST_MATRIX_MAX_LINES,
    prime_length_mm: float = 0.0,
    retract_length_mm: float = 0.0,
    prime_speed_mm_s: float = 2.0,
    retract_speed_mm_s: float = 8.0,
) -> list[str]:
    lines = generate_test_matrix_gcode(
        start_pose=start_pose,
        layer_heights_mm=layer_heights_mm,
        extrusion_scales=extrusion_scales,
        speed_mm_s=speed_mm_s,
        line_length_mm=line_length_mm,
        y_spacing_mm=y_spacing_mm,
        finish_lift_mm=finish_lift_mm,
        max_lines=max_lines,
        prime_length_mm=prime_length_mm,
        retract_length_mm=retract_length_mm,
        prime_speed_mm_s=prime_speed_mm_s,
        retract_speed_mm_s=retract_speed_mm_s,
    )
    tool_line = f"T{FIBER_GCODE_TOOL}" if tool == "fiber" else f"T{RESIN_GCODE_TOOL}"
    if tool not in ("fiber", "resin"):
        raise ValueError("tool must be 'resin' or 'fiber'")
    return lines[:4] + [tool_line] + lines[4:]


def _append_tool_change_compensation(
    lines: list[str],
    *,
    current_pose: tuple[float, float, float, float, float, float],
    delta_xyz: tuple[float, float, float],
    speed_mm_s: float,
    safe_lift_mm: float,
) -> tuple[float, float, float, float, float, float]:
    x, y, z, a, b, c = current_pose
    feed = _feed(speed_mm_s)
    safe_z = z + float(safe_lift_mm)
    lines.append(f";TOOL_CHANGE_SAFE_LIFT:{float(safe_lift_mm):.6f}")
    lines.append(
        f"G0 X{x:.6f} Y{y:.6f} Z{safe_z:.6f} A{a:.6f} B{b:.6f} C{c:.6f} F{feed:.3f}"
    )
    dx, dy, dz = delta_xyz
    target = (x + dx, y + dy, safe_z + dz, a, b, c)
    lines.append(f";TOOL_CHANGE_COMPENSATION:{dx:.6f},{dy:.6f},{dz:.6f}")
    lines.append(
        f"G0 X{target[0]:.6f} Y{target[1]:.6f} Z{target[2]:.6f} "
        f"A{a:.6f} B{b:.6f} C{c:.6f} F{feed:.3f}"
    )
    return target


def _matrix_final_pose(
    start_pose: Sequence[float], line_count: int, line_length_mm: float,
                       y_spacing_mm: float, last_layer_height: float,
                       finish_lift_mm: float) -> tuple[float, float, float, float, float, float]:
    x, y, z, a, b, c = _pose_values(start_pose)
    final_x = x + float(line_length_mm) if line_count % 2 == 1 else x
    final_y = y + max(0, line_count - 1) * float(y_spacing_mm)
    final_z = z + float(last_layer_height) + float(finish_lift_mm)
    return (final_x, final_y, final_z, a, b, c)


def generate_composite_test_matrix_gcode(
    *,
    start_pose: Sequence[float],
    resin_layer_heights_mm: Sequence[float],
    resin_extrusion_scales: Sequence[float],
    fiber_layer_heights_mm: Sequence[float],
    fiber_extrusion_scales: Sequence[float],
    speed_mm_s: float,
    line_length_mm: float,
    y_spacing_mm: float,
    finish_lift_mm: float,
    prime_length_mm: float,
    retract_length_mm: float,
    prime_speed_mm_s: float,
    retract_speed_mm_s: float,
    fiber_prime_length_mm: float,
    fiber_retract_length_mm: float,
    fiber_prime_speed_mm_s: float,
    fiber_retract_speed_mm_s: float,
    calibration: HeadCalibration,
    tool_change_safe_lift_mm: float = 10.0,
) -> list[str]:
    resin_lines = generate_head_test_matrix_gcode(
        start_pose=start_pose,
        tool="resin",
        layer_heights_mm=resin_layer_heights_mm,
        extrusion_scales=resin_extrusion_scales,
        speed_mm_s=speed_mm_s,
        line_length_mm=line_length_mm,
        y_spacing_mm=y_spacing_mm,
        finish_lift_mm=finish_lift_mm,
        prime_length_mm=prime_length_mm,
        retract_length_mm=retract_length_mm,
        prime_speed_mm_s=prime_speed_mm_s,
        retract_speed_mm_s=retract_speed_mm_s,
    )
    line_count = len(resin_layer_heights_mm) * len(resin_extrusion_scales)
    current_pose = _matrix_final_pose(
        start_pose,
        line_count,
        line_length_mm,
        y_spacing_mm,
        resin_layer_heights_mm[-1],
        finish_lift_mm,
    )
    delta = calibration_relative_offsets(
        calibration, from_tool="resin", to_tool="fiber"
    )
    _append_tool_change_compensation(
        resin_lines,
        current_pose=current_pose,
        delta_xyz=delta,
        speed_mm_s=speed_mm_s,
        safe_lift_mm=tool_change_safe_lift_mm,
    )
    resin_lines.append(f"T{FIBER_GCODE_TOOL}")
    fiber_start = (
        _pose_values(start_pose)[0] + delta[0],
        _pose_values(start_pose)[1] + delta[1],
        _pose_values(start_pose)[2] + delta[2],
        _pose_values(start_pose)[3],
        _pose_values(start_pose)[4],
        _pose_values(start_pose)[5],
    )
    fiber_lines = generate_head_test_matrix_gcode(
        start_pose=fiber_start,
        tool="fiber",
        layer_heights_mm=fiber_layer_heights_mm,
        extrusion_scales=fiber_extrusion_scales,
        speed_mm_s=speed_mm_s,
        line_length_mm=line_length_mm,
        y_spacing_mm=y_spacing_mm,
        finish_lift_mm=finish_lift_mm,
        prime_length_mm=fiber_prime_length_mm,
        retract_length_mm=fiber_retract_length_mm,
        prime_speed_mm_s=fiber_prime_speed_mm_s,
        retract_speed_mm_s=fiber_retract_speed_mm_s,
    )
    return resin_lines + fiber_lines[5:]
```

- [ ] **Step 4: Run generator tests**

Run:

```bash
python3 -m pytest src/my_project/gcode_planner/test/test_print_test_generator.py src/my_project/gcode_planner/test/test_head_calibration.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/my_project/gcode_planner/gcode_planner/print_test_generator.py src/my_project/gcode_planner/test/test_print_test_generator.py
git commit -m "feat: generate fiber test matrices"
```

---

### Task 3: Static UI Contract Tests

**Files:**
- Modify: `src/my_project/my_project_ui/test/test_mode_selection_layout.py`
- Modify: `src/my_project/my_project_ui/test/test_print_test_matrix_export.py`

- [ ] **Step 1: Add failing UI contract tests**

Append to `src/my_project/my_project_ui/test/test_mode_selection_layout.py`:

```python

def test_test_mode_exposes_fiber_calibration_and_print_actions():
    src = _source()
    test_section = src.split("# ======== Print Test 区域 ========", 1)[1].split(
        "# ======== Launch Control 区域 ========", 1
    )[0]

    for text in (
        "确认树脂打印高度",
        "继续调整纤维头",
        "开始测试树脂打印",
        "应用纤维偏置",
        "确认纤维头偏置",
        "直接打印纤维",
        "复合打印",
        "剪切",
    ):
        assert text in test_section

    for attr in (
        "_test_resin_z_comp_input",
        "_test_fiber_x_comp_input",
        "_test_fiber_y_comp_input",
        "_test_fiber_z_comp_input",
        "_test_fiber_temp_input",
        "_test_line_length_input",
        "_test_y_spacing_input",
        "_test_tool_change_safe_lift_input",
    ):
        assert attr in test_section


def test_scissor_button_checks_fiber_tool_before_uart_command():
    src = _source()
    block = src.split("    def _on_print_test_cut", 1)[1].split(
        "    def _on_print_test_prepare", 1
    )[0]

    assert "current_tool_id() != 1" in block
    assert "EV 0 cut_cf\\n" in block
    assert "self.uart_command_submit.emit" in block
```

Append to `src/my_project/my_project_ui/test/test_print_test_matrix_export.py`:

```python

def test_print_test_job_generation_supports_resin_fiber_and_composite_modes():
    src = UI_PANEL.read_text(encoding="utf-8")
    block = src.split("    def _run_print_test_job", 1)[1].split(
        "    # ---- Offset persistence ----", 1
    )[0]

    assert 'job_type == "resin_matrix"' in block
    assert 'job_type == "fiber_matrix"' in block
    assert 'job_type == "composite_matrix"' in block
    assert "generate_head_test_matrix_gcode" in block
    assert "generate_composite_test_matrix_gcode" in block
    assert "save_head_calibration" in block
```

- [ ] **Step 2: Run UI contract tests to verify failure**

Run:

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py::test_test_mode_exposes_fiber_calibration_and_print_actions src/my_project/my_project_ui/test/test_mode_selection_layout.py::test_scissor_button_checks_fiber_tool_before_uart_command src/my_project/my_project_ui/test/test_print_test_matrix_export.py::test_print_test_job_generation_supports_resin_fiber_and_composite_modes -q
```

Expected: FAIL because UI controls and handlers do not exist yet.

- [ ] **Step 3: Commit tests only**

```bash
git add src/my_project/my_project_ui/test/test_mode_selection_layout.py src/my_project/my_project_ui/test/test_print_test_matrix_export.py
git commit -m "test: define fiber test mode ui contract"
```

---

### Task 4: UI Calibration State and Layout

**Files:**
- Modify: `src/my_project/my_project_ui/my_project_ui/ui_panel.py`

- [ ] **Step 1: Add imports and constants**

In `src/my_project/my_project_ui/my_project_ui/ui_panel.py`, add near other planner imports/constants:

```python
from gcode_planner.head_calibration import (
    DEFAULT_HEAD_CALIBRATION_PATH,
    HeadCalibration,
    load_head_calibration,
    save_head_calibration,
)

_TEST_TOOL_CHANGE_SAFE_LIFT_DEFAULT_MM = 10.0
_PRINT_TEST_ZERO_CORRECTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 2: Add widget state fields in `_UiStatusWidget.__init__`**

Add after existing `_print_test_params` fields:

```python
self._head_calibration = load_head_calibration()
self._print_test_resin_height_confirmed = False
self._print_test_fiber_confirmed = False
self._print_test_waiting_for_tool = None
self._print_test_pending_after_zero = None
self._print_test_requested_target_tool = None
```

- [ ] **Step 3: Replace print test form fields with grouped controls**

In `_build_ui`, inside the print test section, keep existing parameter names for resin compatibility and add these new inputs:

```python
self._test_line_length_input = QtWidgets.QLineEdit("300.0")
self._test_y_spacing_input = QtWidgets.QLineEdit("10.0")
self._test_tool_change_safe_lift_input = QtWidgets.QLineEdit("10.0")
self._test_resin_z_comp_input = QtWidgets.QLineEdit(
    f"{self._head_calibration.resin_z_print_compensation_mm:.3f}"
)
self._test_fiber_temp_input = QtWidgets.QLineEdit("250")
self._test_fiber_layer_height_min_input = QtWidgets.QLineEdit("0.5")
self._test_fiber_layer_height_max_input = QtWidgets.QLineEdit("1.0")
self._test_fiber_scale_min_input = QtWidgets.QLineEdit("0.8")
self._test_fiber_scale_max_input = QtWidgets.QLineEdit("1.2")
self._test_fiber_prime_length_input = QtWidgets.QLineEdit("5.0")
self._test_fiber_prime_speed_input = QtWidgets.QLineEdit("2.0")
self._test_fiber_retract_length_input = QtWidgets.QLineEdit("3.0")
self._test_fiber_retract_speed_input = QtWidgets.QLineEdit("8.0")
self._test_fiber_x_comp_input = QtWidgets.QLineEdit(
    f"{self._head_calibration.fiber_x_print_compensation_mm:.3f}"
)
self._test_fiber_y_comp_input = QtWidgets.QLineEdit(
    f"{self._head_calibration.fiber_y_print_compensation_mm:.3f}"
)
self._test_fiber_z_comp_input = QtWidgets.QLineEdit(
    f"{self._head_calibration.fiber_z_print_compensation_mm:.3f}"
)
```

Add buttons:

```python
self._btn_test_confirm_resin_height = QtWidgets.QPushButton("确认树脂打印高度")
self._btn_test_continue_fiber = QtWidgets.QPushButton("继续调整纤维头")
self._btn_test_print_resin = QtWidgets.QPushButton("开始测试树脂打印")
self._btn_test_apply_fiber_offset = QtWidgets.QPushButton("应用纤维偏置")
self._btn_test_confirm_fiber_offset = QtWidgets.QPushButton("确认纤维头偏置")
self._btn_test_print_fiber = QtWidgets.QPushButton("直接打印纤维")
self._btn_test_print_composite = QtWidgets.QPushButton("复合打印")
self._btn_test_cut = QtWidgets.QPushButton("剪切")
```

Connect them:

```python
self._btn_test_confirm_resin_height.clicked.connect(self._on_print_test_confirm_resin_height)
self._btn_test_continue_fiber.clicked.connect(self._on_print_test_continue_fiber)
self._btn_test_print_resin.clicked.connect(self._on_print_test_print_resin)
self._btn_test_apply_fiber_offset.clicked.connect(self._on_print_test_apply_fiber_offset)
self._btn_test_confirm_fiber_offset.clicked.connect(self._on_print_test_confirm_fiber_offset)
self._btn_test_print_fiber.clicked.connect(self._on_print_test_print_fiber)
self._btn_test_print_composite.clicked.connect(self._on_print_test_print_composite)
self._btn_test_cut.clicked.connect(self._on_print_test_cut)
```

- [ ] **Step 4: Update control enabling**

Replace `_set_print_test_controls_enabled` body with explicit state gates:

```python
def _set_print_test_controls_enabled(self, enabled):
    base_ready = bool(enabled and not self._print_test_busy)
    for btn in getattr(self, "_test_z_buttons", []):
        btn.setEnabled(base_ready)
    self._btn_test_confirm_resin_height.setEnabled(base_ready)
    self._btn_test_print_resin.setEnabled(base_ready and self._print_test_resin_height_confirmed)
    self._btn_test_continue_fiber.setEnabled(base_ready and self._print_test_resin_height_confirmed)
    fiber_ready = base_ready and self.current_tool_id() == 1
    self._btn_test_apply_fiber_offset.setEnabled(fiber_ready)
    self._btn_test_confirm_fiber_offset.setEnabled(fiber_ready)
    self._btn_test_print_fiber.setEnabled(fiber_ready and self._print_test_fiber_confirmed)
    self._btn_test_print_composite.setEnabled(fiber_ready and self._print_test_fiber_confirmed)
    self._btn_test_cut.setEnabled(base_ready)
```

- [ ] **Step 5: Run UI contract tests**

Run:

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py::test_test_mode_exposes_fiber_calibration_and_print_actions -q
```

Expected: PASS for the layout contract test.

- [ ] **Step 6: Commit layout/state skeleton**

```bash
git add src/my_project/my_project_ui/my_project_ui/ui_panel.py
git commit -m "feat: add fiber test mode controls"
```

---

### Task 5: UI Calibration Actions and Job Generation

**Files:**
- Modify: `src/my_project/my_project_ui/my_project_ui/ui_panel.py`

- [ ] **Step 1: Add parameter parsing helpers**

Add methods before `_parse_print_test_params`:

```python
def _float_input(self, widget, label, *, minimum=None, allow_zero=True):
    text = widget.text().strip()
    if not text:
        raise ValueError(f"{label}不能为空。")
    value = float(text)
    if minimum is not None:
        if allow_zero:
            invalid = value < minimum
        else:
            invalid = value <= minimum
        if invalid:
            raise ValueError(f"{label}超出范围。")
    return value


def _current_head_calibration_from_inputs(self):
    return HeadCalibration(
        resin_z_print_compensation_mm=self._float_input(
            self._test_resin_z_comp_input, "树脂Z打印补偿"
        ),
        fiber_x_print_compensation_mm=self._float_input(
            self._test_fiber_x_comp_input, "纤维X打印补偿"
        ),
        fiber_y_print_compensation_mm=self._float_input(
            self._test_fiber_y_comp_input, "纤维Y打印补偿"
        ),
        fiber_z_print_compensation_mm=self._float_input(
            self._test_fiber_z_comp_input, "纤维Z打印补偿"
        ),
    )
```

Modify `_parse_print_test_params` so it returns nested params:

```python
return {
    "global": {
        "speed": speed,
        "line_length": self._float_input(self._test_line_length_input, "线长", minimum=0.0, allow_zero=False),
        "y_spacing": self._float_input(self._test_y_spacing_input, "Y间距", minimum=0.0, allow_zero=False),
        "tool_change_safe_lift": self._float_input(self._test_tool_change_safe_lift_input, "安全抬升", minimum=0.0),
    },
    "resin": {
        "temp": temp,
        "layer_heights": layer_heights,
        "scales": scales,
        "prime_length": prime_length,
        "prime_speed": prime_speed,
        "retract_length": retract_length,
        "retract_speed": retract_speed,
        "line_count": line_count,
    },
    "fiber": {
        "temp": fiber_temp,
        "layer_heights": fiber_layer_heights,
        "scales": fiber_scales,
        "prime_length": fiber_prime_length,
        "prime_speed": fiber_prime_speed,
        "retract_length": fiber_retract_length,
        "retract_speed": fiber_retract_speed,
        "line_count": fiber_line_count,
    },
}
```

- [ ] **Step 2: Add calibration action handlers**

Add methods in the print test section:

```python
def _save_current_head_calibration(self):
    self._head_calibration = self._current_head_calibration_from_inputs()
    save_head_calibration(DEFAULT_HEAD_CALIBRATION_PATH, self._head_calibration)


def _on_print_test_confirm_resin_height(self):
    try:
        self._save_current_head_calibration()
    except Exception as exc:
        self._set_print_test_status(f"保存树脂高度失败: {exc}", "#b42318")
        return
    self._print_test_resin_height_confirmed = True
    self._set_print_test_controls_enabled(self._print_test_seen_correction)
    self._set_print_test_status("树脂打印高度已确认。", "#1b6e3c")


def _on_print_test_apply_fiber_offset(self):
    if self.current_tool_id() != 1:
        self._set_print_test_status("当前未使用纤维头，不能调整纤维偏置。", "#b42318")
        return
    target = (
        self._float_input(self._test_fiber_x_comp_input, "纤维X打印补偿"),
        self._float_input(self._test_fiber_y_comp_input, "纤维Y打印补偿"),
        self._float_input(self._test_fiber_z_comp_input, "纤维Z打印补偿"),
        self._print_test_current_correction[3],
        self._print_test_current_correction[4],
        self._print_test_current_correction[5],
    )
    self._run_print_test_job("travel", self._print_test_current_correction, target_pose=target)


def _on_print_test_confirm_fiber_offset(self):
    if self.current_tool_id() != 1:
        self._set_print_test_status("当前未使用纤维头，不能确认纤维偏置。", "#b42318")
        return
    try:
        self._save_current_head_calibration()
    except Exception as exc:
        self._set_print_test_status(f"保存纤维偏置失败: {exc}", "#b42318")
        return
    self._print_test_fiber_confirmed = True
    self._set_print_test_controls_enabled(self._print_test_seen_correction)
    self._set_print_test_status("纤维头偏置已确认。", "#1b6e3c")


def _on_print_test_cut(self):
    if self.current_tool_id() != 1:
        self._set_print_test_status("当前未使用纤维头，不能剪切。", "#b42318")
        return
    self.uart_command_submit.emit("EV 0 cut_cf\n")
    self._set_print_test_status("已发送剪切命令。", "#1b6e3c")
```

- [ ] **Step 3: Add safe continue-to-fiber flow**

Add `_on_print_test_continue_fiber`:

```python
def _on_print_test_continue_fiber(self):
    if not self._print_test_resin_height_confirmed:
        self._set_print_test_status("请先确认树脂打印高度。", "#b42318")
        return
    reply = _ask_yes_no(
        self,
        "切换到纤维头",
        "系统将先规划回 RSI 全 0 correction，到位后再切换到纤维喷头。是否继续？",
        QtWidgets.QMessageBox.No,
    )
    if reply != QtWidgets.QMessageBox.Yes:
        return
    self._print_test_pending_after_zero = "tool_change_cf"
    self._run_print_test_job(
        "travel",
        self._print_test_current_correction,
        target_pose=_PRINT_TEST_ZERO_CORRECTION,
    )
```

In `_on_current_correction`, after arrival detection, add:

```python
if self._print_test_pending_after_zero == "tool_change_cf":
    self._print_test_pending_after_zero = None
    self._print_test_waiting_for_tool = 1
    self.uart_command_submit.emit("EV 0 tool_change_cf 1\n")
    self._set_print_test_status("已回到 RSI 全 0，正在切换纤维头...", "#b15e00")
```

In `_update_ui`, when printhead status is valid, add:

```python
if self._print_test_waiting_for_tool and ps.current_tool == self._print_test_waiting_for_tool:
    self._print_test_waiting_for_tool = None
    self._set_print_test_controls_enabled(self._print_test_seen_correction)
    self._set_print_test_status("纤维头已切换完成，可调整纤维偏置。", "#1b6e3c")
```

- [ ] **Step 4: Update print action handlers**

Add:

```python
def _on_print_test_print_resin(self):
    self._run_print_test_job("resin_matrix", self._print_test_current_correction)


def _on_print_test_print_fiber(self):
    if self.current_tool_id() != 1:
        self._set_print_test_status("当前未使用纤维头，不能直接打印纤维。", "#b42318")
        return
    self._run_print_test_job("fiber_matrix", self._print_test_current_correction)


def _on_print_test_print_composite(self):
    if not self._print_test_fiber_confirmed:
        self._set_print_test_status("请先确认纤维头偏置。", "#b42318")
        return
    self._run_print_test_job("composite_matrix", self._print_test_current_correction)
```

Modify `_run_print_test_job` to support `travel`, `resin_matrix`, `fiber_matrix`, and `composite_matrix`. Import:

```python
from gcode_planner.print_test_generator import (
    format_gcode,
    generate_composite_test_matrix_gcode,
    generate_head_test_matrix_gcode,
    generate_pose_adjust_gcode,
    generate_z_adjust_gcode,
)
```

Inside worker:

```python
if job_type == "travel":
    lines = generate_pose_adjust_gcode(
        start_pose=start_pose,
        target_pose=target_pose,
        speed_mm_s=min(float(params["global"].get("speed", 10.0)), 10.0),
    )
    stem = "travel"
elif job_type == "resin_matrix":
    self._save_current_head_calibration()
    lines = generate_head_test_matrix_gcode(
        start_pose=start_pose,
        tool="resin",
        layer_heights_mm=params["resin"]["layer_heights"],
        extrusion_scales=params["resin"]["scales"],
        speed_mm_s=float(params["global"]["speed"]),
        line_length_mm=float(params["global"]["line_length"]),
        y_spacing_mm=float(params["global"]["y_spacing"]),
        finish_lift_mm=10.0,
        prime_length_mm=float(params["resin"]["prime_length"]),
        retract_length_mm=float(params["resin"]["retract_length"]),
        prime_speed_mm_s=float(params["resin"]["prime_speed"]),
        retract_speed_mm_s=float(params["resin"]["retract_speed"]),
    )
    stem = "resin_matrix"
elif job_type == "fiber_matrix":
    self._save_current_head_calibration()
    lines = generate_head_test_matrix_gcode(
        start_pose=start_pose,
        tool="fiber",
        layer_heights_mm=params["fiber"]["layer_heights"],
        extrusion_scales=params["fiber"]["scales"],
        speed_mm_s=float(params["global"]["speed"]),
        line_length_mm=float(params["global"]["line_length"]),
        y_spacing_mm=float(params["global"]["y_spacing"]),
        finish_lift_mm=10.0,
        prime_length_mm=float(params["fiber"]["prime_length"]),
        retract_length_mm=float(params["fiber"]["retract_length"]),
        prime_speed_mm_s=float(params["fiber"]["prime_speed"]),
        retract_speed_mm_s=float(params["fiber"]["retract_speed"]),
    )
    stem = "fiber_matrix"
elif job_type == "composite_matrix":
    self._save_current_head_calibration()
    lines = generate_composite_test_matrix_gcode(
        start_pose=start_pose,
        resin_layer_heights_mm=params["resin"]["layer_heights"],
        resin_extrusion_scales=params["resin"]["scales"],
        fiber_layer_heights_mm=params["fiber"]["layer_heights"],
        fiber_extrusion_scales=params["fiber"]["scales"],
        speed_mm_s=float(params["global"]["speed"]),
        line_length_mm=float(params["global"]["line_length"]),
        y_spacing_mm=float(params["global"]["y_spacing"]),
        finish_lift_mm=10.0,
        prime_length_mm=float(params["resin"]["prime_length"]),
        retract_length_mm=float(params["resin"]["retract_length"]),
        prime_speed_mm_s=float(params["resin"]["prime_speed"]),
        retract_speed_mm_s=float(params["resin"]["retract_speed"]),
        fiber_prime_length_mm=float(params["fiber"]["prime_length"]),
        fiber_retract_length_mm=float(params["fiber"]["retract_length"]),
        fiber_prime_speed_mm_s=float(params["fiber"]["prime_speed"]),
        fiber_retract_speed_mm_s=float(params["fiber"]["retract_speed"]),
        calibration=self._head_calibration,
        tool_change_safe_lift_mm=float(params["global"]["tool_change_safe_lift"]),
    )
    stem = "composite_matrix"
else:
    raise ValueError(f"未知测试动作: {job_type}")
```

- [ ] **Step 5: Run UI tests**

Run:

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py::test_scissor_button_checks_fiber_tool_before_uart_command src/my_project/my_project_ui/test/test_print_test_matrix_export.py::test_print_test_job_generation_supports_resin_fiber_and_composite_modes -q
```

Expected: PASS.

- [ ] **Step 6: Commit UI behavior**

```bash
git add src/my_project/my_project_ui/my_project_ui/ui_panel.py
git commit -m "feat: wire fiber test mode actions"
```

---

### Task 6: Formal Print Offset UI Re-layout

**Files:**
- Modify: `src/my_project/my_project_ui/my_project_ui/ui_panel.py`
- Modify: `src/my_project/my_project_ui/test/test_mode_selection_layout.py`

- [ ] **Step 1: Add failing formal UI layout test**

Append to `src/my_project/my_project_ui/test/test_mode_selection_layout.py`:

```python

def test_formal_print_offsets_show_resin_z_fiber_z_and_fiber_xy_only():
    src = _source()
    export_section = src.split("# ======== GCode Export 区域 ========", 1)[1].split(
        "# ======== Print Test 区域 ========", 1
    )[0]

    assert "喷头 Z 打印补偿" in export_section
    assert "树脂 Z" in export_section
    assert "纤维 Z" in export_section
    assert "纤维头 XY 偏置" in export_section
    assert "self._fiber_z_print_comp_spin" in export_section
    assert 'for axis, default_val in [("X",' in export_section
    assert '("Z", offset_cfg["tool_offset_z"])' not in export_section
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py::test_formal_print_offsets_show_resin_z_fiber_z_and_fiber_xy_only -q
```

Expected: FAIL because formal print UI still shows old layout.

- [ ] **Step 3: Re-layout formal print offset controls**

In `ui_panel.py` GCode Export area:

- Rename subtitle text to `喷头 Z 打印补偿`.
- Add `self._fiber_z_print_comp_spin = _NoWheelDoubleSpinBox()` beside resin Z.
- Set fiber Z default from calibration JSON or existing `tool_offset_z` fallback.
- Rename old `工具偏移` subtitle to `纤维头 XY 偏置`.
- Change offset loop to only X/Y:

```python
for axis, default_val in [
    ("X", offset_cfg["tool_offset_x"]),
    ("Y", offset_cfg["tool_offset_y"]),
]:
```

Update getters:

```python
def get_tool_offset(self):
    return (
        self._offset_spins["X"].value(),
        self._offset_spins["Y"].value(),
        self._fiber_z_print_comp_spin.value(),
    )


def current_resin_z_print_compensation(self):
    return self._resin_z_print_comp_spin.value()


def current_fiber_z_print_compensation(self):
    return self._fiber_z_print_comp_spin.value()
```

Update `_on_offset_changed` to save both legacy offset config and `head_offsets.json`:

```python
x = self._offset_spins["X"].value()
y = self._offset_spins["Y"].value()
fiber_z = self._fiber_z_print_comp_spin.value()
resin_z = self.current_resin_z_print_compensation()
_save_offset_config(x, y, fiber_z, resin_z)
save_head_calibration(
    DEFAULT_HEAD_CALIBRATION_PATH,
    HeadCalibration(
        resin_z_print_compensation_mm=resin_z,
        fiber_x_print_compensation_mm=x,
        fiber_y_print_compensation_mm=y,
        fiber_z_print_compensation_mm=fiber_z,
    ),
)
```

- [ ] **Step 4: Run formal UI layout test**

Run:

```bash
python3 -m pytest src/my_project/my_project_ui/test/test_mode_selection_layout.py::test_formal_print_offsets_show_resin_z_fiber_z_and_fiber_xy_only -q
```

Expected: PASS.

- [ ] **Step 5: Commit formal UI re-layout**

```bash
git add src/my_project/my_project_ui/my_project_ui/ui_panel.py src/my_project/my_project_ui/test/test_mode_selection_layout.py
git commit -m "feat: adjust formal print offset layout"
```

---

### Task 7: Documentation and Verification

**Files:**
- Modify: `src/my_project/my_project_ui/UI_ARCHITECTURE.md`

- [ ] **Step 1: Update architecture documentation**

In `src/my_project/my_project_ui/UI_ARCHITECTURE.md`, update the test mode section to mention:

```markdown
- 测试模式支持树脂单独矩阵、纤维单独矩阵和树脂加纤维复合矩阵。
- 喷头标定值保存到 `/home/jayson/kuka_ram_ws/data/head_calibration_offsets/head_offsets.json`。
- 人工进入纤维标定前必须先回 RSI 全 0 correction，再切换纤维头。
- 剪切按钮仅在当前工具为 CF/纤维头时发送预留 UART 命令。
```

Update formal print offset section to mention:

```markdown
- 正式打印 UI 显示树脂 Z、纤维 Z、纤维 X、纤维 Y。
- 本轮正式打印导出仍沿用既有补偿行为，未接入测试模式的新复合补偿模型。
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
python3 -m pytest src/my_project/gcode_planner/test -q -k 'not flake8 and not pep257'
python3 -m pytest src/my_project/my_project_ui/test -q -k 'not flake8 and not pep257'
python3 -m py_compile src/my_project/gcode_planner/gcode_planner/head_calibration.py src/my_project/gcode_planner/gcode_planner/print_test_generator.py src/my_project/my_project_ui/my_project_ui/ui_panel.py
```

Expected:

- gcode planner focused tests pass.
- UI focused tests pass.
- `py_compile` exits with no output.

- [ ] **Step 3: Run package build**

Run:

```bash
colcon build --packages-select gcode_planner my_project_ui
```

Expected: build finishes successfully for selected packages.

- [ ] **Step 4: Commit docs**

```bash
git add src/my_project/my_project_ui/UI_ARCHITECTURE.md
git commit -m "docs: describe fiber test mode"
```

---

## Self-Review

Spec coverage:

- 测试模式树脂 Z 输入和原按钮：Task 4 和 Task 5。
- 纤维 X/Y/Z 输入和应用：Task 4 和 Task 5。
- 偏置 JSON 覆盖写入并带时间：Task 1 和 Task 5。
- 树脂单独、纤维单独、复合打印：Task 2 和 Task 5。
- 剪切按钮只校验当前纤维头：Task 3 和 Task 5。
- 人工标定切换前回 RSI 全 0：Task 5。
- 已标定测试作业 tool change 安全抬升和补偿：Task 2。
- 正式打印只调整 UI，不接入导出补偿：Task 6。
- 文档和验证：Task 7。

Placeholder scan:

- 本计划没有占位词或未定义的后续步骤。
- 每个代码任务都有明确路径、测试命令和预期结果。

Type consistency:

- `HeadCalibration` 字段名在 Task 1、Task 2、Task 5、Task 6 中保持一致。
- 工具名使用 `resin` 和 `fiber`，工具号保持 resin `2`、fiber `1`。
- JSON 路径全程使用 `/home/jayson/kuka_ram_ws/data/head_calibration_offsets/head_offsets.json`。
