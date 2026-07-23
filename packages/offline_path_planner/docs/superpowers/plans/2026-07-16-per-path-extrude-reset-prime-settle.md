# Per-Path Extrusion Reset and Prime Settle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 在正式 external NPZ 链路中为每条打印路径增加收尾 extrude_reset，并在每次原地预挤出后增加默认 0.5 s 的稳定等待，同时保持 UART、RSI、控制中心和普通 GCode 行为不变。

**Architecture:** 参数只进入 external_npz_preprocessor.ProcessParams，并由现有 JSON、CLI 和 UI 入口统一传递。converter 在 prime 后生成零增量 ExtrudeWait，在每条路径收尾后生成带专用 raw 标记的 ResetECommand 和单周期 reset anchor；共享 npz_exporter 只对该 anchor 标记从 E=0 展开，其他命令完全沿用原逻辑。

**Tech Stack:** Python 3、dataclasses、NumPy NPZ、pytest、ROS 2 ament/colcon、Git。

---

## File map

- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/process_params.py
  - 定义 prime_settle_s 默认值与非负校验。
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/param_config.py
  - 从旧版或新版 JSON 读取全局参数。
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/cli.py
  - 暴露 --prime-settle-s 并构造 ProcessParams。
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py
  - 正式导出 UI 显示、加载和保存该参数。
- Modify: data/external_npz_preprocessor/print_params.json
  - 保存 prime_settle_s: 0.5。
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py
  - 生成 prime settle、每路径 reset、E 归零和 reset anchor。
- Modify: src/my_project/path_processing_core/path_processing_core/npz_exporter.py
  - 仅对 external_npz_reset_anchor 以 E=0 展开一个 ExtrudeWait。
- Modify: src/my_project/external_npz_preprocessor/test/test_process_params.py
- Modify: src/my_project/external_npz_preprocessor/test/test_param_config.py
- Modify: src/my_project/external_npz_preprocessor/test/test_cli.py
- Modify: src/my_project/external_npz_preprocessor/test/test_converter.py
- Modify: src/my_project/external_npz_preprocessor/test/test_export_runner.py
  - 覆盖参数、命令顺序、导出行数、E 基准、CUT 顺序和最终 anchor。
- Modify: src/my_project/external_npz_preprocessor/README.md
- Modify: src/my_project/external_npz_preprocessor/docs/PARAMETER_MODEL.md
- Modify: src/my_project/external_npz_preprocessor/docs/PROCESS_FLOW.md
- Modify: src/my_project/external_npz_preprocessor/docs/SOURCE_NPZ_FORMAT.md
- Modify: src/my_project/external_npz_preprocessor/docs/DEVELOPMENT_LOG.md
  - 删除“同工具路径持续累计 E”的旧说明，记录新边界。

### Task 1: Global prime settle parameter and entry points

**Files:**
- Modify: src/my_project/external_npz_preprocessor/test/test_process_params.py
- Modify: src/my_project/external_npz_preprocessor/test/test_param_config.py
- Modify: src/my_project/external_npz_preprocessor/test/test_cli.py
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/process_params.py
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/param_config.py
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/cli.py
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py
- Modify: data/external_npz_preprocessor/print_params.json

- [ ] **Step 1: Write failing model, JSON and CLI tests**

Append to test_process_params.py:

~~~python
def test_prime_settle_defaults_to_half_second_and_rejects_negative_values():
    assert ProcessParams().prime_settle_s == pytest.approx(0.5)
    assert ProcessParams(prime_settle_s=0.0).prime_settle_s == 0.0

    with pytest.raises(ValueError, match="prime_settle_s must be >= 0"):
        ProcessParams(prime_settle_s=-0.001)
~~~

Set prime_settle_s=0.75 in test_save_and_load_print_params_round_trip and append:

~~~python
def test_legacy_print_params_without_prime_settle_use_current_default(tmp_path):
    path = tmp_path / "legacy_without_settle.json"
    path.write_text('{"params":{"travel_feed_mm_s":12.0}}', encoding="utf-8")

    params = load_print_params(path)

    assert params.travel_feed_mm_s == 12.0
    assert params.prime_settle_s == pytest.approx(0.5)


def test_negative_prime_settle_in_json_is_rejected(tmp_path):
    path = tmp_path / "negative_settle.json"
    path.write_text('{"params":{"prime_settle_s":-0.1}}', encoding="utf-8")

    with pytest.raises(ValueError, match="prime_settle_s must be >= 0"):
        load_print_params(path)
~~~

Change test_cli.py imports and append:

~~~python
from external_npz_preprocessor.cli import build_parser, params_from_args


def test_prime_settle_cli_defaults_and_override_are_forwarded():
    default_args = build_parser().parse_args(["--source", "input.npz"])
    override_args = build_parser().parse_args(
        ["--source", "input.npz", "--prime-settle-s", "0.25"]
    )

    assert params_from_args(default_args).prime_settle_s == pytest.approx(0.5)
    assert params_from_args(override_args).prime_settle_s == pytest.approx(0.25)
~~~

Also add import pytest at the top of test_cli.py.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

~~~bash
python3 -m pytest -q   src/my_project/external_npz_preprocessor/test/test_process_params.py   src/my_project/external_npz_preprocessor/test/test_param_config.py   src/my_project/external_npz_preprocessor/test/test_cli.py
~~~

Expected: FAIL because ProcessParams has no prime_settle_s and the CLI has no --prime-settle-s option.

- [ ] **Step 3: Implement the parameter model and non-negative validation**

Add this field immediately before dt:

~~~python
    prime_settle_s: float = 0.5
    dt: float = 0.004
~~~

After max_fit_points_per_segment and before default_abc, add the validation method:

~~~python
    def __post_init__(self) -> None:
        if float(self.prime_settle_s) < 0.0:
            raise ValueError("prime_settle_s must be >= 0")
~~~

In process_params_from_dict, pass the merged value:

~~~python
        prime_settle_s=float(
            merged.get("prime_settle_s", defaults["prime_settle_s"])
        ),
        dt=float(merged.get("dt", defaults["dt"])),
~~~

- [ ] **Step 4: Wire the CLI and formal export UI**

Add to build_parser:

~~~python
    parser.add_argument("--prime-settle-s", type=float, default=0.5)
~~~

Pass it in params_from_args:

~~~python
        prime_settle_s=args.prime_settle_s,
        dt=args.dt,
~~~

In ExternalNpzPreprocessorWindow._build_ui create:

~~~python
        self.prime_settle_s = self._spin(0.5)
~~~

Replace the parameter-grid tail from 树脂风扇 through 默认 C with:

~~~python
            ("树脂风扇", self.resin_fan, "纤维风扇", self.fiber_fan),
            (
                "预挤出稳定等待 s",
                self.prime_settle_s,
                "空走速度 mm/s",
                self.travel_feed,
            ),
            ("左下角 X mm", self.start_x, "左下角 Y mm", self.start_y),
            ("", QtWidgets.QLabel(""), "默认 A", self.default_a),
            ("", QtWidgets.QLabel(""), "默认 B", self.default_b),
            ("", QtWidgets.QLabel(""), "默认 C", self.default_c),
~~~

In _apply_params add:

~~~python
        self.prime_settle_s.setValue(params.prime_settle_s)
~~~

In _params pass:

~~~python
            prime_settle_s=self.prime_settle_s.value(),
~~~

- [ ] **Step 5: Store the tracked default and verify parameter tests**

Add beside other global JSON parameters:

~~~json
    "prime_settle_s": 0.5,
    "dt": 0.004,
~~~

Run:

~~~bash
python3 -m pytest -q   src/my_project/external_npz_preprocessor/test/test_process_params.py   src/my_project/external_npz_preprocessor/test/test_param_config.py   src/my_project/external_npz_preprocessor/test/test_cli.py
python3 -m py_compile   src/my_project/external_npz_preprocessor/external_npz_preprocessor/process_params.py   src/my_project/external_npz_preprocessor/external_npz_preprocessor/param_config.py   src/my_project/external_npz_preprocessor/external_npz_preprocessor/cli.py   src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py
~~~

Expected: all focused tests pass and py_compile exits 0.

- [ ] **Step 6: Commit the parameter slice**

~~~bash
git add   data/external_npz_preprocessor/print_params.json   src/my_project/external_npz_preprocessor/external_npz_preprocessor/process_params.py   src/my_project/external_npz_preprocessor/external_npz_preprocessor/param_config.py   src/my_project/external_npz_preprocessor/external_npz_preprocessor/cli.py   src/my_project/external_npz_preprocessor/external_npz_preprocessor/ui.py   src/my_project/external_npz_preprocessor/test/test_process_params.py   src/my_project/external_npz_preprocessor/test/test_param_config.py   src/my_project/external_npz_preprocessor/test/test_cli.py
git diff --cached --check
git commit -m "feat: add external npz prime settle parameter"
~~~

### Task 2: Converter path-boundary commands

**Files:**
- Modify: src/my_project/external_npz_preprocessor/test/test_converter.py
- Modify: src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py

- [ ] **Step 1: Add a compact two-path resin fixture and failing sequence tests**

Append this fixture and token helper to test_converter.py:

~~~python
def _two_resin_path_job():
    return SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        "R",
                        0,
                        np.array(
                            [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                             [10.0, 0.0, 0.5, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    ),
                    MaterialPath(
                        "R",
                        1,
                        np.array(
                            [[20.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                             [30.0, 0.0, 0.5, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    ),
                ],
                fiber_paths=[],
            )
        ],
    )


def _external_boundary_tokens(commands):
    tokens = []
    for cmd in commands:
        if isinstance(cmd, GlobalCurveCommand) and cmd.type == "PRINT":
            tokens.append("print")
        elif isinstance(cmd, ResetECommand) and cmd.raw == "external_npz_path_reset":
            tokens.append("reset")
        elif isinstance(cmd, MoveCommand) and cmd.raw == "external_npz_travel":
            tokens.append("travel")
        elif isinstance(cmd, ExtrudeWait) and cmd.raw in {
            "external_npz_retract",
            "external_npz_prime",
            "external_npz_prime_settle",
            "external_npz_reset_anchor",
        }:
            tokens.append(cmd.raw.removeprefix("external_npz_"))
    return tokens
~~~

Append these tests:

~~~python
def test_resin_paths_reset_after_retract_and_settle_after_prime():
    commands = source_job_to_parsed_commands(
        _two_resin_path_job(),
        ProcessParams(prime_settle_s=0.5, dt=0.004),
    )

    assert _external_boundary_tokens(commands) == [
        "retract", "prime", "prime_settle", "print",
        "retract", "reset", "reset_anchor", "travel",
        "prime", "prime_settle", "print",
        "retract", "reset", "reset_anchor", "travel",
        "prime", "prime_settle", "print",
        "retract", "reset", "reset_anchor",
    ]

    path_resets = [
        cmd for cmd in commands
        if isinstance(cmd, ResetECommand)
        and cmd.raw == "external_npz_path_reset"
    ]
    path_anchors = [
        cmd for cmd in commands
        if isinstance(cmd, ExtrudeWait)
        and cmd.raw == "external_npz_reset_anchor"
    ]
    assert len(path_resets) == 3
    assert len(path_anchors) == 3
    assert commands[-2:] == [path_resets[-1], path_anchors[-1]]
    assert path_anchors[-1].wait_sec == pytest.approx(0.004)
    assert path_anchors[-1].delta_e == 0.0


def test_path_reset_rebases_travel_and_next_prime_to_zero():
    params = ProcessParams(prime_settle_s=0.5)
    commands = source_job_to_parsed_commands(_two_resin_path_job(), params)

    first_path_reset_idx = next(
        idx for idx, cmd in enumerate(commands)
        if isinstance(cmd, ResetECommand)
        and cmd.raw == "external_npz_path_reset"
    )
    next_travel = next(
        cmd for cmd in commands[first_path_reset_idx + 1:]
        if isinstance(cmd, MoveCommand) and cmd.raw == "external_npz_travel"
    )
    next_prime = next(
        cmd for cmd in commands[first_path_reset_idx + 1:]
        if isinstance(cmd, ExtrudeWait) and cmd.raw == "external_npz_prime"
    )
    next_curve = next(
        cmd for cmd in commands[first_path_reset_idx + 1:]
        if isinstance(cmd, GlobalCurveCommand) and cmd.type == "PRINT"
    )

    assert next_travel.e_val == 0.0
    assert next_travel.delta_e == 0.0
    assert next_prime.delta_e == pytest.approx(params.resin.prime_length_mm)
    assert next_curve.e_val - next_curve.delta_e == pytest.approx(
        params.resin.prime_length_mm
    )


def test_zero_prime_settle_disables_only_settle_rows():
    commands = source_job_to_parsed_commands(
        _two_resin_path_job(),
        ProcessParams(prime_settle_s=0.0),
    )

    assert not any(
        isinstance(cmd, ExtrudeWait)
        and cmd.raw == "external_npz_prime_settle"
        for cmd in commands
    )
    assert sum(
        isinstance(cmd, ResetECommand)
        and cmd.raw == "external_npz_path_reset"
        for cmd in commands
    ) == 3
~~~

Extend the existing fiber CUT test with:

~~~python
    fiber_print_idx = max(
        idx for idx, cmd in enumerate(commands)
        if isinstance(cmd, GlobalCurveCommand)
        and cmd.subtype == "FIBER_PRINT"
    )
    assert isinstance(commands[fiber_print_idx + 1], MCommand)
    assert commands[fiber_print_idx + 1].code == "CUT"
    assert isinstance(commands[fiber_print_idx + 2], ResetECommand)
    assert commands[fiber_print_idx + 2].raw == "external_npz_path_reset"
    assert isinstance(commands[fiber_print_idx + 3], ExtrudeWait)
    assert commands[fiber_print_idx + 3].raw == "external_npz_reset_anchor"
~~~

- [ ] **Step 2: Run converter tests and confirm feature assertions fail**

Run:

~~~bash
python3 -m pytest -q src/my_project/external_npz_preprocessor/test/test_converter.py
~~~

Expected: new tests fail because prime settle, path reset and reset anchor commands do not exist.

- [ ] **Step 3: Generate prime settle immediately after a real prime**

Replace _path_prime_waits with:

~~~python
def _path_prime_waits(
    material: str,
    params: ProcessParams,
    line: int,
    layer: int,
    subtype: str,
) -> list[ExtrudeWait]:
    process = _process_params_for_material(material, params)
    prime = _make_extrude_wait(
        delta_e=float(process.prime_length_mm),
        speed_mm_s=float(process.prime_speed_mm_s),
        line=line,
        layer=layer,
        subtype=subtype,
        raw="external_npz_prime",
    )
    if prime is None:
        return []

    waits = [prime]
    if float(params.prime_settle_s) > 0.0:
        waits.append(
            ExtrudeWait(
                type="EXTRUDE_WAIT",
                wait_sec=float(params.prime_settle_s),
                delta_e=0.0,
                feedrate=prime.feedrate,
                line=line + 1,
                layer=layer,
                subtype=subtype,
                raw="external_npz_prime_settle",
            )
        )
    return waits
~~~

This intentionally produces no settle when prime_length_mm is zero because there is no prime completion to settle.

- [ ] **Step 4: Add one path reset and one anchor after every path**

Add:

~~~python
def _path_reset_commands(
    *,
    params: ProcessParams,
    line: int,
    layer: int,
    subtype: str,
    pose: Position,
) -> list[ResetECommand | ExtrudeWait]:
    return [
        ResetECommand(
            type="RESET_E",
            val=0.0,
            line=line,
            layer=layer,
            subtype=subtype,
            raw="external_npz_path_reset",
            pose=pose,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=float(params.dt),
            delta_e=0.0,
            feedrate=max(float(params.travel_feed_mm_s), _EPS) * 60.0,
            line=line + 1,
            layer=layer,
            subtype=subtype,
            raw="external_npz_reset_anchor",
        ),
    ]
~~~

After the existing resin retract block or fiber CUT insertion, append:

~~~python
            for boundary_cmd in _path_reset_commands(
                params=params,
                line=line,
                layer=layer.index,
                subtype=subtype,
                pose=previous_pose,
            ):
                commands.append(boundary_cmd)
                line += 1
            current_e = 0.0
            current_pose = previous_pose
~~~

Keep the existing tool-change ResetECommand untouched. The new reset is identified only by raw="external_npz_path_reset".

- [ ] **Step 5: Update existing converter assertions to preserve their original intent**

In the first converter test split reset assertions:

~~~python
    tool_resets = [
        cmd for cmd in resets if cmd.raw == "G92 E0"
    ]
    path_resets = [
        cmd for cmd in resets if cmd.raw == "external_npz_path_reset"
    ]
    assert len(tool_resets) == 2
    assert len(path_resets) == 3
~~~

In tests that verify prime/retract magnitudes, filter zero-delta settle and anchor waits:

~~~python
    waits = [
        cmd for cmd in commands
        if isinstance(cmd, ExtrudeWait) and abs(cmd.delta_e) > 1e-9
    ]
~~~

Replace the fiber compact helper and assertion with:

~~~python
    compact = []
    for cmd in commands:
        if isinstance(cmd, GlobalCurveCommand) and cmd.type == "PRINT":
            compact.append(("print", None))
        elif isinstance(cmd, ExtrudeWait):
            compact.append((cmd.raw, cmd.delta_e))
        elif isinstance(cmd, MoveCommand) and cmd.type == "TRAVEL":
            compact.append(("travel", cmd.raw))
        elif isinstance(cmd, MCommand) and cmd.code == "CUT":
            compact.append(("cut", cmd.params))
        elif (
            isinstance(cmd, ResetECommand)
            and cmd.raw == "external_npz_path_reset"
        ):
            compact.append(("reset", None))

    assert compact == [
        ("travel", "external_npz_start_xy_travel"),
        ("external_npz_retract", -15.0),
        ("external_npz_prime", 18.0),
        ("external_npz_prime_settle", 0.0),
        ("print", None),
        ("external_npz_retract", -15.0),
        ("reset", None),
        ("external_npz_reset_anchor", 0.0),
        ("travel", "external_npz_travel"),
        ("external_npz_prime", 12.0),
        ("external_npz_prime_settle", 0.0),
        ("print", None),
        ("cut", {"P": 1.0}),
        ("reset", None),
        ("external_npz_reset_anchor", 0.0),
        ("travel", "external_npz_travel"),
        ("external_npz_prime", 12.0),
        ("external_npz_prime_settle", 0.0),
        ("print", None),
        ("cut", {"P": 1.0}),
        ("reset", None),
        ("external_npz_reset_anchor", 0.0),
    ]
~~~

- [ ] **Step 6: Run converter tests and commit**

Run:

~~~bash
python3 -m pytest -q src/my_project/external_npz_preprocessor/test/test_converter.py
~~~

Expected: all converter tests pass.

Commit:

~~~bash
git add   src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py   src/my_project/external_npz_preprocessor/test/test_converter.py
git diff --cached --check
git commit -m "feat: reset extrusion after external npz paths"
~~~

### Task 3: Export reset anchor as a single E=0 trajectory row

**Files:**
- Modify: src/my_project/external_npz_preprocessor/test/test_export_runner.py
- Modify: src/my_project/path_processing_core/path_processing_core/npz_exporter.py

- [ ] **Step 1: Add a failing exporter boundary test**

Add imports for export_npz and command types, then append:

~~~python
def test_external_reset_anchor_exports_one_zero_e_row_without_changing_regular_holds(
    tmp_path,
):
    from path_processing_core.npz_exporter import export_npz
    from path_processing_core.types import ExtrudeWait, Position, ResetECommand

    pose = Position(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
    before = ExtrudeWait(
        type="EXTRUDE_WAIT",
        wait_sec=0.004,
        delta_e=5.0,
        feedrate=600.0,
        line=1,
        layer=0,
        subtype="RESIN_PRINT",
        raw="before_reset",
    )
    reset = ResetECommand(
        type="RESET_E",
        val=0.0,
        line=2,
        layer=0,
        subtype="RESIN_PRINT",
        raw="external_npz_path_reset",
        pose=pose,
    )
    anchor = ExtrudeWait(
        type="EXTRUDE_WAIT",
        wait_sec=0.004,
        delta_e=0.0,
        feedrate=600.0,
        line=3,
        layer=0,
        subtype="RESIN_PRINT",
        raw="external_npz_reset_anchor",
    )

    anchor_out = tmp_path / "anchor.npz"
    export_npz(
        [before, reset, anchor],
        str(anchor_out),
        dt=0.004,
        enable_extrude_wait=True,
        enable_travel_extrude_overlap=False,
    )
    with np.load(anchor_out) as data:
        src_lines = _decoded_src_lines(data)
        reset_idx = src_lines.index("2")
        anchor_idx = [idx for idx, src in enumerate(src_lines) if src == "3"]
        assert len(anchor_idx) == 1
        assert data["event_flag"][reset_idx] == 1
        assert data["e"][reset_idx] == pytest.approx(5.0)
        assert data["e"][anchor_idx[0]] == pytest.approx(0.0)
        assert np.allclose(
            data["x"][anchor_idx],
            data["x"][reset_idx],
        )
        assert np.allclose(
            data["z"][anchor_idx],
            data["z"][reset_idx],
        )

    regular_out = tmp_path / "regular_hold.npz"
    regular_hold = replace(anchor, raw="ordinary_zero_delta_hold")
    export_npz(
        [before, reset, regular_hold],
        str(regular_out),
        dt=0.004,
        enable_extrude_wait=True,
        enable_travel_extrude_overlap=False,
    )
    with np.load(regular_out) as data:
        regular_idx = [
            idx for idx, src in enumerate(_decoded_src_lines(data)) if src == "3"
        ]
        assert len(regular_idx) == 1
        assert data["e"][regular_idx[0]] == pytest.approx(5.0)
~~~

Add from dataclasses import replace, import numpy as np, and import pytest at the top of test_export_runner.py.

- [ ] **Step 2: Verify the anchor test fails for the expected reason**

Run:

~~~bash
python3 -m pytest -q   src/my_project/external_npz_preprocessor/test/test_export_runner.py::test_external_reset_anchor_exports_one_zero_e_row_without_changing_regular_holds
~~~

Expected: FAIL because the anchor currently inherits E=5.0.

- [ ] **Step 3: Add the raw-gated exporter behavior**

In _append_extrude_wait replace the start_e assignment with:

~~~python
        start_e = (
            0.0
            if (cmd.raw or "") == "external_npz_reset_anchor"
            else hold_row.e
        )
~~~

No event vocabulary, UART payload, reset semantics or ordinary ExtrudeWait branch changes.

- [ ] **Step 4: Verify 0.5 s becomes exactly 125 stationary rows**

Append:

~~~python
def test_prime_settle_exports_125_stationary_rows_at_four_milliseconds(tmp_path):
    from path_processing_core.npz_exporter import export_npz
    from path_processing_core.types import ExtrudeWait

    out = tmp_path / "prime_settle.npz"
    commands = [
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.004,
            delta_e=2.0,
            feedrate=600.0,
            line=10,
            raw="external_npz_prime",
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.5,
            delta_e=0.0,
            feedrate=600.0,
            line=11,
            raw="external_npz_prime_settle",
        ),
    ]

    export_npz(
        commands,
        str(out),
        dt=0.004,
        enable_extrude_wait=True,
        enable_travel_extrude_overlap=False,
    )

    with np.load(out) as data:
        src_lines = _decoded_src_lines(data)
        settle_idx = [idx for idx, src in enumerate(src_lines) if src == "11"]
        assert len(settle_idx) == 125
        assert np.allclose(data["x"][settle_idx], data["x"][settle_idx[0]])
        assert np.allclose(data["y"][settle_idx], data["y"][settle_idx[0]])
        assert np.allclose(data["z"][settle_idx], data["z"][settle_idx[0]])
        assert np.allclose(data["e"][settle_idx], 2.0)
~~~

- [ ] **Step 5: Update the fiber end-to-end assertion for CUT, reset, anchor, travel and prime**

In test_fiber_cut_lift_retracts_before_travel_and_next_path_prepares_after_travel, locate the first extrude_reset after cut:

~~~python
    reset_idx = next(
        idx for idx in range(cut_idx + 1, len(event_types))
        if event_types[idx] == "extrude_reset"
    )
    cut_motion_idx = [
        idx for idx in range(cut_idx + 1, reset_idx)
        if event_types[idx] == ""
    ]
    anchor_idx = reset_idx + 1
    anchor_src = src_lines[anchor_idx]
    travel_start_idx = next(
        idx for idx in range(anchor_idx + 1, len(src_lines))
        if src_lines[idx] != anchor_src and event_types[idx] == ""
    )
    travel_src = src_lines[travel_start_idx]
    travel_idx = [
        idx for idx in range(travel_start_idx, len(src_lines))
        if src_lines[idx] == travel_src
    ]

    assert cut_motion_idx
    assert np.isclose(data["e"][cut_motion_idx[-1]], cut_e)
    assert np.isclose(data["e"][reset_idx], cut_e)
    assert np.isclose(data["e"][anchor_idx], 0.0)
    assert np.allclose(data["e"][travel_idx], 0.0)

    next_prime_idx = travel_idx[-1] + 1
    assert np.isclose(data["e"][next_prime_idx], 6.0)
    assert np.isclose(data["z"][next_prime_idx], data["z"][travel_idx[-1]])
~~~

This replaces the old expectation that travel retained cut_e.

In test_convert_writes_startup_events_and_tool_reset_order_to_npz, replace the leading event assertion with:

~~~python
    assert non_empty_events[:9] == [
        "fan_resin",
        "fan_cf",
        "heat_resin",
        "heat_cf",
        "extrude_reset",
        "extrude_reset",
        "extrude_reset",
        "tool_change_cf",
        "extrude_reset",
    ]
~~~

These are the initial resin tool reset, generated primeline path reset, source resin path reset, fiber tool change, and existing post-tool-change reset in order.

- [ ] **Step 6: Run exporter and existing GCode reset regressions**

Run:

~~~bash
python3 -m pytest -q   src/my_project/external_npz_preprocessor/test/test_export_runner.py   src/my_project/gcode_planner/test/test_extrude_reset_payload.py
~~~

Expected: all tests pass, including existing ordinary G92 and ExtrudeWait tests.

- [ ] **Step 7: Commit the exporter boundary slice**

~~~bash
git add   src/my_project/path_processing_core/path_processing_core/npz_exporter.py   src/my_project/external_npz_preprocessor/test/test_export_runner.py
git diff --cached --check
git commit -m "fix: export external npz reset anchor from zero"
~~~

### Task 4: Documentation and full regression verification

**Files:**
- Modify: src/my_project/external_npz_preprocessor/README.md
- Modify: src/my_project/external_npz_preprocessor/docs/PARAMETER_MODEL.md
- Modify: src/my_project/external_npz_preprocessor/docs/PROCESS_FLOW.md
- Modify: src/my_project/external_npz_preprocessor/docs/SOURCE_NPZ_FORMAT.md
- Modify: src/my_project/external_npz_preprocessor/docs/DEVELOPMENT_LOG.md

- [ ] **Step 1: Update the documented boundary sequence**

Use this canonical sequence in README.md, PARAMETER_MODEL.md and PROCESS_FLOW.md:

~~~text
树脂:
PRINT -> RETRACT -> extrude_reset -> RESET_ANCHOR(E=0)
      -> TRAVEL(E=0) -> PRIME -> PRIME_SETTLE(default 0.5 s) -> PRINT

纤维:
PRINT -> CUT(lift/wait/safety retract) -> extrude_reset
      -> RESET_ANCHOR(E=0) -> TRAVEL(E=0)
      -> PRIME -> PRIME_SETTLE(default 0.5 s) -> PRINT
~~~

State explicitly:

~~~text
prime_settle_s is a global external-NPZ parameter. 0 disables it; negative
values are rejected. Every external print path, including the generated
primeline and the final path, emits exactly one external_npz_path_reset.
Tool-change resets remain unchanged and are additional to path resets.
~~~

Replace wording that says E accumulates across consecutive same-tool paths. Update SOURCE_NPZ_FORMAT.md to clarify that source geometry is unchanged and these are generated command boundaries, not new source NPZ fields. Append a dated DEVELOPMENT_LOG entry describing the two new behaviors and the raw-gated anchor exporter rule.

- [ ] **Step 2: Run the complete external NPZ suite**

~~~bash
python3 -m pytest -q src/my_project/external_npz_preprocessor/test
~~~

Expected: all external NPZ tests pass.

- [ ] **Step 3: Run shared exporter, UART and control-center boundary regressions**

~~~bash
python3 -m pytest -q   src/my_project/gcode_planner/test/test_extrude_reset_payload.py   src/my_project/uart_bridge/test/test_extrude_reset_handshake.py   src/my_project/uart_bridge/test/test_temperature_ready_threshold.py   src/my_project/uart_bridge/test/test_uart_extrusion_boundary_contract.py   src/my_project/control_center/test/test_print_test_reset_queue.py   src/my_project/control_center/test/test_auto_abort_on_print_complete.py   src/my_project/control_center/test/test_staged_pause_contract.py
~~~

Expected: all selected regressions pass with no UART, RSI or control-center source changes.

- [ ] **Step 4: Build and run package-level tests**

~~~bash
source /opt/ros/humble/setup.bash
colcon build --packages-select   path_processing_core external_npz_preprocessor gcode_planner uart_bridge control_center
colcon test --packages-select   path_processing_core external_npz_preprocessor gcode_planner uart_bridge control_center   --event-handlers console_direct+
colcon test-result --verbose
~~~

Expected: build succeeds, selected package tests pass, and test-result reports no failures.

- [ ] **Step 5: Verify the Git scope**

~~~bash
git diff --check
git status --short
git diff --name-only origin/feature/uart-canonical-v1...HEAD
~~~

Expected changed implementation paths are limited to external_npz_preprocessor, the one raw-gated npz_exporter location, tracked parameter JSON, tests and documentation. No files under uart_bridge, control_center, rsi_server or message definitions appear in the diff.

- [ ] **Step 6: Commit documentation**

~~~bash
git add   src/my_project/external_npz_preprocessor/README.md   src/my_project/external_npz_preprocessor/docs/PARAMETER_MODEL.md   src/my_project/external_npz_preprocessor/docs/PROCESS_FLOW.md   src/my_project/external_npz_preprocessor/docs/SOURCE_NPZ_FORMAT.md   src/my_project/external_npz_preprocessor/docs/DEVELOPMENT_LOG.md
git diff --cached --check
git commit -m "docs: describe external npz extrusion boundaries"
~~~

- [ ] **Step 7: Final branch check without pushing**

~~~bash
git status --short --branch
git log --oneline --decorate -5
~~~

Expected: feature/uart-canonical-v1 is clean and ahead of origin. Do not merge, delete, rewrite or push the branch without separate user authorization.
