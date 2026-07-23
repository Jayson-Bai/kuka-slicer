# External NPZ Preprocessor 处理流程

## 1. 总体链路

外部源 NPZ 不直接写成本系统 NPZ。完整链路固定为：

```text
外部源 NPZ
-> load_source_npz()
-> SourceJob / LayerPaths / MaterialPath
-> source_job_to_parsed_commands()
-> path_processing_core.types 命令序列
-> path_processing_core.npz_exporter.export_npz()
-> 系统可用 NPZ
```

核心原则：最终系统 NPZ 必须通过 `path_processing_core.npz_exporter.export_npz()` 写出，preprocessor 只负责把外部路径转换成 exporter 已支持的命令序列。

## 2. 源 NPZ 读取

源文件是标准 `.npz`，按层和材料分 key：

```text
meta
layer_0000_R
layer_0000_F
layer_0001_R
layer_0001_F
```

`R` 是树脂，`F` 是纤维。每个 `layer_xxxx_R/F` 是数值型 `float32` 三维数组：

```text
[path_count, max_points_per_path, columns]
```

短路径用整行 `NaN` padding。有效点只支持：

```text
Nx3: [x, y, z]
Nx6: [x, y, z, a, b, c]
```

`Nx3` 会补默认 `ABC` 成 `Nx6`。源 NPZ 必须包含 Z；UI 中的树脂/纤维层高不会生成或覆盖路径 Z。这样曲面切片软件产生的曲面高度会被保留。

## 3. 路径顺序

转换时按层顺序处理。每一层内目前先处理该层所有树脂路径，再处理该层所有纤维路径：

```text
layer.resin_paths
layer.fiber_paths
```

每条 R/F 路径都会独立形成一段打印流程。路径之间如果当前位置和下一条路径起点不同，会插入空走 `MoveCommand(type="TRAVEL")`。

## 4. 工具切换事件

材料和 GCode 工具号映射为：

```text
树脂 R -> gcode tool 1 -> exporter 内映射为系统树脂工具
纤维 F -> gcode tool 0 -> exporter 内映射为系统纤维工具
```

当下一条路径的材料工具和当前工具不同时，preprocessor 先插入：

```text
ToolChangeCommand(type="TOOL_CHANGE", tool=...)
```

`npz_exporter` 接收这个命令后，会在系统 NPZ 中生成工具切换事件：

```text
tool_change_cf
tool_change_resin
```

实际工具偏置补偿不在 preprocessor 中重复实现，而是通过 `export_npz()` 的 `tool_offset` 参数交给 exporter 处理。exporter 默认初始工具为系统树脂工具 `2`；当发生工具切换且存在偏置时，顺序固定为先安全抬升 `20 mm`，再执行喷头 XYZ 偏置补偿 travel，最后写入 `tool_change_cf/tool_change_resin` 事件。

## 5. 加热和风扇事件

preprocessor 会在命令序列开头插入两个喷头的启动工艺事件，使系统开始执行 NPZ 后先让树脂/纤维风扇进入目标状态，并把两个喷头加热到配置温度：

```text
M106 T1 或 M107 T1
M106 T0 或 M107 T0
M104 T1 S<resin_temperature>
M104 T0 S<fiber_temperature>
```

具体规则：

- 风扇事件总是在路径前插入；树脂/纤维风扇默认开启。
- 温度 `temperature_c > 0` 时插入 `M104`。
- 树脂默认温度 `250 C`，纤维默认温度 `250 C`。

`npz_exporter` 会把这些 `MCommand` 转成系统 NPZ 事件：

```text
fan_resin
fan_cf
heat_resin
heat_cf
```

事件 payload 中保存温度或风扇开关状态。

## 6. 挤出量重置

工具切换后的既有重置保持不变。preprocessor 在每次 `ToolChangeCommand` 后立即插入：

```text
ResetECommand(type="RESET_E", val=0.0, raw="G92 E0")
```

此外，每条可打印路径完成自己的树脂回抽或纤维 `CUT` 展开后，都会追加一组独立路径边界：

```text
ResetECommand(val=0.0, raw="external_npz_path_reset", pose=path_endpoint)
ExtrudeWait(wait_sec=dt, delta_e=0.0, raw="external_npz_reset_anchor")
```

该规则覆盖 converter 生成的 primeline、所有源路径和最终路径；最终路径即使没有后续 travel，也仍然以 reset + anchor 结束。路径 reset 与工具切换 reset 是叠加关系，不替换工具切换逻辑。

`npz_exporter` 仍把每个 `ResetECommand` 写成 `extrude_reset` 事件，事件行保留 reset 前的旧 E。随后只有精确内部标记 `external_npz_reset_anchor` 会让一个 `dt` 周期的等待行从 `E=0` 导出；普通 `ExtrudeWait` 和 GCode reset 行为不变。converter 同时把 `current_e` 置零，所以后续 travel 保持 `E=0`，下一条路径的 prime 从零重新计算。

## 7. 预挤出和回抽

预挤出/回抽不伪装成移动路径，而是插入 `ExtrudeWait`，表示原地挤出或原地回抽。导出时必须启用：

```text
enable_extrude_wait=True
```

`prime_settle_s` 是全局 external-NPZ 参数，默认 `0.5 s`。它保存到 `print_params.json`，并由 CLI、独立 UI 和正式打印 UI 暴露。旧 JSON 缺字段时使用 `0.5 s`；`0` 只关闭 settle，负数、`NaN` 和无穷值会被拒绝。prime 长度为零时不生成 prime，也不生成 settle。

### 路径准备规则

等待段不并入 travel。树脂继续使用原有的逐路径 `prime -> 可选 prime_settle -> PRINT -> retract` 规则，整件第一条可打印路径前的既有初始回抽也保持不变。

纤维 UI 预挤出/回抽按“含纤维层”的首尾路径生效。每个独立 UI 动作都用 reset + anchor 从 `E=0` 开始，避免与打印累计值或 CUT 固定动作共享绝对 E 基准：

```text
# Whole job's first fiber, after travel reaches its start:
ResetECommand(raw="external_npz_fiber_prepare_reset")
ExtrudeWait(delta_e=0, raw="external_npz_reset_anchor")
ExtrudeWait(delta_e=-fiber.retract_length_mm, raw="external_npz_fiber_initial_retract")

# First fiber of every fiber-bearing layer:
ResetECommand(raw="external_npz_fiber_prime_reset")
ExtrudeWait(delta_e=0, raw="external_npz_reset_anchor")
ExtrudeWait(delta_e=+fiber.prime_length_mm, raw="external_npz_prime")
ExtrudeWait(wait_sec=prime_settle_s, delta_e=0, raw="external_npz_prime_settle")
ResetECommand(raw="external_npz_fiber_print_reset")
ExtrudeWait(delta_e=0, raw="external_npz_reset_anchor")
PRINT path

# Middle fibers in the same layer:
TRAVEL(E=0) -> PRINT path

# Every fiber path end:
extrude_reset event (blocking, pre-CUT)
ExtrudeWait(delta_e=0, raw="external_npz_reset_anchor")
CUT
# exporter expands external-NPZ CUT as:
# cut event -> immediate lift(Z:+L, E:0→+L)
# -> hold XYZ high and E=+L for 3 s
# -> reset -> anchor(E=0)
# -> retract at high pose(E:0→-L)
# -> hold XYZ high and E=-L for 3 s
# -> reset -> anchor(E=0)
# -> hold XYZ high for the remaining cut_wait_s budget

# Last fiber of every fiber-bearing layer, after CUT:
ResetECommand(raw="external_npz_fiber_layer_retract_reset")
ExtrudeWait(delta_e=0, raw="external_npz_reset_anchor")
ExtrudeWait(delta_e=-fiber.retract_length_mm, raw="external_npz_fiber_layer_retract")

# Every printable path still ends with:
ResetECommand(raw="external_npz_path_reset")
ExtrudeWait(wait_sec=dt, delta_e=0, raw="external_npz_reset_anchor")

# If another path follows, travel remains:
TRAVEL(E=0)
```

其中 `L=cut_lift_mm`。CUT 前先完成一次阻塞式 reset 和 E=0 anchor，随后 CUT 事件仍为非阻塞，RSI 紧接着抬升并同步执行 `E:0→+L`。抬升末端保持 3 秒后 reset，回抽由新基准独立执行 `E:0→-L`，再保持 3 秒后 reset。两个 3 秒保持段都在高位，且计入原 `cut_wait_s` 总窗口；剩余等待继续保持高位和 `E=0`。若 UI 等待短于完整执行抬升、两段保持和回抽所需的安全时间，则优先完整执行这些动作。

CUT 的挤出和固定回抽只受 L 控制，不读取 `fiber.prime_length_mm` 或 `fiber.retract_length_mm`。层末 UI 回抽前的 reset 同时阻断 CUT 向后读取 UI 回抽速度；CUT 固定回抽继续使用默认移动速度，UI 回抽使用自己的 `fiber.retract_speed_mm_s`。

只有一条纤维的层同时执行层首预挤出和层末回抽。无纤维层不产生纤维 UI 动作；下一个实际含纤维层仍按层首规则处理。path reset 之后不保留任何正/负 E 累计，travel 行保持 `E=0`。

`prime_settle_s=0.5` 且 `dt=0.004` 时，settle 导出为恰好 125 个固定 XYZ、固定 E 的采样行。

当前默认值：

```text
树脂回抽:   15 mm @ 30 mm/s
树脂预挤出: 18 mm @ 15 mm/s
纤维回抽:   10 mm @ 5 mm/s
纤维预挤出: 12 mm @ 5 mm/s
全局稳定等待: 0.5 s
```

## 8. 打印路径和挤出量计算

每条路径从第一个点开始，逐段生成 `MoveCommand(type="PRINT")`：

```text
start_pos = 上一点
pos       = 下一点
feedrate  = （材料首层打印速度或非首层打印速度） * 60
delta_e   = 三维路径长度 * e_per_mm
e_val     = 当前工具内累计 E
```

路径长度使用三维距离：

```text
sqrt(dx^2 + dy^2 + dz^2)
```

所以曲面路径中的 Z 起伏会参与路径长度和挤出量计算。树脂和纤维分别记录自己的首个实际含该材料层；二者不要求出现在同一个层号。自动擦料线固定使用首层树脂打印速度。

树脂 `E/mm`：

```text
resin_e_per_mm = 2.0 * resin_layer_height_mm * resin_extrusion_scale / (pi * (1.75 / 2)^2)
```

这里 `2.0 mm` 是固定树脂线宽，`1.75 mm` 是固定树脂耗材直径；计算方式与原 G-code 测试线一致，先得到每毫米路径的沉积体积，再除以耗材截面积换成 E 轴料长。树脂层高只作为挤出计算参考，不改变源 Z。

纤维 `E/mm`：

```text
fiber_e_per_mm = fiber_extrusion_scale
```

默认 `fiber_extrusion_scale = 1.0`，表示纤维挤出速度和 TCP 移动速度一致。纤维层高目前是工艺参考参数，不改变源 Z，也不改变默认纤维 E。

纤维路径额外携带 `fiber_start_accel_s` 作为曲线级起始加速时间，默认 `2.0 s`。导出器只在纤维曲线显式携带该字段时把它传给七阶时间参数化的 `t_acc`；未携带该字段的树脂、空走、剪切抬升等路径继续使用全局默认七阶逻辑。

## 9. 空走路径

如果上一条路径终点和下一条路径起点不同，preprocessor 插入：

```text
MoveCommand(type="TRAVEL", cmd="G0")
```

空走速度按终点路径所属材料选择：

```text
终点是该材料首层 -> first_layer_travel_feed_mm_s
终点是该材料后续层 -> travel_feed_mm_s
```

两个空走速度默认都是 `10 mm/s`；external-NPZ 转换入口要求它们有限且大于 `0`。每条路径 reset + anchor 后，空走不产生挤出并保持零基线：

```text
delta_e = 0
e_val = 0
```

## 10. 偏置和最终导出

`convert_external_npz()` 在导出前读取共享偏置数据：

```text
data/head_calibration_offsets/head_offsets.json
```

读取后传给 exporter：

```text
tool_offset = (fiber_x_print_compensation_mm,
               fiber_y_print_compensation_mm,
               fiber_z_print_compensation_mm)

resin_z_print_compensation_mm = resin_z_print_compensation_mm
```

最终调用：

```text
export_npz(
    commands,
    output_path,
    dt=params.dt,
    default_feed_mm_s=params.travel_feed_mm_s,
    corner_angle_deg=params.corner_angle_deg,
    corner_retreat_ratio=params.corner_retreat_ratio,
    density=params.density,
    degree=params.degree,
    max_fit_points_per_segment=params.max_fit_points_per_segment,
    enable_extrude_wait=True,
    tool_offset=tool_offset,
    resin_z_print_compensation_mm=resin_z_print_compensation_mm,
)
```

`first_layer_travel_feed_mm_s` 只用于 preprocessor 生成的定位和路径间 `G0`，不会传入 exporter。剪切抬升、工具偏置切换等安全动作继续读取 `default_feed_mm_s=params.travel_feed_mm_s`，默认 `10 mm/s`。

因此，工具切换补偿、树脂 Z 补偿、事件编码、采样和系统 NPZ 字段写入，都由 `path_processing_core.npz_exporter` 统一完成。工具切换补偿保持旧 exporter 顺序：安全抬升 `20 mm` -> 偏置补偿 travel -> 工具切换事件。

## 11. 默认路径

源模板目录：

```text
data/external_npz_preprocessor/source_npz_templates
```

输出路径为空时默认导出到：

```text
data/output_npz/<source_stem>/<source_stem>.npz
```

打印参数 JSON 保存到：

```text
data/external_npz_preprocessor/print_params.json
```

这些路径都基于项目共享的 `DEFAULT_DATA_ROOT` 推导，不在代码中写死工作区绝对路径。


## 12. 与 GCode 输入的关系

正式打印 UI 的源文件选择支持两条输入分支：

```text
.gcode / .gc / .g -> gcode_planner.gcode_parser -> path_processing_core.npz_exporter
.npz              -> external_npz_preprocessor   -> path_processing_core.npz_exporter
```

因此，`gcode_planner` 现在是 GCode 适配和预览兼容包，`external_npz_preprocessor` 是约定源 NPZ 适配包，二者不互相依赖。共享数据结构、B 样条拟合、采样、事件编码、偏置应用和系统 NPZ 写出都归属于 `path_processing_core`。
