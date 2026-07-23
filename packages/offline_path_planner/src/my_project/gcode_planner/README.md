# gcode_planner 使用说明

## 当前定位

`gcode_planner` 现在是 GCode 输入适配包和旧接口兼容包，不再拥有系统 NPZ 导出的底层实现。它负责读取 GCode、解析成 `path_processing_core.types` 命令序列，并调用 `path_processing_core.npz_exporter.export_npz()` 生成系统 NPZ。

当前职责：

- GCode CLI：`ros2 run gcode_planner gcode_planner_npz ...`
- GCode 解析：`gcode_planner.gcode_parser`
- 旧 import 兼容：`gcode_planner.types`、`gcode_planner.npz_exporter`、`gcode_planner.bspline_approximation`、`gcode_planner.polynomial_interpolator`、`gcode_planner.head_calibration` 和 `gcode_planner.bspline.*`
- 三维/分层预览辅助：`gcode_planner.path_preview`
- 测试模式临时 GCode/NPZ 生成：`gcode_planner.print_test_generator`

真实共享实现位于 `path_processing_core`。新代码应优先直接依赖 `path_processing_core`，只有 GCode 输入路径和旧代码兼容场景继续使用 `gcode_planner`。

## 快速开始

```bash
cd <kuka_ram_ws>
colcon build --packages-select path_processing_core gcode_planner
source install/setup.bash
```

离线导出系统 NPZ：

```bash
ros2 run gcode_planner gcode_planner_npz \
  --gcode /path/to/file.gcode \
  --out /path/to/output.npz
```

如果不显式传 `--gcode`，CLI 会从 `data_root/input_gcode` 中按文件名选择首个 `.gcode` 文件。通过 `ros2 run` 运行安装后的包时，建议显式传 `--data-root` 或 `--output-dir`，避免默认路径落到安装目录内。

## CLI 参数

参数来自 `gcode_planner/cli.py`：

- `--gcode`：GCode 文件路径；缺省时从输入目录选择首个 `.gcode`
- `--data-root`：数据根目录；默认推导为工作空间下的 `data`
- `--input-gcode-dir`：GCode 输入目录；默认 `data_root/input_gcode`
- `--output-dir`：NPZ 输出目录；默认 `data_root/output_npz`
- `--out`：输出 NPZ 文件路径，优先级最高
- `--dt`：采样周期秒，默认 `0.004`
- `--default-feed-mm-s`：GCode 缺失或无效 F 值时的兜底速度
- `--corner-angle-deg`：角点判定夹角阈值
- `--corner-retreat-ratio`：角点回退比例
- `--density`：拟合前点加密密度
- `--degree`：B 样条阶次
- `--max-fit-points-per-segment`：单段拟合点数上限
- `--export-sleep-ms` / `--export-yield-every`：大文件导出节流参数
- `--split-by-layer-type`：按层和打印子类型拆分导出
- `--plot-layer-xy` / `--plot-stride`：导出分层 XY 预览图
- `--cut-lift-mm`：纤维 `CUT` 事件后的 Z 向抬升距离，默认 `20.0`
- `--cut-wait-s`：从 `cut` 事件触发开始计算的剪切等待总时长，默认 `15.0`

路径优先级：

```text
输出: --out > --output-dir > data_root/output_npz
输入: --gcode > --input-gcode-dir > data_root/input_gcode
```

## GCode 解析规则

基础规则：

- 行内注释以 `;` 截断。
- 支持 `G0/G1`、`G90/G91`、`M82/M83`、`G92 E...`、`Tn`、`M104/M109`、`M106/M107`。
- `G0` 标记为空走，`G1` 在 `delta_e != 0` 时标记为打印，否则为空走。
- 纯挤出、无位移的指令会被识别为 `ExtrudeWait`。
- 默认坐标模式为绝对，默认挤出模式为绝对。

工具映射在导出阶段统一处理：

```text
GCode T0 -> 系统工具 1（纤维）
GCode T1 -> 系统工具 2（树脂）
```

默认导出工具写死为系统树脂工具 `2`。工具切换补偿沿用共享 exporter 逻辑：先安全抬升 `20 mm`，再执行喷头 XYZ 偏置补偿 travel，最后写入工具切换事件。

## 事件映射

`gcode_parser` 只把 GCode 语义转换为共享命令，事件最终由 `path_processing_core.npz_exporter` 写入系统 NPZ：

- `T0/T1` -> `tool_change_cf` / `tool_change_resin`
- `M104/M109 S...` -> `heat_cf` / `heat_resin`
- `M106/M107` -> `fan_cf` / `fan_resin`
- `G92 E...` -> `extrude_reset`
- 外部命令 `CUT` -> `cut`，并由共享 exporter 展开为非阻塞剪切事件、抬升、等待和安全回抽序列

## 拟合、采样和 NPZ 导出

GCode 路径进入 exporter 后的处理位于 `path_processing_core`：

1. `path_processing_core.npz_exporter` 按连续同类型/同层/同子类型的 `MoveCommand` 分段。
2. `path_processing_core.bspline_approximation` 对打印/空走段做角点回退、点加密和 B 样条控制点生成。
3. `path_processing_core.polynomial_interpolator` 按 `dt` 做时间参数化与采样。连续过短线段簇会作为原始折线 `POLYLINE` 连续采样，避免每个短段独立加减速导致局部停滞；普通长路径仍走 B 样条拟合。
4. exporter 写出系统 NPZ 字段、事件 vocab、offset sidecar 和预览分层字段。

输出常用字段：

```text
seq, x, y, z, a, b, c, e
tool_id, move_type, src_line
event_flag, event_type, payload, trigger_seq
preview_layer_index, layer_index
move_type_vocab_keys, move_type_vocab_vals
event_type_vocab_keys, event_type_vocab_vals
```

当启用 `--split-by-layer-type` 时，会生成 manifest，并按原 GCode 顺序播放拆分后的文件。

## E 挤出链路

GCode 输入时，E 以绝对挤出量流转：

1. `gcode_planner.gcode_parser` 根据 `M82/M83` 解析绝对或相对 E，生成 `MoveCommand.e_val` 和 `delta_e`。
2. `path_processing_core.polynomial_interpolator` 按路径弧长比例分配 `delta_e`，输出采样后的绝对 E。
3. `path_processing_core.npz_exporter` 写入 NPZ 的 `e` 字段，并把 `G92 E...` 写成 `extrude_reset` 事件。
4. `control_center` 从 NPZ 读取 `e`，经 `/planned_trajectory` 和 `/rsi/heartbeat` 传递给 `uart_node`。
5. `uart_node` 按 `E <seq> <tool_id> <extrude_abs>` 文本协议发送到串口。

## 三维预览

`gcode_planner.path_preview` 仍是 UI 预览使用的兼容模块。现代系统 NPZ 的分层显示优先使用 `preview_layer_index`，其次使用 `layer_index`，最后才回退到旧的 PNG/物理 Z 推断。为避免超大 NPZ 让 rqt 卡死，预览读取和 VTK 绘制都设置了采样/点数上限。

## 模块说明

- `cli.py`：GCode 离线导出 CLI。
- `gcode_parser.py`：GCode 解析和可选 ROS2 节点封装。
- `path_preview.py`：系统 NPZ 预览数据提取。
- `print_test_generator.py`：测试模式临时 GCode/NPZ 生成。
- `types.py`、`npz_exporter.py`、`head_calibration.py`、`bspline_approximation.py`、`polynomial_interpolator.py`：旧 import 兼容包装，真实实现位于 `path_processing_core`。
- `bspline/`：旧 `gcode_planner.bspline.*` import 兼容包装，真实实现位于 `path_processing_core.bspline`。
