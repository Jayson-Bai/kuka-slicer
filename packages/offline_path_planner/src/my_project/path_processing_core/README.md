# path_processing_core 总说明

## 包定位

`path_processing_core` 是系统路径处理的共享核心包。它负责保存系统 NPZ 导出所需的通用数据结构、事件映射、曲线拟合、时间参数化、喷头偏置读取和最终 NPZ 写出逻辑。

当前约定是：所有能被正式打印、测试打印或预览加载的系统 NPZ，最终都必须通过 `path_processing_core.npz_exporter.export_npz()` 写出。上游输入可以是 GCode，也可以是约定格式的源 NPZ，但它们都需要先转换成 `path_processing_core.types` 中定义的命令序列。

## 核心模块

- `types.py`：定义 `Position`、`MoveCommand`、`ToolChangeCommand`、`MCommand`、`ResetECommand`、`ExtrudeWait`、`GlobalCurveCommand` 等共享命令和轨迹结构。
- `npz_exporter.py`：系统 NPZ 的唯一核心导出实现，负责命令分段、短线段折线合并、B 样条拟合调用、采样、事件写入、offset sidecar 写入和 layer preview 元数据写入。
- `head_calibration.py`：读取和保存共享喷头偏置数据，数据源位于 `data/head_calibration_offsets/head_offsets.json`。
- `bspline_approximation.py`：上层 B 样条拟合前处理，包括角点回退、点密度加密和控制点组织。
- `polynomial_interpolator.py`：对直线和拟合曲线做时间参数化与采样，输出带绝对 E 的采样点。
- `bspline/`：底层 B 样条曲线/曲面算法实现。

## 输入链路

```text
约定格式源 NPZ
-> external_npz_preprocessor
-> path_processing_core.types 命令序列
-> path_processing_core.npz_exporter.export_npz()
-> 系统 NPZ
```

```text
GCode
-> gcode_planner.gcode_parser
-> path_processing_core.types 命令序列
-> path_processing_core.npz_exporter.export_npz()
-> 系统 NPZ
```

`gcode_planner` 中保留的 `npz_exporter.py`、`types.py`、`head_calibration.py`、`bspline_approximation.py`、`polynomial_interpolator.py` 和 `bspline/*` 现在是兼容旧 import 路径的包装入口，真实实现归属于本包。

## 系统 NPZ 输出

核心 exporter 写出的 NPZ 包含轨迹、挤出、工具、事件和预览分层相关字段。常用字段包括：

```text
seq, x, y, z, a, b, c, e
tool_id, move_type, src_line
event_flag, event_type, payload, trigger_seq
preview_layer_index, layer_index
move_type_vocab_keys, move_type_vocab_vals
event_type_vocab_keys, event_type_vocab_vals
```

其中 `preview_layer_index` 是三维预览优先使用的显示层编号；`layer_index` 保留为物理/解析层信息。这样可以避免同一物理 Z 或同一旧层编号导致阀座等短段在预览中被错误合并。

## 拟合与短线段采样

exporter 会先按连续同类型、同层、同子类型的 `MoveCommand` 分段。普通长路径仍优先进入 B 样条拟合；已有的 wall outline 路径继续作为原始折线 `POLYLINE` 输出，避免样条超出原始包围盒。

当分段器识别到连续的过短打印线段簇时，3 段及以上的短簇会被合并为单条 `POLYLINE` 采样。该处理只改变时间参数化的承载曲线，不重排点、不做 B 样条平滑，也不改变总 `delta_e`；折线控制点仍是原始 GCode/源 NPZ 的每个终点。这样可以避免每个极短线段都独立触发固定加减速时间，在树脂路径起点、终点或细碎点附近造成长时间停滞。1 到 2 段的强制线性回退仍按原来的单段直线逻辑输出。

## 事件与偏置

`npz_exporter` 统一处理以下事件：

- 工具切换：`tool_change_cf`、`tool_change_resin`
- 加热：`heat_cf`、`heat_resin`
- 风扇：`fan_cf`、`fan_resin`
- 挤出重置：`extrude_reset`
- 原地预挤出/回抽：通过 `ExtrudeWait` 转成可采样的挤出等待段
- 纤维剪切：`cut`

`CUT` 命令在 exporter 中统一展开：先在当前纤维路径末端写入 `cut` 事件，再继续导出 Z 向抬升段。默认 `cut_lift_mm=20.0`，抬升过程中 Z 增加同样距离，E 也增加同样数值；若抬升耗时短于 `cut_wait_s`，则在高位补足剩余等待；随后在高位做等量安全回抽。之后的正式路径回抽、预挤出和 travel 仍按原命令顺序执行，第一段真实运动会从当前高位姿态重新连接。

喷头偏置也集中在 exporter 路径中应用。默认初始工具写死为系统树脂工具 `2`。发生工具切换且存在纤维头偏置时，exporter 保持旧顺序：先安全抬升 `20 mm`，再执行喷头 XYZ 偏置补偿 travel，最后写入 `tool_change_cf/tool_change_resin` 事件。preprocessor 和 UI 只负责读取或传入共享偏置参数，不重复实现切换工具时的补偿数学。

## 维护原则

- 新的非 GCode 输入适配器应依赖 `path_processing_core`，不要依赖 `gcode_planner` 的兼容包装模块。
- 旧代码如果仍从 `gcode_planner.*` import，可以继续工作，但新代码应直接使用 `path_processing_core.*`。
- 修改 exporter、类型、拟合或采样逻辑后，需要同时跑 core 兼容测试、GCode 导出测试和 external NPZ preprocessor 测试。
