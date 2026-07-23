# external_npz_preprocessor 总说明

## 包定位

`external_npz_preprocessor` 用于把约定格式的外部源 NPZ 转换成系统可加载的 NPZ。它不直接写系统 NPZ，而是把源 NPZ 中的分层树脂/纤维路径转换成 `path_processing_core.types` 命令序列，再调用 `path_processing_core.npz_exporter.export_npz()` 导出。

这个包是系统未来主要的非 GCode 输入入口。正式打印 UI 已支持兼容输入：选择 `.npz` 时走本包，选择 `.gcode/.gc/.g` 时走 `gcode_planner` 的 GCode 适配路径；两条路径最终都汇入 `path_processing_core`。

## 源 NPZ 格式

源文件是标准 `.npz`，按层和材料分 key：

```text
meta
layer_0000_R
layer_0000_F
layer_0001_R
layer_0001_F
```

每个 `layer_xxxx_R/F` 是 `float32` 三维数组：

```text
[path_count, max_points_per_path, columns]
```

短路径用整行 `NaN` padding。有效点只支持显式带 Z 的格式：

```text
Nx3: [x, y, z]
Nx6: [x, y, z, a, b, c]
```

源 Z 是轨迹几何真值，preprocessor 不生成也不覆盖 Z。UI 中的树脂/纤维层高只作为挤出和工艺参数参考。

## 参数与持久化

UI 中的打印参数可通过 `保存打印参数json文件` 保存到：

```text
data/external_npz_preprocessor/print_params.json
```

启动 UI 时会自动读取该文件。所有路径都基于项目共享 data 根目录推导，不在代码中写死工作区绝对路径。

`prime_settle_s` 是一项全局 external-NPZ 参数，默认 `0.5 s`。它会持久化到上述 `print_params.json`，并由 CLI、独立 UI 和正式打印 UI 暴露；旧 JSON 缺少该字段时使用 `0.5 s`。设置为 `0` 只关闭预挤出后的稳定等待，负数、`NaN` 和无穷值会被拒绝。`travel_feed_mm_s`、`first_layer_travel_feed_mm_s` 以及两个材料的 `first_layer_feed_mm_s` 在 external-NPZ 转换入口必须是有限且大于 `0` 的值。

主要默认值：

- 树脂固定线宽：`2.0 mm`，不允许用户输入
- 树脂耗材直径：固定 `1.75 mm`，用于把沉积体积换算成 E 轴料长
- 树脂层高：`0.5 mm`，用于树脂挤出计算
- 纤维层高：`0.1 mm`，作为工艺参考
- 首层树脂打印速度：`10 mm/s`
- 首层纤维打印速度：`10 mm/s`
- 首层共享空走速度：`10 mm/s`
- 非首层树脂/纤维打印速度：`10 mm/s`
- 非首层空走速度：`10 mm/s`
- 全局预挤出稳定等待：`0.5 s`
- 树脂/纤维温度：`250 C`
- 树脂/纤维风扇：默认常开
- 纤维默认挤出倍率：`1.0`，表示纤维进给速度与 TCP 移动速度一致
- 纤维起步加速时间：`2.0 s`，只覆盖外部 NPZ 纤维打印路径的七阶起始加速段

树脂首层和纤维首层按各材料第一次实际出现的层独立判定；擦料线使用首层树脂打印速度。空走速度按终点路径所属材料是否位于其首层选择。首层参数不传入剪切抬升、工具偏置切换等 exporter 安全动作，这些动作继续使用 `default_feed_mm_s=params.travel_feed_mm_s`，其默认值为 `10 mm/s`。

共享喷头偏置读取自：

```text
data/head_calibration_offsets/head_offsets.json
```

## 导出流程

```text
source NPZ
-> load_source_npz()
-> SourceJob / LayerPaths / MaterialPath
-> source_job_to_parsed_commands()
-> ToolChangeCommand / MCommand / ResetECommand / ExtrudeWait / MoveCommand
-> path_processing_core.npz_exporter.export_npz()
-> system NPZ
```

preprocessor 会在转换开始处插入双喷头风扇/加热事件，并保留工具切换后的既有挤出量重置。到达每条可打印路径起点后，统一执行 `prime -> 可选 prime_settle -> PRINT`；整件第一条可打印路径还会在 prime 前执行一次既有初始回抽。每条路径的结束边界为：

```text
树脂: PRINT -> normal retract -> path reset -> one-cycle E=0 anchor -> optional travel(E=0)
纤维: PRINT -> CUT/lift/wait/safety retract -> path reset -> one-cycle E=0 anchor -> optional travel(E=0)
```

这套规则覆盖转换器生成的 primeline、所有源路径和没有后续 travel 的最终路径。每个路径边界后 converter-side `current_e` 都设为 `0`，下一条路径在到达起点后从零计算自己的 prime。travel 命令不携带 prime/retract，导出的 travel 行保持 `E=0`。剪切展开、事件编码、短线段折线连续采样、偏置补偿和 NPZ 字段写入仍由 `path_processing_core.npz_exporter` 统一完成。

该更新只改变 external-NPZ adapter/exporter 的命令与导出边界；外部源 NPZ schema、GCode 行为以及 RSI/UART 协议语法均未改变。

## 模板与测试

两层 R/F 源 NPZ 模板位于：

```text
data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz
```

该模板使用数值数组和 `NaN` padding，可用 `np.load(..., allow_pickle=False)` 读取，不包含序列化 Python object。
