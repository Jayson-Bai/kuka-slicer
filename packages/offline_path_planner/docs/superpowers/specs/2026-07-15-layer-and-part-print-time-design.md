# 当前层与整体零件预计打印时间设计

## 目标

在正式打印页同时显示两组独立的进度与时间：

1. 当前层进度条、当前层序号、当前层预计剩余时间。
2. 整个零件进度条、整体已用时间、整体预计剩余时间。

保持上一版约束：只计算 RSI 侧计划时间；空走、打印采样行和离线展开的 RSI 侧挤出等待计入；打印头事件等待与 ABORT 不计入；RSI UDP 实时线程不承担估算计算。

## 现状与问题

当前 NPZ timing sidecar 只有零件总时长和运动段信息。运行时的 `TrajectoryPoint` 只有单行 `planned_time_s`，因此可以计算整体剩余时间，但无法在不扫描未来队列的情况下知道当前层的结束时间。`system_manager_node` 的轨迹队列也可能只缓存一部分当前层，不能依赖队列末端推断层边界。

## 方案选择

采用“导出阶段生成每层时间边界，运行时 O(1) 查表”的方案。

- 运行时从 sidecar 读取每层的起始时间和结束时间，避免扫描队列，也不受队列水位影响。
- 不把完整的层索引表复制到每个 NPZ 行，避免增加大量重复数组；只在 `NpzLoader` 内存中保存轻量层表，并在读取行时附加当前层元数据。
- 不采用运行时根据队列中 layer 切换推断层结束时间，因为当前层后半段可能尚未进入 `system_manager_node` 队列，估计会随队列补充而跳变。

## 时间边界定义

导出器按 RSI 轨迹行建立层边界：

- `start_time_s`：该层第一条非事件轨迹行的 `planned_time_s`。
- 非最后一层的 `end_time_s`：下一层第一条非事件轨迹行的 `planned_time_s`。因此层间空走和层切换运动计入前一层的剩余时间，直到下一层真正开始。
- 最后一层的 `end_time_s`：零件 `total_planned_time_s`。
- `duration_s = max(0, end_time_s - start_time_s)`。
- 事件行不推进时间，也不单独创建层时间边界。

当前层计算：

```text
layer_elapsed = clamp(current.planned_time_s - layer.start_time_s, 0, layer.duration_s)
layer_remaining = layer.duration_s - layer_elapsed
```

整体计算继续使用已有时间轴：

```text
part_elapsed = clamp(current.planned_time_s, 0, total_planned_time_s)
part_remaining = total_planned_time_s - part_elapsed
```

## 数据流改动

### 1. Python 导出

`RsiTimingAccumulator` 增加按 `layer_index` 收集首行时间的能力，`summary()` 增加：

```json
"layers": [
  {
    "layer_index": 0,
    "start_time_s": 0.0,
    "end_time_s": 12.5,
    "duration_s": 12.5
  }
]
```

sidecar 的 `format` 和 `version` 保持不变，旧 sidecar 没有 `layers` 时只使当前层时间失效，不影响整体时间和轨迹加载。

### 2. C++ 轨迹管线

`NpzLoader` 以轻量字符串扫描方式读取 sidecar 的 `layers` 数字字段，不引入通用 JSON 依赖。按 `layer_index` 为 `NpzRow` 填充：

```text
planned_layer_start_time_s
planned_layer_total_time_s
planned_layer_time_valid
```

`QueueManager` 将这些字段传递至 `TrajectoryPoint`。缺层、重复层、非有限值、负时长或 sidecar 缺失时，当前层时间标记无效，整体时间仍按现有逻辑处理。

### 3. UI 状态聚合

`UiStatus` 增加：

```text
float32 planned_layer_total_time_s
float32 planned_layer_elapsed_time_s
float32 planned_layer_remaining_time_s
bool print_layer_time_valid
```

`system_manager_node` 在已有非实时 UI 定时器中，以默认 500 ms 更新当前层和整体缓存。新一轮打印、序号回退、整体总时长变化或 layer index 变化时立即刷新缓存；暂时无法对齐当前轨迹时复用最后有效值。

### 4. 正式打印 UI

将现有顶部打印进度区域改为上下两行：

```text
当前层 3 / 12  [当前层进度条]  本层剩余 00:02:18
整体 27%       [整体进度条]    总 01:02:30 | 已用 00:16:00 | 剩余约 00:46:30
```

两行只在正式打印模式显示，测试模式不显示。整体或当前层 timing 无效时分别显示 `时间估计 --` 或 `本层时间 --`，不影响已有层进度逻辑。

## 兼容性与降级

- 旧 NPZ 没有 `planned_time_s` 或旧 sidecar 没有总时长：整体和当前层时间均无效，但轨迹仍可正常加载。
- 旧 sidecar 有总时长但没有 `layers`：整体时间继续显示，当前层时间显示无效。
- sidecar 层表格式错误只影响当前层时间，不阻断正式打印。
- 不修改 ABORT 行为，也不把打印头事件等待加入时间轴。

## 测试范围

- Python accumulator：多层首行时间、层边界、层间时间、最后一层结束时间和 JSON 可序列化。
- NPZ exporter：sidecar 包含 `layers`，单文件与拆分输出路径正确，事件不推进层时间。
- C++ source/build contract：消息字段、loader 层表读取、缺失层降级和 QueueManager 转发。
- UI contract：双进度条、双时间标签、正式模式可见、测试模式隐藏、格式化和无效状态。
- 构建受影响 ROS 2 包并运行已有相关测试。
