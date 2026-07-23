# 正式打印预计剩余时间设计

## 目标

在正式打印页面显示基于系统 NPZ 实际导出结果的预计总时长、已执行时长和剩余时长。时间基准采用 RSI 轨迹序号推进，不把打印头事件等待和 ABORT 收尾纳入估计。

## 范围和非目标

### 纳入范围

- `PRINT`、`TRAVEL`/空走、加减速采样帧、挤出等待展开帧、剪切抬升等已经写入系统 NPZ 的轨迹帧。
- NPZ 导出时使用的 `dt`、目标速度、加速时间、匀速时间和减速时间所生成的实际采样结果。
- 正式打印运行中由 `/rsi/heartbeat.seq_used` 表示的实际 RSI 执行位置。
- UI 低频显示时间状态；暂停、RSI 暂停推进或轨迹队列停顿时，剩余计划时间不继续倒计时。

### 不纳入范围

- 换刀、加热、挤出复位等打印头事件的真实等待时长。
- ABORT 触发后的 Z 轴抬升和通信收尾。
- 修改 RSI UDP 线程的控制逻辑或改变轨迹发送节拍。
- 根据历史打印数据建立复杂的预测模型。

## 设计原则

1. 导出阶段计算，运行阶段读取。运行时不重新拟合 B 样条、不重新计算路径长度、不扫描 NPZ 文件。
2. 以实际写入的轨迹帧为准，而不是在 C++ 端根据行数或当前启动参数重新猜测时间。这样可以保留短路径退化、`ceil(total_time / dt)` 和曲线首帧等现有采样语义。
3. 事件行不作为运动轨迹帧计入计划时间；事件等待本身按本设计明确忽略。事件之后实际写入的 TRAVEL/PRINT/等待展开轨迹帧仍计入。
4. 保持旧 NPZ 可加载。旧 NPZ 没有时间字段时，UI 显示时间估计不可用，不伪造高精度剩余时间；重新导出后自动获得完整时间信息。
5. 新增字段采用可选兼容方式，所有默认导出路径自动生成时间数据。

## 数据流

```text
export_npz()
  ├─ 按现有 sample_global_curve_iter() 生成轨迹行
  ├─ 为每个轨迹行写 planned_time_s（累计 RSI 计划时间）
  └─ 写 <npz-base>.timing.json（总时长和模型摘要）

NpzLoader
  ├─ 读取 planned_time_s
  └─ 读取 timing.json 的 total_planned_time_s

QueueManager -> TrajectoryPoint
  └─ 转发 planned_time_s / planned_total_time_s / valid

SystemManagerNode（低频估计更新）
  └─ current = trajectory.planned_time_s
     remaining = max(total - current, 0)

UiStatus -> RQT UI
  └─ 显示总时长、已用时长、预计剩余
```

## NPZ 时间数据格式

系统 NPZ 新增可选数组：

```text
planned_time_s: float32[N]
```

该数组长度与同一 NPZ 分片的 `seq` 数组一致。非事件轨迹行的值表示该行被 RSI 使用时在计划轨迹时间轴上的累计时间；事件行重复前一条轨迹时间，不增加计划时间。第一条轨迹行时间为 `0`；之后每使用一条轨迹行增加一个 `dt`。因此它反映的是 RSI 实际按帧消费 NPZ 的时间轴，包含空走和导出的等待展开帧，但不包含事件阻塞等待。

导出器同时写一个小型 sidecar：

```text
<npz-base>.timing.json
```

内容包含：

- `format`、`version`、`time_model`；
- `sample_period_s`；
- `total_planned_time_s`；
- `trajectory_rows`、`event_rows_ignored`；
- 每个采样路径的 `path_id`、类型、起止序号、累计时长、`t_acc_s`、`t_flat_s`、`t_dec_s`。

sidecar 只用于总时长和可追溯说明，实时循环不读取或解析它。拆分 NPZ 时 sidecar 与 manifest 同目录并由 `NpzLoader` 根据输入路径解析。

## ROS 消息扩展

`TrajectoryPoint.msg` 新增：

```text
float32 planned_time_s
float32 planned_total_time_s
bool planned_time_valid
```

`UiStatus.msg` 新增：

```text
float32 planned_total_time_s
float32 planned_elapsed_time_s
float32 planned_remaining_time_s
bool print_time_valid
```

所有缺少 NPZ 时间字段的旧数据，`planned_time_valid=false`，不影响原有轨迹发送。

## 运行时计算

`NpzLoader` 和 `QueueManager` 只做字段读取和转发。`SystemManagerNode` 在已有 UI 状态定时器中维护一个独立的低频时间更新时间点，默认每 `500 ms` 更新一次时间估计；其他系统状态保持现有发布频率。

每次更新时间估计时：

```text
elapsed   = current_traj.planned_time_s
total     = current_traj.planned_total_time_s
remaining = clamp(total - elapsed, 0, total)
```

当前轨迹序号未在 UI 队列中对齐时，保持上一次有效时间值；没有有效时间数据时显示 `--`。RSI 心跳丢失、暂停或队列停顿不会人为扣减剩余时间，因为 `planned_time_s` 不会推进。

## UI 展示

正式打印页顶部现有层进度旁增加：

```text
总时长 01:23:45   已用 00:12:08   剩余约 01:11:37
```

时间显示只在正式打印模式显示。无时间数据时显示“时间估计 --”；不会影响测试模式和现有层进度条。

## 错误处理和兼容性

- sidecar 缺失但 NPZ 有 `planned_time_s`：可以显示已用时间；总时长标记无效并显示 `--`。
- NPZ 缺少 `planned_time_s`：原有轨迹照常运行，时间估计整体标记无效。
- 时间数组长度不等于 `seq` 长度或出现非有限值：记录警告，禁用时间估计，不阻断打印。
- sidecar JSON 损坏：记录警告，仍允许轨迹加载。
- 旧 ROS 节点不理解新增消息字段时，由 ROS 2 类型支持处理版本一致性；本仓库内所有相关消息构造和测试同步更新。

## 测试策略

- 导出器：验证 PRINT、TRAVEL、等待展开行的累计时间，事件行不增加时间，路径元数据包含加速/匀速/减速字段。
- NPZ 加载：验证新字段读回、sidecar 总时长读回、缺失字段的兼容行为。
- 消息/聚合：验证 `seq_used` 对应当前轨迹时能得到正确 elapsed/remaining，暂停或停顿时数值保持不变。
- UI：验证正式打印页显示时间、无效数据显示 `--`，测试模式不显示时间栏。
- 回归：运行现有 Python 测试、接口消息构建和相关 ROS 2 包编译。

## 明确假设

- RSI 轨迹消费周期仍为当前约定的 `4 ms`，时间轴由导出时的 `dt` 写入，不在运行时读取新的速度参数。
- 用户选择的“剩余时间”是 RSI 计划轨迹剩余时间，不是包含打印头事件和 ABORT 的完整墙钟完成时间。
- 任何实际运行中无法从 NPZ 和 RSI 序号确定的停顿都不被强行预测。
