# 上位机侧 Core NPZ 局部注入交接文档

版本：`core_npz_local_injection_v1`  
适用分支：`merged_core`  
目标：上位机现场修改机械参数时，只对最终 Core NPZ 的局部区域进行注入，不重新执行 Prusa 切片，也不重新运行完整 Core。

## 1. 上位机侧只需要做什么

上位机侧只需要完成以下工作：

1. 读取 UI 导出的最终 Core NPZ；如果导出被分块，则按 `*_part0000.npz`、`*_part0001.npz` 的顺序读取全部分块。
2. 读取 NPZ 内嵌的 `core_injection_manifest`。
3. 根据 `core_injection_block_id` 和 `core_injection_role` 找到工具切换块、CUT 块以及树脂 Z 补偿锚点。
4. 对全局偏置参数做数组级局部变换。
5. 对工具切换和 CUT 的标记区域进行局部重建，并使用与 Core 相同的 `dt` 和七阶参数化采样。
6. 修复局部重建导致的后续 `seq`、`trigger_seq` 和 `planned_time_s`。
7. 校验连续性后，把结果原子替换回原来的 NPZ 路径。

这里的“原子替换”可以使用同目录临时文件后 `replace`。最终仍只保留一个 NPZ，不需要额外的 manifest、参数或注入结果文件。

## 2. 明确不需要做什么

上位机侧不要重复执行以下工作：

- 不重新调用 Prusa 切片内核。
- 不重新读取 STL 或重新生成树脂/纤维路径。
- 不重新运行完整 `path_processing_core.export_npz()`。
- 不重新做全部路径平滑、B 样条拟合或路径重排。
- 不重新规划 Prusa 已经提供的树脂层内空走。
- 不重新计算或重新分配普通打印路径的 E。
- 不修改 Prusa 预览逻辑。
- 不改变树脂优先、纤维后置的打印顺序。
- 不删除每条路径结束后的 E reset。
- 不把 `path_id` 当作唯一的路径定位依据；注入定位必须使用新的 block marker。

普通打印路径的几何、Prusa 导出的 E、路径顺序和原有 reset 都应直接保留。

## 3. 必须读取的 NPZ 字段

### 3.1 必须字段

```text
seq
x, y, z, a, b, c
e
tool_id
move_type
event_flag, event_type, payload, trigger_seq
layer_index, preview_layer_index
path_id, path_end_flag
planned_time_s

core_injection_manifest
core_injection_block_id
core_injection_role
core_injection_role_vocab_keys
core_injection_role_vocab_vals
```

建议使用：

```python
with np.load(path, allow_pickle=False) as data:
    manifest = json.loads(str(data["core_injection_manifest"].item()))
```

`core_injection_manifest` 是 0-D JSON 字符串，不是 pickle 对象。

### 3.2 不需要作为注入输入的文件

当前新格式下，局部注入不需要依赖：

- Prusa 原始 GCode；
- Prusa 源 NPZ；
- UI 预览文件；
- `.offset.json`；
- `.timing.json`；
- STL 文件；
- 完整 Core 参数配置。

`.offset.json` 和 `.timing.json` 可以保留用于诊断，但不能作为新的注入定位依据。

## 4. 内嵌标记含义

`core_injection_block_id` 是全局块编号，`-1` 表示普通行。  
`core_injection_role` 的当前值如下：

| 数值 | 名称 | 含义 |
|---:|---|---|
| 0 | `none` | 普通运动行 |
| 1 | `tool_change_pre` | 工具切换前的安全抬升/偏置 travel |
| 2 | `tool_change_event` | 工具切换事件行 |
| 3 | `tool_change_post` | 工具切换后的连接 travel |
| 4 | `cut_event` | CUT 事件行 |
| 5 | `cut_action` | CUT 内部的 reset、抬升、等待、回抽等行 |
| 6 | `cut_post` | CUT 后的连接 travel |
| 7 | `post_anchor` | 下一条有效打印路径的首个锚点行 |
| 8 | `resin_z_compensation` | 树脂 Z 补偿 travel 行 |

每个分块 NPZ 都包含相同的 manifest 和 role vocabulary；行 marker 只对应本分块中的行，但 block ID 和 `seq` 是全局的。

## 5. 允许现场注入的参数

当前建议只开放以下参数：

```text
tool_offset = [fiber_x, fiber_y, fiber_z]
resin_z_print_compensation_mm
tool_change_safe_lift_mm
cut_lift_mm
cut_wait_s
```

### 5.1 工具偏置 `tool_offset`

工具偏置有两部分：

1. 对所有 `tool_id == 1` 的纤维轨迹行，按“新偏置 - 导出时偏置”做 XYZ 平移；
2. 对每个 `tool_change` block，重建 `tool_change_pre` 区域，使安全抬升和工具偏置 travel 的终点与新的纤维坐标一致。

`tool_change_pre` 不应再次套用普通纤维全局平移，否则会发生偏置重复计算。事件行和后续连接点必须以重建后的块末端为起点。

### 5.2 树脂 Z 补偿 `resin_z_print_compensation_mm`

manifest 中始终有一个 `resin_z_compensation` block。即使导出时补偿值为 0、没有实际 marker 行，也必须使用该 block 的：

```text
anchor = before_first_effective_print_path
```

进行插入或替换。

树脂 Z 补偿的作用是：

- 替换/插入首条有效打印路径前的 Z 补偿 travel；
- 对该锚点之后的轨迹行按新旧补偿差值统一调整 Z；
- 不改变 X/Y/A/B/C/E。

### 5.3 安全抬升 `tool_change_safe_lift_mm`

只重建 `tool_change_pre` 区域。不要重新平滑整条打印路径。该区域通常是直线运动，但仍必须使用原 Core 的 `dt` 和七阶参数化采样。

### 5.4 CUT 抬升和等待

只重建对应 `cut` block 的 `cut_event`、`cut_action` 和 `cut_post` 区域。

对于外部 NPZ 纤维 CUT，必须保持现有顺序：

```text
E reset anchor
→ cut event
→ Z 抬升，同时 E 增加相同长度
→ 高位等待
→ E reset
→ 等量回抽
→ 高位等待
→ E reset
→ 剩余 cut_wait
→ 后续连接 travel
```

`cut_lift_mm` 和 `cut_wait_s` 改变后，局部采样点数量可能变化；这时需要重建该 block，并修复其后的全局序号和时间，不允许只修改某几个 Z 值而留下 RSI 断点。

## 6. 局部重建要求

### 6.1 采样

- 采样周期使用 manifest 中的 `sample_period_s`，当前通常为 `0.004 s`；
- 直线块不需要 B 样条平滑；
- 仍必须使用 Core 现有的七阶参数化采样；
- 不要使用新的速度规划规则；层内空走和层间抬升继续使用 Core 导出的速度约定；
- 新行的 `move_type`、`event_flag`、`event_type`、`payload` 要和原 Core 语义一致。

### 6.2 E 与事件

- 普通树脂/纤维打印行的 E 不得重新计算；
- 每条路径原有的 reset 不得删除或合并；
- CUT 内部的 E 只按现有 CUT 展开规则生成；
- `event_flag`、`event_type`、`payload`、`trigger_seq` 必须保留；
- 不能用额外的假运动行替代事件行。

### 6.3 RSI 连续性

保存时必须满足：

```text
seq[0] >= 0
seq[i+1] == seq[i] + 1
所有 XYZABC/E 均为有限值
所有 event 的 trigger_seq 指向对应的新 seq
后一个运动块的起点 == 前一个块的终点
path_end_flag 仍只标记每条路径的最后一个采样行
```

如果局部块新增或删除采样行：

- 该块之后所有行重新编号；
- 该块之后所有事件的 `trigger_seq` 同步偏移；
- `planned_time_s` 按局部新增/减少的运动时间整体平移；
- 不需要重新拟合后续路径，只需要做索引和累计时间修复。

## 7. 推荐的上位机处理流程

```text
读取一个 NPZ 或全部 NPZ 分块
    ↓
校验 manifest/schema/seq
    ↓
读取现场参数并计算相对导出值的 delta
    ↓
应用 tool_offset 和 resin-Z 的全局数组变换
    ↓
按 block_id 重建 tool_change/CUT 局部块
    ↓
接回前后姿态，修复 E、event、seq、trigger_seq、planned_time_s
    ↓
做连续性校验
    ↓
同路径原子替换 NPZ
```

不要把所有行重新送回 Core；这会重新触发全部路径拟合和采样，失去现场调参的时间优势。

## 8. 耗时目标

已有测试参考：

| 操作 | 参考耗时 |
|---|---:|
| 完整 Core 重新导出 | 约 50–56 s |
| 局部注入原型（含 CUT/等待局部重建） | 约 4.5–5.0 s |
| 只做同长度的 XYZ/Z 数组变换 | 目标小于 1 s |

上位机实现应以“局部数组变换 + 局部块重建”为原则，目标保持在约 5 s 以内；不能因为现场修改一个偏置而重新执行完整 Core。

## 9. 当前状态

当前 UI 导出的 Core NPZ 已自动包含上述 manifest 和行级 marker，旧消费者忽略这些新字段即可继续读取原有轨迹字段。上位机侧下一步只需要实现本交接文档定义的 local injector，不需要修改 Prusa 预览或重新设计 Core 输出格式。
