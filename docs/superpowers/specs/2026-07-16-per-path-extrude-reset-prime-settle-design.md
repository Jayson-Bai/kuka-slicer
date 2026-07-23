# 外部 NPZ 每路径挤出复位与预挤出稳定等待设计

## 状态

- 日期：2026-07-16
- 分支：`feature/uart-canonical-v1`
- 用户确认：每条路径收尾后复用现有 `extrude_reset`；新增全局
  `prime_settle_s`，默认 `0.5 s`

## 目标

只调整正式外部 NPZ 预处理链的路径边界：

1. 每条打印路径完成自己的收尾动作后，发送现有阻塞式
   `extrude_reset`，丢弃该路径尚未执行完的正向或反向 E 目标。
2. reset 完成后将上位机后续绝对 E 同步从零重新累计。
3. travel 全程保持 `E=0`；下一条路径到达起点后再原地预挤出。
4. 每次预挤出 E 斜坡完成后，固定 XYZ 和最终 E 再保持
   `prime_settle_s`，默认 `0.5 s`，然后才开始打印。

## 非目标

本次不修改：

- UART ASCII 语法、`canonical_v1`/`legacy_v1`、去重和串口重试边界；
- RSI 4 ms 主循环、心跳消息、暂停、恢复、终止或事件状态机；
- 控制中心轨迹发布和自动收尾逻辑；
- MCU 固件及现有 `extrude_reset` 语义；
- 普通 GCode 解析和 G92 行为；
- 打印测试矩阵生成器、手动挤出和其他非外部 NPZ 链路；
- 预挤出/回抽长度和速度的现有配置值。

## 路径时序

树脂路径采用：

```text
PRINT
  -> RETRACT（固定 XYZ，E 下降）
  -> extrude_reset（阻塞，RSI 保持路径末端）
  -> RESET_ANCHOR（固定 XYZ，E=0，内部单周期同步点）
  -> TRAVEL（E=0）
  -> PRIME（到达下一路径起点后，固定 XYZ，E 从 0 上升）
  -> PRIME_SETTLE（固定 XYZ 和最终 E，默认 0.5 s）
  -> 下一条 PRINT
```

纤维路径采用：

```text
PRINT
  -> CUT（含现有抬升、等待和安全回抽展开）
  -> extrude_reset
  -> RESET_ANCHOR
  -> TRAVEL
  -> PRIME
  -> PRIME_SETTLE
  -> 下一条 PRINT
```

由外部预处理器插入的 primeline 也属于打印路径，使用相同的路径后 reset
规则。最后一条路径同样执行 reset；其后保留一个 E=0 的内部同步点，使
`extrude_reset` 事件先完成，再让现有“最终轨迹完成”逻辑收尾，避免事件与自动
ABORT 竞争。

## ResetE 与绝对 E 基准

每条路径的收尾命令之后插入：

```text
ResetECommand(type="RESET_E", val=0.0, raw="external_npz_path_reset")
```

同时将转换器的 `current_e` 设为 `0.0`。因此 reset 后：

- 下一段 travel 的所有 NPZ 行保持 `E=0`；
- UART 挤出转发器在 reset 完成后回到初始零抑制状态，travel 不产生 E 串口行；
- 下一次预挤出从 `E=0` 增加到 `prime_length_mm`；
- 后续打印 E 从预挤出终值继续累计，直到本路径结束并再次 reset。

禁止只向固件发送 reset 而保留旧的上位机累计 E，否则 reset 后第一条旧绝对 E
会被解释为从零开始的大幅挤出目标。

## reset 同步点

现有 `extrude_reset` 是阻塞事件，但最终路径后若没有新的轨迹行，现有自动完成
检测可能在 reset 状态返回前启动 ABORT。外部 NPZ 转换器因此在每个新增路径
reset 后插入一个内部 `ExtrudeWait`：

```text
wait_sec = dt
delta_e = 0
raw = "external_npz_reset_anchor"
```

该行不增加新的用户参数，不发送变化 E，只为事件完成后提供明确的下一轨迹点。
普通路径随后进入 travel；最终路径则以该同步点作为最后轨迹点。

现有 exporter 的普通零位移轨迹会进入七阶时间规划，实测会被扩展为约 4 秒，
不能作为单周期同步点；而普通零增量 `ExtrudeWait` 默认会继承 reset 前的绝对 E。
因此 exporter 只对 `raw="external_npz_reset_anchor"` 这一内部标记把等待起始 E
设为 `0.0`。其他 `ExtrudeWait`、普通 GCode reset 和共享导出行为保持原样。

## prime_settle_s

在 `ProcessParams` 增加一个全局字段：

```text
prime_settle_s: float = 0.5
```

选择全局字段是因为本次需求只有一个统一等待值，不增加树脂/纤维两套新参数。
该值通过外部 NPZ 的 JSON 配置、正式导出 UI 和 CLI 读取；`0` 表示关闭稳定
等待，负值应被拒绝。

每个 prime `ExtrudeWait` 后插入另一个零增量等待：

```text
wait_sec = prime_settle_s
delta_e = 0
raw = "external_npz_prime_settle"
```

NPZ exporter 继续使用现有 `ExtrudeWait` 展开逻辑，生成
`ceil(prime_settle_s / dt)` 个固定 XYZ、固定 E 的 4 ms 行。UART 去重会抑制这些
相同 E，但 MCU 可利用这段时间追赶最终预挤出目标。

## 修改范围

允许修改：

- `external_npz_preprocessor/process_params.py`：参数默认值；
- `external_npz_preprocessor/converter.py`：路径 reset、E 重基准、reset 同步点和
  prime settle 命令顺序；
- `external_npz_preprocessor/param_config.py`、CLI 和正式导出 UI：参数持久化与入口；
- `data/external_npz_preprocessor/print_params.json`：保存默认值；
- `path_processing_core/npz_exporter.py`：仅识别
  `external_npz_reset_anchor`，将该单周期内部同步行导出为 `E=0`；
- 外部 NPZ 转换/导出相关测试与文档。

禁止修改 `uart_bridge`、`rsi_server`、`control_center`、消息接口和固件协议实现。

## 测试策略

先写失败测试，再实现：

1. `ProcessParams().prime_settle_s == 0.5`，JSON 缺字段时使用默认值，保存后可回读；
2. 两条连续树脂路径的命令顺序为
   `PRINT -> RETRACT -> RESET -> RESET_ANCHOR -> TRAVEL -> PRIME -> SETTLE -> PRINT`；
3. reset 后 `current_e` 重置，travel E 为零，下一 prime 从零累计；
4. 纤维路径的 reset 位于 CUT 之后，导出后位于安全回抽之后；
5. 每条路径（含 primeline 和最终路径）恰有一个路径 reset，不改变工具切换 reset；
6. `prime_settle_s=0.5`、`dt=0.004` 时生成 125 个固定 XYZ/E 行；
7. `prime_settle_s=0` 时不生成 settle 行，负值被拒绝；
8. 最终路径 reset 后存在 E=0 同步点，事件位于最终轨迹点之前；
9. 普通 `ExtrudeWait` 和普通 GCode reset 不使用该内部标记，导出值保持不变；
10. 现有 external NPZ、path processing、GCode reset、UART、RSI 和控制中心相关回归
   测试保持通过。

## Git 管理

1. 本设计规格单独提交；
2. 实现使用 TDD，测试和对应实现以通过状态组成独立代码提交；
3. 文档与最终验证记录单独提交或与对应逻辑提交保持清晰边界；
4. 所有工作只发生在 `feature/uart-canonical-v1`，不合并、不删除分支、不改写历史；
5. 推送远端前再次征得用户授权。

## 验收标准

- 正式外部 NPZ 的每条打印路径收尾后均执行现有 `extrude_reset`；
- reset 位于树脂回抽之后、纤维 CUT 安全回抽之后、下一 travel 之前；
- reset 后上位机与固件 E 基准都为零，travel 不产生变化 E；
- 下一路径到达起点后才原地预挤出；
- 每次预挤出后固定保持默认 `0.5 s` 再开始打印；
- 最终路径 reset 不与自动 ABORT 竞争；
- UART、RSI、控制中心、GCode 和其他无关链路没有行为改动；
- 新增测试和相关回归测试通过。
