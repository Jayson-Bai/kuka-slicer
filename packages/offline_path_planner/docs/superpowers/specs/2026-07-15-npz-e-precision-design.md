# NPZ E 精度统一设计

## 目标

将 NPZ 轨迹中的绝对挤出量 E 统一为固定 6 位小数语义，并保证在线链路最终发给 UART 的 `E` 文本也固定保留 6 位小数。

## 范围

- 保持 E 的绝对值语义、`extrude_scale` 倍率语义和 RSI 心跳时序不变。
- 保持现有 NPZ 文件格式不变；NPZ 的 `e` 数组仍以 `float32` 存储，避免破坏已有文件兼容性。
- `center_node` 继续在发布前按 `e_decimals` 舍入，默认值保持 6。
- `uart_node` 发送 E 时使用固定小数格式，并明确设置 6 位小数，避免 C++ 默认流格式产生有效数字截断。
- 不修改 GCode/外部 NPZ 的挤出计算公式。

## 数据流

```text
NPZ float32 e
  -> NpzLoader float
  -> center_node 按 e_decimals 舍入
  -> rsi_node RsiHeartBeat.extrude_abs
  -> uart_node 乘 extrude_scale
  -> E <seq> <tool> <固定6位小数>
```

## 精度约定

- 约定为“小数点后 6 位”，不是“6 位有效数字”。
- 正负值、回抽值和跨工具切换后的绝对 E 都使用同一格式。
- `extrude_scale` 先作用于绝对 E，再进行最终 UART 文本格式化。
- NPZ 内部 `float32` 量化误差继续存在，但在线边界不再额外引入默认 `ostringstream` 的有效数字截断。

## 测试策略

- 增加静态/源码级测试，验证 UART 发送函数使用固定 6 位小数格式。
- 覆盖普通 E、较大 E、小数 E 和负回抽 E 的格式化结果。
- 保留并运行现有 UART、NPZ、控制中心相关测试，确认事件握手和 E 转发逻辑不变。
