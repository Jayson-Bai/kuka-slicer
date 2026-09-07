# Core NPZ 下游点列排查说明

案例文件：`resin_0728_core (1).npz`

## 结论

当前案例的 Core NPZ 中，首层 Brim 的长直线段没有发现高频横向摆动。

- `move_type=PRINT` 的直线采样间隔约为 `0.06 mm`；
- 一段约 `38.5 mm` 的水平直线，最大横向残差约 `3.4e-9 mm`；
- 一段约 `55.4 mm` 的垂直直线，横向残差约为 `0`；
- 二阶点差最大约 `1.5e-5 mm`，属于 float32 输出量化误差范围。

因此，若真实机器人轨迹出现可见的垂直方向往复摆动，优先检查下游注入、点列重组或机器人控制器插值，不应先判定为 Prusa/Core 已经生成了摆动。

## Core NPZ 字段解释

必须按 `seq` 的原始顺序消费点列，不能按 `path_id` 排序或拼接。

`move_type` 由文件内词典解码：

| 数值 | 含义 | 处理方式 |
|---:|---|---|
| 0 | `TRAVEL` | 空走点列 |
| 1 | `PRINT` | 树脂/打印点列 |
| 2 | `TRAVEL_FIT` | 平滑后的空走点列 |
| 3 | `PRINT_FIT` | 平滑后的打印点列 |
| 4 | `EVENT` | 事件行，不参与几何点列 |

`path_id` 不是可靠的全局唯一打印路径编号；同一个 `path_id` 可能包含多个打印段和事件行。路径边界应使用以下组合判断：

1. `seq` 连续顺序；
2. `move_type` 是否改变；
3. `path_end_flag=1` 是否表示当前运动段结束；
4. 必要时结合 `src_line` 和 `layer_index`。

不要把 `EVENT` 行当作运动点，也不要把 `TRAVEL` 与 `PRINT` 直接拼成一条打印曲线。

## 下游注入的允许操作

如果只是注入工具偏置或树脂 Z 补偿：

- 可以对指定行做恒定 XYZ 平移；
- 不得重新排序、合并、拆分或重新采样点列；
- 不得对 XY 再做 B-spline、Catmull-Rom 或其他二次平滑；
- 不得按 `path_id` 重建路径；
- 不得改变 `seq`、`move_type`、`event_type`、`path_end_flag`、`e` 和时间字段；
- 不得把 `EVENT` 行删除后再用相邻点重新连线。

当前案例的注入清单中，基准工具偏置为 `[0, 0, 0]`，基准树脂 Z 补偿为 `0`。如果实际注入了非零值，注入后应表现为对应区域的恒定平移，而不是横向周期摆动。

## 建议的三阶段定位方法

下游必须保存三个版本的点列：

1. Core NPZ 原始点列；
2. 注入后、发送给机器人前的笛卡尔点列；
3. 机器人控制器实际接收或执行前的笛卡尔 TCP 点列。

对同一首层 Brim 直线段分别计算：

```python
import numpy as np

def line_metrics(points):
    points = np.asarray(points, dtype=np.float64)
    v = points[-1, :2] - points[0, :2]
    length = np.linalg.norm(v)
    if length == 0:
        return None

    q = points[:, :2] - points[0, :2]
    cross = (q[:, 0] * v[1] - q[:, 1] * v[0]) / length
    second_diff = np.diff(points[:, :2], n=2, axis=0)

    return {
        "max_perpendicular_error_mm": float(np.max(np.abs(cross))),
        "max_second_difference_mm": float(
            np.max(np.linalg.norm(second_diff, axis=1))
        ) if len(second_diff) else 0.0,
        "path_length_mm": float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum()),
        "chord_mm": float(length),
    }
```

判定建议：

- `<= 2e-5 mm`：通常是 float32 量化误差；
- `2e-5 ~ 1e-3 mm`：检查坐标转换和舍入；
- `> 1e-3 mm` 且呈周期性正负交替：重点检查下游插值/平滑；
- `> 0.01 mm`：基本可以排除单纯 NPZ 存储精度，应定位到注入或机器人控制器。

注入前后如只应用恒定偏置，应检查：

```python
delta = after_xyz - before_xyz
assert np.max(np.std(delta[:, :2], axis=0)) < 2e-5
```

对于 Z 补偿，同一区域的 `delta[:, 2]` 应为恒定值；XY 不应产生周期性变化。

## 最容易出现问题的下游错误

1. 用 `path_id` 排序，导致运动顺序改变；
2. 未过滤 `EVENT` 行；
3. 把 `TRAVEL`、`PRINT`、`PRINT_FIT` 连接成一条曲线；
4. 将每个 NPZ 点先转成关节角，再用关节空间平滑后重新生成 TCP 点；
5. 对已经经过 Core 七阶采样的点列再次使用三次样条；
6. 将坐标四舍五入到 `0.001 mm` 后再做直线拟合；
7. 把 `planned_time_s` 当作重新规划时间，而不是沿用原始 `seq` 顺序；
8. 在工具偏置注入时对每个点使用不同的偏置，或错误地使用了姿态相关坐标变换。

最终排查时，应首先比较“Core 原始笛卡尔点列”和“注入后发送点列”。如果两者直线残差一致，再比较“发送点列”和“控制器实际执行 TCP 点列”。这样可以直接确定摆动首次出现在哪个阶段。
