# `kuka_drivers` 下游 Codex 排查任务

## 目标

排查真实打印时，首层 Brim 中本应为直线的轨迹出现打印方向垂直方向的微小往复摆动。

案例文件：

```text
resin_0728_core (1).npz
```

上游参考文件：

```text
docs/downstream_npz_pointlist_diagnosis.md
```

## 安全限制

本任务只允许检查代码和点列数据：

- 不进入或修改 Docker 宿主配置；
- 不操作真实机器人、驱动器、串口、网络设备或运动服务；
- 不启动打印、不发送运动指令、不重启服务；
- 不修改生产代码；
- 先完成只读诊断报告，再等待确认后考虑修复。

## 必须检查的 NPZ 规则

先读取文件内的词典：

```text
move_type:
0 = TRAVEL
1 = PRINT
2 = TRAVEL_FIT
3 = PRINT_FIT
4 = EVENT
```

分析时必须：

1. 按 `seq` 保持全局顺序；
2. 只将 `PRINT` 和 `PRINT_FIT` 作为打印几何；
3. 将 `TRAVEL` 和 `TRAVEL_FIT` 单独分析；
4. `EVENT` 不得参与几何点列；
5. 不能按 `path_id` 排序或直接拼接，`path_id` 不是全局唯一打印路径编号；
6. 使用 `path_end_flag`、`move_type`、`layer_index` 和 `src_line` 识别运动段边界。

## 分阶段对比

如果下游存在注入流程，请保存并比较三份数据：

1. Core 原始 NPZ 点列；
2. 注入后、发送给机器人前的 Cartesian 点列；
3. 机器人控制器实际接收或执行前的 Cartesian TCP 点列。

对于首层 Brim 的同一条长直线，分别计算：

- 起点到终点的弦长；
- 点列实际长度；
- 最大垂直于直线方向的残差；
- 二阶点差；
- 横向残差是否正负交替；
- 注入前后 `x/y/z` 的差值是否为恒定偏置。

可使用：

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
    second = np.diff(points[:, :2], n=2, axis=0)
    return {
        "max_perpendicular_error_mm": float(np.max(np.abs(cross))),
        "max_second_difference_mm": float(
            np.max(np.linalg.norm(second, axis=1))
        ) if len(second) else 0.0,
        "path_length_mm": float(
            np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum()
        ),
        "chord_mm": float(length),
    }
```

## 判定参考

- `<= 2e-5 mm`：通常属于 float32 量化误差；
- `2e-5 ~ 1e-3 mm`：检查坐标转换和舍入；
- 大于 `1e-3 mm` 且横向残差周期性正负交替：重点检查二次插值或平滑；
- 大于 `0.01 mm`：基本可以排除单纯 NPZ 存储误差，应定位到注入或机器人控制器。

若只注入恒定工具偏置，必须满足：

```python
delta = after_xyz - before_xyz
assert np.max(np.std(delta[:, :2], axis=0)) < 2e-5
```

同时保持以下字段不变：

```text
seq
e
planned_time_s
move_type
event_type
path_end_flag
```

## 重点排查项

1. 是否把 `TRAVEL` 或 `EVENT` 行连接进了打印曲线；
2. 是否按 `path_id` 重排点列；
3. 是否对 Core 已采样点再次做 B-spline、Catmull-Rom 或三次样条；
4. 是否在关节空间平滑后重新生成 TCP 点；
5. 是否将坐标四舍五入后再次拟合；
6. 是否重复应用工具偏置或错误使用姿态相关坐标变换；
7. 是否把 `planned_time_s` 重新规划成不同时间后改变了采样位置；
8. 是否在注入后改变了打印段的边界或连接关系。

## 输出要求

先输出只读诊断报告，至少包括：

- 摆动首次出现的阶段；
- 对应层、运动段和字段；
- 注入前后最大横向误差；
- 是否是 Core/Prusa 源路径问题；
- 最小修复建议；
- 是否需要修改下游代码。

未得到确认前，不要修改生产代码或执行任何机器人相关操作。
