# bspline 模块说明

本目录提供 `path_processing_core` 当前使用的 B 样条基础算法实现。`gcode_planner.bspline.*` 仅保留旧 import 路径兼容入口。

## 当前版本

- 版本标记：`v6`
- 本版优化点：在 `v5` 基础上，改为带状矩阵构造与带状 Cholesky 求解

## 参与文件

- `BaseFunction.py`：递归 Cox-de Boor 基函数
- `parameter_selection.py`：参数化方法与节点向量辅助逻辑
- `bspline_curve.py`：曲线插值与最小二乘逼近
- `bspline_surface.py`：曲面相关逻辑，内部也会复用 `curve_approximation()`

## 当前完整拟合逻辑

`path_processing_core` 上层在 [bspline_approximation.py](/home/jayson/kuka_ram_ws/src/my_project/path_processing_core/path_processing_core/bspline_approximation.py) 中完成拟合前处理，本目录负责底层曲线逼近。

### 上层输入准备

1. 上层将连续 `MoveCommand` 组织成一个待拟合段。
2. 对原始折线顶点做去重。
3. 对角点执行回退点插入，避免尖角直接参与平滑拟合。
4. 根据 `density` 做中点递归加密。
5. 估计控制点数量 `H`，并将 6 维位姿整理成：

```text
D = [D_X, D_Y, D_Z, D_A, D_B, D_C]
```

其中每个 `D_*` 长度均为数据点数 `N`。

### 底层最小二乘逼近

当前 `bspline_curve.curve_approximation()` 的数学流程如下：

1. 输入数据点数组 `D`、数据点数 `N`、控制点数 `H`、阶次 `k`、参数 `param`、节点向量 `knot`。
2. 对每个数据参数 `u_i`，只计算该行 `k+1` 个非零基函数：

```text
span = find_span(u_i)
basis_vals = basis_funs(span, u_i, k, knot)
```
3. 固定首末控制点：

```text
P0 = D[:, 0]
PH-1 = D[:, N-1]
```

4. 对每个内部数据点行，按局部支撑直接构造该行对应的未知控制点系数 `w_i`，并同步扣除首末控制点影响得到 `q_i`。

```text
w_i 只包含内部控制点列上的非零基函数
q_i = D[:, i] - N_i,0 * P0 - N_i,H-1 * PH-1
```

5. 直接按行累加法方程，但不再构造稠密 `M`，而是按下带状形式存储：

```text
M_band[offset, col] += w_i^T * w_i 的对应带状项
Q_all += w_i^T * q_i
```

6. 对带状对称正定矩阵做带状 Cholesky 分解并一次性解出全部维度的内部控制点：

```text
P_inner_all = cho_solve_banded(cholesky_banded(M_band), Q_all)
```

7. 将首末控制点与内部控制点拼接成完整控制点矩阵。
8. 返回控制点数组。

## v1 优化说明

本版只做“不改变数学逻辑”的去重计算优化：

- `Nik` 只构造一次，不再对 `X/Y/Z/A/B/C` 六个维度重复构造
- `N_part` 只切片一次
- `M = N_part^T * N_part` 只计算一次
- `M_inv` 只计算一次

本版没有改变以下内容：

- 仍然使用递归 `BaseFunction`
- 仍然使用全矩阵方式构造 `Nik`
- 仍然使用法方程 `inv(M) * Q`
- 仍然按维度分别求右端项和控制点

## v2 优化说明

本版在 `v1` 基础上继续保持相同数学模型，但将内部控制点求解方式从显式矩阵求逆改为线性方程求解：

- 从 `P_inner = inv(M) * Q`
- 变为 `P_inner = solve(M, Q)`

这样做的影响：

- 功能语义不变，仍然是同一个法方程
- 通常比显式求逆更稳定
- 通常比显式求逆更快

本版仍然没有改变以下内容：

- 仍然使用递归 `BaseFunction`
- 仍然使用全矩阵方式构造 `Nik`
- 仍然按维度分别构造 `Qk/Q`
- 仍然没有利用局部支撑特性

## v3 优化说明

本版在 `v2` 基础上将 6 个维度合并为一次多右端求解：

- 从“每个维度单独构造 `Qk`、单独构造 `Q`、单独 `solve(M, Q)`”
- 变为“统一构造 `Qk_all/Q_all`，一次 `solve(M, Q_all)` 解出全部维度”

这样做的影响：

- 功能语义不变，左端矩阵 `M` 完全相同
- 避免对同一个 `M` 重复做 6 次求解
- 减少 Python 循环和小矩阵重复调用

本版仍然没有改变以下内容：

- 仍然使用递归 `BaseFunction`
- 仍然使用全矩阵方式构造 `Nik`
- 仍然没有利用局部支撑特性

## v4 优化说明

本版在 `v3` 基础上，把基函数矩阵 `Nik` 的构造从“全矩阵递归逐项求值”改成“逐行只计算 `k+1` 个非零基函数”：

- 先根据参数 `u_i` 找到所在的 `span`
- 再用迭代版 Cox-de Boor 计算该行的 `k+1` 个非零基函数
- 只把这些值写入 `Nik[i, span-k:span+1]`

这样做的影响：

- 功能语义不变，仍然是同一组 B 样条基函数
- 显著减少基函数求值次数
- 去掉大量无意义的零值列计算
- 避免在 `curve_approximation()` 中继续使用递归 `BaseFunction`

本版仍然没有改变以下内容：

- `curve_interpolation()` 和 `curve()` 仍然沿用旧写法
- 仍然构造稠密矩阵 `Nik`，只是填充方式更高效
- 仍然没有进一步做稀疏矩阵求解

## v5 优化说明

本版在 `v4` 基础上，不再显式构造并参与后续乘法的稠密 `Nik/N_part/Qk_all`，而是利用局部支撑逐行直接累加法方程：

- 对每个内部数据点，只保留该行内部控制点对应的非零基函数系数
- 直接累加 `M += w_i^T w_i`
- 直接累加 `Q_all += w_i^T q_i`

这样做的影响：

- 功能语义不变，仍然在解同一个最小二乘法方程
- 避免稠密 `N_part.T @ N_part`
- 避免稠密 `N_part.T @ Qk_all`
- 进一步把局部支撑特性用到法方程构造阶段

本版仍然没有改变以下内容：

- `curve_interpolation()` 和 `curve()` 仍然沿用旧写法
- `M` 仍以稠密矩阵形式求解
- 还没有引入带状矩阵或稀疏求解器

## v6 优化说明

本版在 `v5` 基础上，利用 `M` 的带状结构，把求解阶段从稠密线性代数切换为带状 Cholesky：

- `M` 不再以完整稠密矩阵存储
- 只保存下带状区域
- 使用带状 Cholesky 分解和带状回代求解全部右端项

这样做的影响：

- 功能语义不变，仍然在解同一个最小二乘法方程
- 计算目标不变，只是改变存储与求解方式
- 对高 `density` 下的大规模控制点问题更友好
- 更符合当前三次 B 样条局部支撑导致的带状结构

本版仍然没有改变以下内容：

- `curve_interpolation()` 和 `curve()` 仍然沿用旧写法
- 上层角点回退、密度加密、参数化、控制点数规则均不变
- 若运行环境没有 `scipy`，仍保留回退到 `numpy` 稠密求解的路径

## 当前性能特点

当前版本相较初始实现，已经去掉“同一条曲线在 6 个维度上重复构造同一基函数矩阵”的明显浪费。

剩余主要性能瓶颈仍然是：

1. `curve_interpolation()` 和 `curve()` 仍使用旧的递归基函数求值
2. 当前只优化了 `curve_approximation()` 主路径
3. 还没有进一步评估更激进的稀疏矩阵或专用样条库替换

## 后续版本规划

- `v7`：评估是否将 `curve_interpolation()` 和 `curve()` 也切换到同一套非零基函数实现
- `v8`：评估是否需要更进一步的稀疏矩阵或专用样条库替换
