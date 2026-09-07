# Core 共形蜂窝挤出量与零 E 连线适配流程

> 文档性质：Core 修改边界、实施顺序与验收指导书  
> 当前状态：只形成修改规划，不代表代码已经完成适配  
> 适用仓库：`kuka_slicer` 内的 `packages/offline_path_planner` Core 链路  
> 关联流程：`蜂窝网格曲面共形映射流程.md`、`蜂窝网格曲面共形映射UI适配流程.md`  
> 首要约束：默认链路、旧 G-code、普通平面切片、纤维、Travel、事件和最终 NPZ 字段语义不得改变

---

## 0. 核心结论

本次适配不修改 2.0 mm 喷嘴、0.5 mm 层高、1.75 mm 耗材和现有挤出倍率组成的平面体积模型。当前共形源路径已经按实际三维弧长计算 E，错误发生在 Core 将一组 `MoveCommand` 拟合成 `GlobalCurveCommand` 后：Core 只保留整条曲线的总 `delta_e`，再按整条拟合曲线弧长均匀分配，导致蜂窝宏路径中原本 `delta_e=0` 的图连接段重新获得挤出量。

必须修复为：

1. External Source NPZ 中的逐点累计 E 仍是权威输入；
2. 共形蜂窝宏路径进入 B 样条拟合后，保留一条与样条参数对应的、单调不减且允许平台区的源 E 曲线；
3. 4 ms 采样时从该 E 曲线求当前绝对 E，而不是用总 E 按整条曲线弧长重新均分；
4. 零 E 图连接段继续属于 PRINT 上下文，但 E 在整个连接区间保持常数；
5. 未显式启用该能力的所有既有链路继续执行当前算法，保证旧输出不变。

零 E 图连接段的语义必须明确：

> 它是“PRINT 上下文中的连续无沉积运动”，不是普通沉积段，也不是 Travel。保留 PRINT 是为了不触发 Travel 专属的抬升、避障、工具补偿、回抽/预挤和路径终止语义；是否出料只由 E 是否变化决定。

---

## 1. 已确认的事实与问题定位

### 1.1 当前平面体积模型

现有树脂参数采用：

\[
A_f=\pi\left(\frac{1.75}{2}\right)^2
\]

\[
A_b=w\,h\,k_e
\]

\[
q_E=\frac{A_b}{A_f}
\]

其中：

- 耗材直径为 `1.75 mm`；
- 单道名义宽度 `w=2.0 mm`；
- 实际层高 `h` 来自主界面 Core 树脂参数；
- `k_e` 为 Core 树脂挤出倍率；
- `q_E` 为每毫米实际打印中心线对应的 E。

对当前已检查案例，`h=0.5 mm`、`k_e=0.95`，所以：

\[
A_b=2.0\times0.5\times0.95=0.95\ \text{mm}^2
\]

\[
q_E\approx0.394964\ \text{E/mm}
\]

该模型与既有平面打印经验一致，本次不得修改 `ResinProcessParams.e_per_mm()`、`RESIN_FIXED_BEAD_WIDTH_MM` 或相关 UI 参数含义。

### 1.2 曲面路径的正确补偿方式

曲面本身不应再叠加一个独立的经验“曲率倍率”。源端应使用实际三维路径弧长：

\[
\Delta E_i=q_E\,\Delta s_{3D,i}
\]

其中：

\[
\Delta s_{3D,i}=\sqrt{\Delta x_i^2+\Delta y_i^2+\Delta z_i^2}
\]

这已经把双正弦表面的坡度引起的路径增长计入挤出量。若再次乘 `1/cos(theta)` 或另一个曲率系数，会重复补偿。

本次实测源 NPZ 中：

- 平面层 E：约 `1415.995`；
- 曲面最强层 E：约 `1428.330`；
- 曲面最强层相对平面层增加约 `0.871%`；
- 全部 20 层因三维弧长增加而额外增加约 `67.464 E`。

因此，共形路径桥接层的三维弧长公式是正确的，不是本次修改目标。

### 1.3 Core 中的实际失真点

当前链路为：

```text
External Source NPZ 的逐点 E
    -> source_npz.py 读取 MaterialPath.extrusion
    -> converter.py 转成逐段 MoveCommand.delta_e
    -> npz_exporter.py 聚合连续 PRINT MoveCommand
    -> GlobalSplinePlanner.fit_global_curve()
    -> GlobalCurveCommand 只保留 total delta_e
    -> polynomial_interpolator.py 按整条拟合曲线弧长均分 total delta_e
```

失真发生在最后两步之间。`MoveCommand` 阶段仍能得到例如：

```text
[正 E, 正 E, 0, 0, 正 E, ...]
```

进入全局样条后却退化为：

```text
整条样条统一 E/弧长
```

当前 150×100×10 mm 实际输出的第 0 层显示：

| 路径 | 应沉积三维长度 | 含零 E 连接的总长度 | E | Core 拟合后平均 E/长度 |
| --- | ---: | ---: | ---: | ---: |
| 矩形外轮廓 | 500.000 mm | 500.000 mm | 197.482 | 0.395008 E/mm |
| 蜂窝宏路径 | 3085.124 mm | 5251.790 mm | 1218.513 | 0.234637 E/mm |

外轮廓没有零 E 连接，结果正确。蜂窝宏路径中零 E 连接占用了较大长度，Core 把相同总 E 分摊到整条宏路径后，平均 E 密度只剩目标值的约 `59.4%`，同时零 E 连接区间被错误赋予挤出。

---

## 2. 不可改变的既定语义

以下内容属于兼容契约，实施时不得改变。

### 2.1 External Source NPZ

- 格式仍为 `external_layer_paths_v1`；
- 路径点仍支持 `XYZ` 或 `XYZABC`；
- `layer_xxxx_R_E` 仍为与每个路径点对齐的累计 E；
- E 可以出现平台区，但不得倒退；
- 路径、点、层和材料顺序不变；
- 零 E 连接不拆成新的外部路径，不重排蜂窝一笔画顺序。

### 2.2 MoveCommand

- 零 E 连接继续生成 `type="PRINT"`、`cmd="G1"`；
- 其 `delta_e` 必须为 0；
- 不转换为 `TRAVEL`；
- 不在连接前后自动插入 prime、retract、reset、safe lift 或 tool change；
- 非零沉积段继续使用同一树脂工具、进给速度和逻辑层。

### 2.3 最终 Core NPZ

- 不新增或删除现有数组；
- `move_type`/`PRINT_FIT` 编码不变；
- `x,y,z,a,b,c,e,tool_id,event_flag,layer_index,path_id` 等字段含义不变；
- `e` 仍表示绝对 E 状态；零 E 区间通过连续采样行中的相同 E 值表达；
- 不增加新的 `ZERO_E_PRINT`、`COAST` 等最终枚举，避免影响上位机和现有读取器；
- 本次修复不得改变 XYZABC、4 ms 周期、事件触发序号或工具补偿。

### 2.4 默认兼容行为

没有显式请求“保留源逐段 E”的输入，继续使用当前“总 E 按最终曲线弧长分配”逻辑。这样旧 G-code、普通平面 STL、旧 External NPZ、纤维路径及既有黄金文件保持原样。

---

## 3. 新增的最小内部契约

### 3.1 路由级显式开关

不得仅根据“路径中碰巧有零 E”自动切换 Core 全局行为。共形路径桥接应在源 NPZ 元数据中增加可选声明，例如：

```json
{
  "core_processing": {
    "source_e_profile_mode": "piecewise_preserve_v1",
    "zero_e_connector_semantics": "print_context_constant_e"
  }
}
```

规则：

- 字段缺失：完全沿用旧逻辑；
- 精确等于 `piecewise_preserve_v1`：启用本次新逻辑；
- 未知值：硬失败并给出可诊断错误，不静默回落；
- 只有路径存在与点数严格一致的显式 E 数组时才允许启用；
- 元数据只控制 Core 内部 E 采样，不改变输出 NPZ schema。

首版只由新共形蜂窝路径桥接写入该字段。不得让普通 Prusa/G-code 路线自动进入新模式。

### 3.2 GlobalCurveCommand 内部字段

不要复用当前只为 `POLYLINE` 控制点定义的 `e_profile`，因为 B 样条控制点数量通常不等于源路径点数量。建议为 `GlobalCurveCommand` 增加两个可选内部字段：

```python
source_e_parameters: Optional[List[float]] = None
source_e_values: Optional[List[float]] = None
```

定义：

- `source_e_parameters`：与位置 B 样条相同的归一化参数轴，首尾必须为 0 和 1；
- `source_e_values`：该参数处的绝对 E；
- 两者长度相同且至少为 2；
- 参数严格递增；
- E 单调不减，允许连续多个值相等；
- 首值必须等于 `e_val-delta_e`；
- 末值必须等于 `e_val`；
- 字段为 `None` 时执行旧的总量按弧长分配；
- 这些字段是 Core 内部信息，不写入最终 NPZ。

### 3.3 为什么使用绝对 E 曲线

使用绝对 E 而不是只有沉积/非沉积布尔标记，可以同时覆盖：

- 平面恒定 E/mm；
- 曲面按三维弧长变化后的 E；
- 零 E 平台；
- 未来由上游明确给出的非均匀流量；
- 现有每条路径的 reset 后局部 E 起点。

Core 不重新猜测喷嘴宽度、层高或耗材面积。2.0 mm、0.5 mm、1.75 mm 和挤出倍率已经在源 E 中完成换算。

---

## 4. B 样条 E 参数化算法

### 4.1 输入序列

对一组准备拟合的连续 PRINT moves，构造：

```text
P0, P1, ..., Pn
E0, E1, ..., En
```

其中：

- `P0 = moves[0].start_pos`；
- `Pi = moves[i-1].pos`；
- `E0 = moves[0].e_val - moves[0].delta_e`；
- `Ei = moves[i-1].e_val`。

不得通过 XYZ 最近邻反推 E 对应点。蜂窝路径可能自接近或重复经过同一区域，最近邻会把 E 映射到错误分支。

### 4.2 拟合点生成时同步传播 E

当前 `_generate_fitting_points()` 会在角点两侧增加回退点，`_subdivide_points()` 会增加中点。适配后，位置和 E 必须作为同一个内部样本传播：

```text
FitSample(position, absolute_e)
```

传播规则：

- 原始顶点使用原始绝对 E；
- 角点前后回退点按所在源线段的相同比例线性插值 E；
- 加密中点同时对 XYZ、ABC 和 E 插值；
- 源 `delta_e=0` 的整个线段上，所有生成点的 E 必须完全相同；
- 相邻零 XYZ 且 E 不同的输入不得被静默去重，应硬失败或转交既有纯挤出事件链路；
- 数值容差只能用于校验，不得把小的合法正 E 强制归零。

位置拟合仍使用当前算法。E 不参与位置 B 样条最小二乘，避免改变几何、速度和姿态。

### 4.3 建立共同参数轴

在完成角点回退和可选加密后，继续使用位置拟合当前得到的 `param`。把每个 `FitSample.absolute_e` 与同索引的归一化 `param` 写入：

```text
source_e_parameters
source_e_values
```

这与现有 `orientation_parameters/orientation_quaternions` 的思路一致：位置由 B 样条拟合，姿态和 E 都作为共享参数轴上的局部样本保留。

不得对 E 单独做高阶 B 样条拟合。高阶拟合可能产生过冲、负 `delta_e` 或破坏零 E 平台。E 只允许在相邻样本之间分段线性插值。

### 4.4 4 ms 采样

在 `sample_global_curve_iter()` 的普通 B 样条分支中：

1. 时间规划和 `curr_s` 计算保持原样；
2. 通过弧长查表得到当前 B 样条参数 `u`；
3. 计算 `normalized_u`；
4. 若存在合法的 `source_e_parameters/source_e_values`，则分段线性求 `target_e(normalized_u)`；
5. `delta_e = target_e - previous_e`；
6. `extrude_speed = delta_e / dt`；
7. 若字段为空，继续执行现有 `curve.delta_e * delta_s / total_length`。

采样首点和末点必须强制使用源 E 曲线首尾值，避免累计浮点误差。

零 E 平台的验收不是只看整条路径总 E，而是要求：只要连续采样点都落在该平台参数区间，输出 E 必须逐位相同，允许误差不大于 `1e-9`。

### 4.5 输出总量语义

首版以 External Source NPZ 的累计 E 为权威：

\[
E_{core,end}-E_{core,start}=E_{source,end}-E_{source,start}
\]

Core 不根据拟合后曲线长度再次修改路径总 E。这样同时保证：

- 平面路径保持当前已验证用量；
- 曲面路径保留源端已经计算好的三维弧长增量；
- 零 E 连接保持零；
- External Source NPZ 的既定语义不被改写。

拟合误差导致的最终曲线长度微小变化，应由 `spline_max_error_mm` 控制并作为几何质量指标报告，不在本次通过二次 E 缩放补偿。否则会违背“源 E 是权威输入”的现有契约。

---

## 5. 修改边界

### 5.1 允许修改的模块

| 模块 | 允许的最小修改 |
| --- | --- |
| `kuka_slicer/conformal_lattice/path_bridge.py` | 只增加共形源 NPZ 的 opt-in 元数据；不改路径、E 公式或一笔画规划 |
| `external_npz_preprocessor/export_runner.py` | 读取并校验 opt-in 模式，将布尔/枚举选项传给 Core exporter |
| `path_processing_core/types.py` | 给 `GlobalCurveCommand` 增加两个默认 `None` 的内部 E 参数字段 |
| `path_processing_core/bspline_approximation.py` | opt-in 时随拟合点传播 E 并生成参数化 E 样本；默认路径保持原实现 |
| `path_processing_core/npz_exporter.py` | 将 opt-in 选项传给拟合器；fallback/polyline 分支保持兼容 |
| `path_processing_core/polynomial_interpolator.py` | B 样条采样时可选使用参数化源 E；旧分支原样保留 |
| 对应测试目录 | 增加单元、集成、黄金输出和共形实际案例测试 |

### 5.2 禁止修改的模块和行为

本次不得修改：

- 2.0 mm 固定树脂线宽常量；
- 1.75 mm 耗材直径和现有平面 E/mm 公式；
- 共形路径的三维弧长计算；
- 蜂窝拓扑、最大不重复 trail cover、分区策略和外矩形轮廓；
- 零 E 连接的路径坐标、顺序和 PRINT 上下文；
- KUKA ABC 曲面法向姿态；
- Core 的 XYZ B 样条、速度曲线、角点回退、4 ms 周期和姿态 SLERP；
- Travel、纤维、工具切换、切断、prime、retract、reset 和事件处理；
- 最终 NPZ schema、枚举值和上位机读取代码；
- 旧曲面映射器和普通 Prusa 切片链路；
- `kuka_ram_ws` 上位机仓库。

禁止用以下方式规避问题：

- 把零 E PRINT 改为 Travel；
- 删除零 E 图连接，导致一笔画路径断开；
- 对整条宏路径统一重新计算 E/mm；
- 按 XY 长度重新计算曲面 E；
- 为共形模式复制一套独立 Core；
- 修改最终 `move_type` 枚举让上位机承担修复；
- 仅在预览层把连接画成零 E，而最终 Core NPZ 仍在出料。

---

## 6. 分阶段实施顺序

### Gate C1：锁定回归基线

只增加诊断和测试夹具，不改变生产逻辑：

1. 固定一个普通平面 NPZ、一个普通 Prusa G-code、一个纤维案例；
2. 记录当前最终 NPZ 的文件哈希或逐数组哈希；
3. 固定一个含 `[正 E, 0 E, 正 E]` 的最小宏路径；
4. 固定当前 150×100×10 mm 共形案例的源 E、层数和路径统计；
5. 确认当前失败证据可重复：零 E 段进入样条后 E 发生变化。

验收：只新增测试，测试应准确暴露当前缺陷；旧测试全部保持通过。

### Gate C2：增加内部数据契约

1. 为 `GlobalCurveCommand` 增加默认 `None` 的源 E 参数字段；
2. 增加字段合法性辅助函数；
3. 不连接生产调用；
4. 增加字段缺失时旧采样完全不变的单元测试。

验收：旧黄金输出逐数组一致；新字段不会被序列化进最终 NPZ。

### Gate C3：拟合阶段传播 E

1. 在 opt-in 分支中生成带 E 的拟合样本；
2. 角点回退和加密同步插值 E；
3. 构建 `source_e_parameters/source_e_values`；
4. 验证 E 单调性、首尾值和零 E 平台；
5. 默认分支仍走现有位置/姿态拟合路径。

验收：拟合前后的参数化 E 首尾严格一致，零 E 段至少形成两个相同 E 样本。

### Gate C4：4 ms 样条采样适配

1. 新增分段线性 E 查询；
2. 在普通 B 样条采样分支中按 opt-in 字段计算当前 E；
3. 保留旧的总量按弧长分配作为字段为空时的 fallback；
4. 检查 `delta_e>=-1e-9`，超过容差立即失败；
5. 首末点强制闭合到源 E。

验收：最小 `[3,0,4]` E 案例经过拟合与 4 ms 采样后，总增量为 7，零 E 参数区间内 E 不变。

### Gate C5：仅接入共形蜂窝链路

1. 共形路径桥接写入 `piecewise_preserve_v1` 元数据；
2. `convert_source_job()` 校验元数据和显式 E 数组；
3. 只把该模式传给 Core exporter；
4. 未标记的 External NPZ 和 G-code 不启用；
5. 未知模式硬失败，不允许静默使用旧算法生成错误结果。

验收：同一 Core 可同时处理旧链路和新共形链路，路由选择在日志/统计中可见。

### Gate C6：端到端验收与性能检查

1. 用实际 150×100×10 mm 共形案例重新输出 Core NPZ；
2. 比较源路径和最终 Core 每条路径的累计 E；
3. 检查平面层与峰值曲面层的 E 比例；
4. 检查零 E 连接区间；
5. 检查 XYZABC、层、路径、事件和工具字段；
6. 记录拟合时间、采样时间、输出行数和文件大小。

验收后再提交生产代码；每个 Gate 单独使用中文提交说明。

---

## 7. 必须增加的测试

### 7.1 Core 单元测试

至少包含：

1. `source_e_parameters=None` 时与旧公式结果完全一致；
2. 直线 `[0,2,10]` E 曲线按参数正确采样；
3. `[0,3,3,7]` 中间平台在 B 样条采样后保持常数；
4. 曲面 XYZ 路径中 E 总量不因 ABC 变化而改变；
5. E 参数非递增、长度不匹配、首尾不匹配和 E 倒退均硬失败；
6. 角点回退点和加密点的 E 插值正确；
7. fallback linear、POLYLINE 和普通 SPLINE 各自行为明确且不串用字段。

### 7.2 External NPZ/Core 集成测试

构造一条路径：

```text
P0 -> P1：沉积 3 E
P1 -> P2：零 E 连线
P2 -> P3：沉积 4 E
```

必须验证：

- 转换后仍是连续 PRINT 上下文；
- 不产生额外 Travel、retract、prime 或 reset；
- 最终 E 总增量为 7；
- P1—P2 对应的采样区间 E 保持常数；
- 启用和不启用 opt-in 时能明确复现新旧差异；
- 最终 NPZ 数组集合和 dtype 不变。

### 7.3 平面兼容测试

对既有普通平面路径必须验证：

- 未标记输入的输出逐数组一致；
- 平面 `E/mm` 仍由 `2.0 × layer_height × extrusion_scale / filament_area` 得出；
- 已有 Prusa 与等价 External NPZ 的一致性测试保持通过；
- 旧 G-code、Brim、Raft、轮廓和普通 infill 不发生哈希漂移；
- prime/retract/reset 的绝对 E 边界不变。

### 7.4 共形曲面测试

至少比较三个逻辑层：平面层、过渡层、峰值层。

必须满足：

\[
\sum\Delta E_{core}=E_{source,end}-E_{source,start}
\]

并验证：

- 曲面层 E 大于对应 XY 投影长度计算值；
- 平面层保持既有 E；
- 峰值层/平面层的 E 比与源三维弧长比一致；
- 外矩形轮廓和蜂窝结构分别统计；
- 零 E 连接的累计长度大于零，但累计 E 增量为零；
- ABC 非零时 E 结果只受 XYZ 三维长度和源 E 影响，不直接受角度值影响。

### 7.5 实际案例验收阈值

对当前固定案例建议使用：

- 每条源路径首尾 E 残差：`<=1e-6`；
- 全任务 E 总量残差：`<=1e-5`；
- 零 E 平台相邻采样 E 差：`<=1e-9`；
- 不允许出现小于 `-1e-9` 的打印 `delta_e`；
- 未启用新模式的黄金输出：逐数组完全一致；
- XYZABC、行数、路径 ID 和事件序号：除非测试证明现有采样必然变化，否则应完全一致；
- 运行时间和峰值内存增量应单独报告，不能用关闭 B 样条作为性能规避方案。

---

## 8. 诊断与失败策略

新模式执行时，Core 统计结果应至少报告但不新增最终 NPZ 字段：

```text
source_e_profile_mode
profiled_curve_count
zero_e_connector_segment_count
source_e_total
sampled_e_total
maximum_path_e_residual
negative_delta_e_count
```

以下情况必须停止转换：

- opt-in 路径缺失 E 数组；
- E 点数与路径点数不一致；
- E 包含 NaN/Inf；
- E 明显倒退；
- B 样条 E 参数与位置参数长度不一致；
- E 首尾与 `GlobalCurveCommand` 的绝对 E 状态不一致；
- 采样后路径 E 总量超出验收容差。

不得在这些情况下静默回落为总 E 均分，因为静默回落会重新引入零 E 连接出料问题。

---

## 9. 兼容性矩阵

| 输入链路 | 显式 E | 新元数据 | Core E 行为 | 预期输出变化 |
| --- | --- | --- | --- | --- |
| 普通 Prusa G-code | 有或无 | 无 | 旧逻辑 | 无 |
| 普通平面 External NPZ | 无 | 无 | 旧逻辑按 E/mm | 无 |
| 旧版曲面映射 NPZ | 有 | 无 | 旧逻辑 | 无 |
| 纤维路径 | 有或无 | 无 | 旧纤维逻辑 | 无 |
| 新共形蜂窝 NPZ | 有 | `piecewise_preserve_v1` | 参数化源 E | 仅修正 E 分布 |
| 未知第三方 NPZ | 有 | 未知值 | 拒绝 | 明确错误 |

---

## 10. 提交与变更控制

后续实施前必须分别检查：

1. `kuka_slicer` 工作树中的未提交改动；
2. `packages/offline_path_planner` 下相关文件是否已有用户修改；
3. Graphify 索引是否包含最新 Core 源码；
4. 基线测试和黄金输出是否可重复。

每个 Gate 只提交本 Gate 的文件，中文提交建议为：

```text
增加Core源E参数化内部契约
保留共形样条逐段挤出语义
接入共形蜂窝零E连线保护
补充Core共形挤出端到端验证
```

禁止顺带格式化或重写无关 Core 文件。若目标文件存在用户未提交修改，必须先识别重叠范围，不能覆盖或把无关修改带入提交。

---

## 11. 最终验收条件

只有同时满足以下条件，才能认为本次适配完成：

- 2.0 mm 喷嘴、1.75 mm 耗材和现有层高/倍率模型未改变；
- 平面共形层使用当前已验证的 E/mm；
- 曲面层保留由三维弧长产生的额外 E；
- 蜂窝结构沉积段不再因零 E 连接长度而被稀释；
- 零 E 连接在最终 Core NPZ 中存在连续运动且 E 保持常数；
- 零 E 连接仍是 PRINT 上下文，没有触发 Travel 行为；
- 外矩形轮廓 E 不发生非预期变化；
- 未启用新元数据的所有旧链路输出不变；
- 最终 NPZ schema、枚举、事件和上位机接口不变；
- 实际共形案例通过逐路径 E 守恒检查和零 E 平台检查；
- 所有相关回归测试通过，并保存可重复的数值审计结果。

---

## 12. 本文档的实现边界

本文档只规划 Core 对“权威源 E 曲线”的保留方式，不修改任何生产代码。它不讨论材料流变、背压、启停延迟、喷嘴离面高度和实验流量标定；这些因素后续可以通过 `extrusion_scale` 或 `e_per_mm_override` 标定，但不能与本次零 E 语义修复混在同一个改动中。

本次适配的唯一目标是：

> 在不改变旧链路和最终输出契约的前提下，让新共形蜂窝路径在 Core B 样条拟合及 4 ms 采样后仍严格保留源端已经计算正确的平面/曲面 E 分布，并保证图连接段持续运动但绝不出料。
