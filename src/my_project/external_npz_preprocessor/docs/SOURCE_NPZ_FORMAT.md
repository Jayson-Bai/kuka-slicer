# External Source NPZ Format

本文档规定 `external_npz_preprocessor` 期望读取的外部源 NPZ 格式。这个 NPZ 是外部建模/切片软件交给本系统的输入格式，不是 `path_processing_core.npz_exporter` 写出的系统运行时 NPZ。

读取入口是：

```text
external_npz_preprocessor.source_npz.load_source_npz()
```

转换链路是：

```text
外部源 NPZ
-> load_source_npz()
-> SourceJob / LayerPaths / MaterialPath
-> source_job_to_parsed_commands()
-> path_processing_core.npz_exporter.export_npz()
-> 系统运行时 NPZ
```

## Archive Keys

源文件必须是标准 `.npz`。层和材料使用一个 key 表示：

```text
layer_0000_R
layer_0000_F
layer_0001_R
layer_0001_F
...
```

key 必须匹配：

```text
^layer_(\d{4})_([RF])$
```

含义如下：

| 部分 | 含义 |
| --- | --- |
| `layer_0000` | 4 位十进制层号。层号用于排序和分组，允许不连续。 |
| `R` | 树脂路径 resin。 |
| `F` | 纤维路径 fiber。 |

`meta` 是可选 key。除 `meta` 和匹配 `layer_xxxx_R/F` 的 key 外，其他 key 当前会被忽略。

至少必须存在一个有效的 `layer_xxxx_R` 或 `layer_xxxx_F` key；否则加载失败。

## Path Array Shape

推荐格式是数值型三维数组：

```text
[path_count, max_points_per_path, columns]
```

要求：

| 维度 | 含义 |
| --- | --- |
| `path_count` | 当前层、当前材料下的路径数量。 |
| `max_points_per_path` | 当前数组中最长路径的点数。短路径用 padding 补齐。 |
| `columns` | 点列数，只允许 `3` 或 `6`。 |

推荐 dtype 是 `float32`。loader 会把路径点转成 `float32`，但生成方应直接写 `float32`，便于用 `np.load(..., allow_pickle=False)` 读取并避免 object 序列化。

## Point Columns

有效路径只允许两种点格式：

```text
Nx3: [x, y, z]
Nx6: [x, y, z, a, b, c]
```

规则：

- 每条路径必须是二维数组。
- 每条路径至少 2 个有效点。
- `Nx2`、`Nx5` 或其他列数无效。
- `Nx3` 会在加载时追加配置中的默认 `a/b/c`，归一化为 `Nx6`。
- `Nx6` 会保留源文件里的 `a/b/c`。

`x/y/z` 单位是 mm。`a/b/c` 是姿态字段，单位和含义沿用机器人侧当前姿态约定；如果外部软件不负责姿态，应使用 `Nx3` 并让 UI/参数提供默认姿态。

## Padding Rules

数值型三维数组用整行 `NaN` 表示 padding：

```text
[
  [x0, y0, z0],
  [x1, y1, z1],
  [nan, nan, nan]
]
```

有效行和 padding 行的判定规则：

- 一行所有列都是 `NaN`：padding 行，会被删除。
- 一行没有 `NaN`：有效点。
- 一行只有部分列是 `NaN`：非法，加载失败。

padding 只能出现在路径末尾。即使中间出现整行 `NaN`，loader 当前也会删除该行并继续保留后续有效点；生成方不应依赖这种容错，应始终把 padding 放在末尾。

## Legacy Object Arrays

当前 loader 仍兼容旧 object array：如果某个 `layer_xxxx_R/F` 的 dtype 是 `object`，会把每个元素当作一条 path 读取。

这个格式只用于历史兼容，不推荐新文件使用。新文件应使用数值型三维 `float32` 数组和整行 `NaN` padding。

## Z Ownership

源 NPZ 必须显式提供 Z。preprocessor 不会根据层号、层高或 UI 参数生成或覆盖轨迹 Z。

这条规则用于支持曲面切片：同一层、同一路径内的 Z 可以变化。转换时三维路径长度按：

```text
sqrt(dx^2 + dy^2 + dz^2)
```

计算，因此 Z 起伏会参与打印距离和挤出量计算。

UI 中的树脂层高和纤维层高只作为工艺参数：

- 树脂层高参与树脂 `E/mm` 计算；树脂体积按固定 1.75 mm 耗材截面积换算成 E 轴料长。
- 纤维层高当前作为纤维工艺参考。
- 纤维起步加速时间只影响外部 NPZ 纤维打印路径的七阶起始加速段。
- 两者都不改变源文件中的 Z。

## Coordinate Offsets

源 NPZ 中的 `x/y/z/a/b/c` 是源几何坐标。preprocessor 转命令时会先计算整个源零件的 XY 最小点，再把这个左下角对齐到 UI 中设置的位置：

```text
x_prime = x - min(source_x) + start_x_mm
y_prime = y - min(source_y) + start_y_mm
z_prime = z
```

喷头共享偏置、工具切换补偿和树脂 Z 打印补偿不属于源 NPZ 格式，由最终 `path_processing_core.npz_exporter.export_npz()` 在系统 NPZ 导出阶段统一处理。

## Meta

`meta` 可选。如果存在，必须是 JSON string，解析后必须是 JSON object。推荐内容：

```json
{
  "format": "external_layer_paths_v1",
  "unit": "mm",
  "point_columns": ["x", "y", "z"],
  "materials": {
    "R": "resin",
    "F": "fiber"
  },
  "description": "Layer/material path arrays for external_npz_preprocessor"
}
```

当前 loader 不强制校验 `format`、`unit` 或 `point_columns` 的具体值，但建议生成方写入这些字段，方便人工检查和后续版本迁移。

## Processing Order

加载后得到的结构是：

```text
SourceJob
  meta
  layers: list[LayerPaths]
    index
    resin_paths: list[MaterialPath]
    fiber_paths: list[MaterialPath]
```

排序规则：

- 层按 key 中的层号升序处理。
- 同一层内，当前转换器先处理全部树脂路径，再处理全部纤维路径。
- 同一层、同一材料内，路径顺序等于数组第一维顺序。

如果上一条路径终点和下一条路径起点不同，preprocessor 会插入空走 travel。

## Material Semantics

材料映射：

| 源材料 | 含义 | 转换时 GCode tool | exporter 内部工具 |
| --- | --- | ---: | --- |
| `R` | 树脂 | `T1` | 系统树脂工具 |
| `F` | 纤维 | `T0` | 系统纤维工具 |

每条源路径会转成独立打印段。整件第一条可打印路径在 prime 前保留一次初始回抽；每条路径到达起点后执行 `prime -> 可选 prime_settle -> PRINT`。

树脂路径结束顺序为 `PRINT -> retract -> path reset -> one-cycle E=0 anchor`。每层最后一条源树脂路径还会在 anchor 后、任何换刀或抬升前，保持当前 Z 和姿态，沿该层树脂 XY 包围盒中心指向路径终点的方向向外 travel 20 mm；primeline 和纯纤维层不触发该动作。若它位于整件最后一层，该 travel 的终点就是运行时 NPZ 的最终轨迹点；`center_node` 等待 RSI 确认该最终序号后才触发 `ABORT`，因此现有安全抬升从外移终点开始。纤维路径先插入语义级 `CUT`；`path_processing_core.npz_exporter.export_npz()` 在 CUT 前建立 E=0 基准，再展开非阻塞 cut 事件、同步抬升/挤出、3 秒高位保持、独立 reset、等量回抽、3 秒高位保持、最终 reset 和剩余等待，之后才执行原有 `path reset -> one-cycle E=0 anchor`。如果还有下一条路径，travel 在 anchor 之后执行并保持 `E=0`。

该边界覆盖 converter 生成的 primeline、所有源路径和最终路径。路径 reset 与工具切换 reset 同时保留；reset 事件行仍是旧 E，只有精确内部 `external_npz_reset_anchor` 行从 `E=0` 导出。

这些规则属于加载后的命令生成和系统 NPZ 导出行为，不改变本文定义的外部源 NPZ schema。源文件生成方不需要增加 E、reset、anchor 或 settle 字段；`layer_xxxx_R/F` 的 key、shape、列和单位约束保持不变。

## Validation Failures

以下情况会加载失败：

- 文件不存在。
- 没有任何匹配 `layer_xxxx_R/F` 的 key。
- `meta` 不是合法 JSON string。
- `meta` 解析后不是 JSON object。
- 数值型材料数组不是三维数组。
- 路径不是二维数组。
- 路径有效点少于 2 个。
- 路径列数不是 3 或 6。
- padding 行存在部分 `NaN`。

## Minimal Example

```python
import json
import numpy as np

layer_0000_R = np.full((2, 3, 3), np.nan, dtype=np.float32)
layer_0000_R[0, :3, :] = [
    [0.0, 0.0, 0.50],
    [30.0, 0.0, 0.50],
    [30.0, 20.0, 0.50],
]
layer_0000_R[1, :2, :] = [
    [5.0, 5.0, 0.50],
    [25.0, 5.0, 0.50],
]

layer_0000_F = np.array([
    [
        [2.0, 2.0, 0.60],
        [28.0, 18.0, 0.60],
    ]
], dtype=np.float32)

meta = {
    "format": "external_layer_paths_v1",
    "unit": "mm",
    "point_columns": ["x", "y", "z"],
    "materials": {"R": "resin", "F": "fiber"},
}

np.savez(
    "source.npz",
    meta=np.array(json.dumps(meta, ensure_ascii=False)),
    layer_0000_R=layer_0000_R,
    layer_0000_F=layer_0000_F,
)
```

## Template File

仓库会生成一份两层树脂/纤维模板：

```text
data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz
```

该模板包含：

```text
meta
layer_0000_R
layer_0000_F
layer_0001_R
layer_0001_F
```

模板使用数值型 `float32` 三维数组和整行 `NaN` padding，可用 `np.load(..., allow_pickle=False)` 读取。
