# 纤维 JSON 格式对比与标准格式建议

本文档对比当前收到的三类纤维 JSON，并提出一个以“每一条纤维轨迹”为最小单位的标准格式。

本文档定义当前 UI 已支持的数据格式和推荐的外部交换格式。

## 1. 三个文件的实际结构

### 1.1 原始 S 文件

文件：`two_hole_tension_specimen_planar_S_family_sorted.json`

实际结构：

```text
顶层对象
├── S+-family: 路径列表，共 35 条
│   ├── 路径 0: 点列表，共 63 点
│   ├── 路径 1: 点列表，共 89 点
│   └── ...
└── S--family: 路径列表，共 38 条
    ├── 路径 0: 点列表，共 60 点
    ├── 路径 1: 点列表，共 83 点
    └── ...
```

单个点的格式是：

```json
{"x": 15.000000954, "y": 46.32623175045949}
```

该文件共有 73 条纤维轨迹。点格式清晰，主要问题是路径没有显式的 `path_id`、`sequence` 和统一的顶层格式描述。

### 1.2 原始 T 文件

文件：`two_hole_tension_specimen_planar_T_family_sorted.json`

实际结构：

```text
顶层对象
└── T-family: 路径列表，共 78 条
    ├── 路径 0: 点列表，共 44 点
    ├── 路径 1: 点列表，共 40 点
    └── ...
```

它和 S 文件的单条路径结构一致，只是顶层路径族不同。

### 1.3 当前最新优化结果文件

文件：`two_hole_tension_specimen_planar__contours-25__offset-1.5mm__metric-geodesic__conn-max-11p8mm__20260728T120819.json`

实际顶层字段包括：

```text
contour_count
contour_offset_mm
stage1_metric
max_connector_length_mm
domain
planning_domain
records
stage1_connectors
accepted_connectors
final_paths
diagnostics
scalar_source
```

这个文件不是单纯的纤维路径文件，而是一个优化过程结果包：

| 字段 | 当前内容 | 是否等价于打印轨迹 |
|---|---|---|
| `records` | 原始输入轨迹记录，共 52 条 T 轨迹 | 最接近单条轨迹，但带有记录元数据 |
| `stage1_connectors` | 第一阶段候选连接器，共 51 条 | 否，属于连接/空走信息 |
| `accepted_connectors` | 最终接受的连接器，共 46 条 | 否，属于连接/空走信息 |
| `final_paths` | 合并连接后的最终路径，共 6 条 | 否，不再是一条原始轨迹对应一条路径 |
| `domain`、`planning_domain` | 外部轮廓和规划域 | 否，属于几何元数据 |
| `diagnostics` | 优化统计和求解信息 | 否，属于诊断信息 |

### 1.4 关键差异

原始 S/T 文件：

```text
family -> [path, path, ...] -> [[point, point, ...], ...]
```

最新优化结果：

```text
result -> records / connectors / final_paths / diagnostics / metadata
```

最新文件的 `final_paths` 只有 6 条，是把多个原始轨迹通过连接器合并后的结果。因此它不能作为“每条纤维轨迹一个路径”的标准原始输入。

## 2. 建议的标准格式

建议定义一个明确版本的格式：`external_fiber_paths_v1`。

核心原则：

1. 顶层只保留格式信息和一个 `paths` 数组；
2. `paths` 中每个对象只表示一条纤维轨迹；
3. 每条轨迹有稳定的 `path_id`、所属 `family` 和顺序 `sequence`；
4. 轨迹点只表示打印几何，不把连接器、空走或诊断信息混进来；
5. 如果外部算法生成了连接器，应放到独立字段，不能拼进普通打印轨迹的 `points`；
6. 当前 UI/Core 仍负责层间复制、路径间 travel、平滑、采样、E 和机器人动作。

推荐结构：

```json
{
  "format": "external_fiber_paths_v1",
  "unit": "mm",
  "coordinate_system": "project_default",
  "point_columns": ["x", "y"],
  "path_order": "sequence",
  "paths": [
    {
      "path_id": "S+-0000",
      "family": "S+-family",
      "source_index": 0,
      "sequence": 0,
      "closed": false,
      "points": [
        {"x": 15.000000954, "y": 46.32623175045949},
        {"x": 15.500000000, "y": 45.900000000},
        {"x": 16.000000000, "y": 45.400000000}
      ]
    },
    {
      "path_id": "T-0000",
      "family": "T-family",
      "source_index": 0,
      "sequence": 1,
      "closed": false,
      "points": [
        {"x": 57.51476867017256, "y": 155.0},
        {"x": 58.100000000, "y": 149.500000000},
        {"x": 59.457941538496414, "y": 133.98371064784067}
      ]
    }
  ]
}
```

## 3. 标准字段定义

### 3.1 顶层字段

| 字段 | 类型 | 必选 | 说明 |
|---|---|---:|---|
| `format` | string | 是 | 固定为 `external_fiber_paths_v1` |
| `unit` | string | 是 | 固定为 `mm` |
| `coordinate_system` | string | 是 | 项目统一 XY 坐标系，建议 `project_default` |
| `point_columns` | string array | 是 | 当前固定为 `["x", "y"]` |
| `path_order` | string | 是 | 固定为 `sequence` |
| `paths` | object array | 是 | 每个元素是一条独立纤维轨迹 |
| `connectors` | object array | 否 | 连接器诊断信息；当前不作为打印轨迹或 Core travel 读取 |
| `metadata` | object | 否 | 外部算法信息，不能替代必选字段 |

### 3.2 单条轨迹字段

| 字段 | 类型 | 必选 | 说明 |
|---|---|---:|---|
| `path_id` | string | 是 | 全文件唯一，例如 `S+-0000` |
| `family` | string | 是 | 轨迹族，例如 `S+-family`、`S--family`、`T-family` |
| `source_index` | integer | 否 | 在原始族中的索引 |
| `sequence` | integer | 是 | 打印前的轨迹顺序，从 0 开始且不重复 |
| `closed` | boolean | 是 | 是否闭合；开放轨迹为 `false` |
| `points` | object array | 是 | 当前轨迹的 XY 点列表 |

### 3.3 点字段

```json
{"x": 10.0, "y": 20.0}
```

要求：

- `x`、`y` 单位为 mm；
- 必须是有限数值；
- 每条轨迹至少 2 个点；
- 点顺序就是该轨迹的几何方向；
- 不使用 `NaN`、`Inf` 或字符串坐标；
- 当前不在纤维 JSON 中传递 `z`、`a`、`b`、`c` 或 `E`。

## 4. 连接器应该如何处理

如果外部优化器需要记录连接器，可以单独保存：

```json
{
  "connector_id": "C-0000",
  "from_path_id": "T-0000",
  "to_path_id": "T-0001",
  "points": [
    {"x": 40.0, "y": 30.0},
    {"x": 45.0, "y": 32.0}
  ],
  "length_mm": 5.385164807,
  "routing_mode": "geodesic"
}
```

连接器必须与 `paths` 分开。不能把连接器点直接拼入某条纤维轨迹的 `points`，否则下游会把连接器当成纤维打印几何，而不是 travel。

纤维 JSON 只提供纤维打印几何路径。当前不读取 `connectors`，也不会把它们复制到每个纤维层的 `layer_NNNN_T`；纤维路径之间的转场空走仍由 Core 按现有逻辑生成，并执行统一的空走速度、E=0、七阶参数化和采样逻辑。

## 5. 与当前 UI 的兼容关系

当前 UI 仍支持：

1. 顶层路径列表；
2. 顶层路径族对象，例如原始 S/T 文件；
3. 带 `final_paths` 的最新优化结果文件。

本标准格式使用顶层 `paths`，优点是结构最明确、每条轨迹有稳定身份、元数据不会与路径族混在一起。当前 UI 支持明确的 `paths` 解析分支，可以直接上传该格式；空走不在此 JSON 中提供。

建议后续兼容策略：

```text
标准格式 external_fiber_paths_v1
        ↓ 读取 paths
当前内部统一格式 list[one_path]
        ↓ 复用现有层映射与 Core 流程
最终 Core NPZ
```

原始 S/T 文件可以无损转换为该标准格式：

- `S+-family` 中第 0 条路径变成 `path_id = S+-0000`；
- `S--family` 中第 0 条路径变成 `path_id = S--0000`；
- `T-family` 中第 0 条路径变成 `path_id = T-0000`；
- 原始点坐标和点顺序保持不变；
- 不生成连接器，不合并路径，不重新采样。

## 6. 结论

原始 S/T 格式的几何层级其实是清楚的，但缺少显式的轨迹身份和统一版本信息。最新优化结果格式的问题不是点格式错误，而是把“输入轨迹、连接器、合并后的最终路径、诊断信息”放在同一个顶层结果对象中，导致它不适合作为单纯的纤维轨迹输入。

建议将 `external_fiber_paths_v1` 作为标准交换格式：

```text
一个 JSON 文件
    └── paths
        └── 一个对象 = 一条纤维轨迹
            └── points = 该轨迹的几何点
```

当前 UI 仍保留 S/T 路径族格式和旧优化结果格式的兼容读取；新生成的外部 JSON 建议统一使用上述标准 `paths` 格式。
