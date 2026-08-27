# KUKA Slicer 外部纤维 JSON 输入说明

本文档说明当前 UI 中“纤维路径 JSON”输入文件的格式、处理范围以及输出前的路径逻辑，供外部路径规划器对接使用。

## 1. 输入文件在整个流程中的作用

UI 导出任务需要：

- 一个 STL 文件：必选，用于 Prusa 内核生成树脂路径；
- 一个纤维 JSON 文件：可选，用于提供纤维的 XY 路径模板；
- UI 中的切片参数和 `process_core` 参数：由 UI 单独提交，不写入纤维 JSON。

纤维 JSON 不是最终机器人 NPZ，也不是按层的完整运动文件。它只提供一组纤维路径模板。UI 会将这些路径放到树脂层之间，再由当前 Core 继续执行平滑、七阶参数化采样、E 计算、运动动作和最终 Core NPZ 导出。

因此，外部 JSON 中不要提前加入：

- 机械头偏置；
- 树脂或纤维 Z 补偿；
- CUT 抬升、CUT 等待；
- 起步、回抽、补偿移动；
- Core 的 travel；
- 机器人姿态或最终采样点；
- 最终累计 E。

## 2. 推荐的简单格式

最简单的格式是“顶层路径列表”：

```json
[
  [
    [10.0, 20.0],
    [11.0, 20.5],
    [12.0, 21.0]
  ],
  [
    {"x": 30.0, "y": 15.0},
    {"x": 31.0, "y": 16.0}
  ]
]
```

也可以使用“路径族”对象。对象的每个值都必须是路径列表：

```json
{
  "S+-family": [
    [[10.0, 20.0], [11.0, 20.5]]
  ],
  "S--family": [
    [{"x": 30.0, "y": 15.0}, {"x": 31.0, "y": 16.0}]
  ]
}
```

路径族名称只用于分组，不会作为材料、层号或工艺参数使用。UI 会按照 JSON 中的族和路径顺序合并读取。

## 3. 当前优化结果 JSON 格式

外部优化器也可以发送带元数据的结果文件。此格式必须把最终纤维路径放在 `final_paths` 中：

```json
{
  "format": "fiber_path_optimization_result_v1",
  "contour_count": 25,
  "contour_offset_mm": 1.5,
  "stage1_metric": "geodesic",
  "domain": {
    "outer": [],
    "holes": []
  },
  "diagnostics": {},
  "accepted_connectors": [],
  "final_paths": [
    {
      "path_id": "optimized-0000",
      "source_family": "optimized",
      "source_index": 0,
      "closed": false,
      "points": [
        {"x": 10.0, "y": 20.0},
        {"x": 11.0, "y": 20.5},
        {"x": 12.0, "y": 21.0}
      ]
    }
  ]
}
```

当前 UI 的读取规则是：

1. 如果顶层存在 `final_paths`，只读取 `final_paths`；
2. `contour_count`、`domain`、`diagnostics`、`accepted_connectors` 等字段只作为元数据，不会被当成路径族；
3. 每个 `final_paths` 元素必须包含 `points` 列表；
4. `path_id`、`source_family`、`source_index`、`closed` 等字段不会改变路径几何；
5. `accepted_connectors` 不会被单独作为纤维打印路径读取。

这就是之前报错的原因：旧格式解析器把所有顶层字段都当成路径族，所以把整数 `contour_count` 错误地当成路径列表。现在已兼容该复合结果格式，同时保留了上面的简单格式。

## 4. 点和路径的要求

### 4.1 推荐点格式

推荐使用二维点：

```json
[x, y]
```

或：

```json
{"x": x, "y": y}
```

单位为 mm，坐标必须使用与 STL、打印平面一致的 XY 坐标系。

当前读取器也接受长度大于等于 2 的数组，例如 `[x, y, z]`，但纤维 JSON 中的第三列不会被保留为最终 Z。为避免误解，正式输入建议只使用 `[x, y]` 或 `{"x": ..., "y": ...}`。

### 4.2 路径要求

- 每条有效路径至少有 2 个点；
- 不同路径可以有不同点数；
- 点必须是有限数值，不能使用 `NaN` 或 `Inf`；
- 不要把路径填充成带 NaN 的 NPZ 数组后再转成 JSON；
- 不要在路径中插入空点或空数组；
- 路径坐标不要提前应用机械头偏置、Z 补偿或 Core 工艺动作；
- 当前纤维 JSON 输入只使用 XY，不支持通过 JSON 输入 A/B/C 姿态或 E 值。

路径少于 2 个点时不会形成有效纤维路径，建议外部生成器在发送前直接拒绝这类路径。

## 5. 层映射规则

纤维 JSON 本身不携带 `layer_0000`、`layer_0001` 等层号。当前 UI 按以下规则生成纤维层：

- 纤维模板会复制到每个零件树脂层之间；
- 筏层不放置纤维路径；
- 最后一层树脂作为顶部封层，不再放置下一条纤维路径；
- 纤维层的 Z 由 UI/Core 的纤维层高和树脂层计划计算；
- JSON 中提供的 Z（如果存在）会被忽略；
- 纤维路径会与树脂路径一起进行 XY 起点归一化，归一化目标由 UI 的零件起始 XY 设置决定。

例如，有 4 个零件树脂层且没有筏层时，通常会生成 3 个纤维层，分别位于树脂层 0/1、1/2、2/3 之间，而不是在最后一层之后再生成一层纤维。

## 6. 路径顺序和方向

输入路径会保留其几何点，但在复制到每个纤维层时，当前 UI 会调用开放路径 travel 优化：

- 第一条路径作为起始路径；
- 后续路径按与当前末端的距离进行贪心排序；
- 开放路径必要时会反向，以减少路径间空走距离；
- 已闭合路径不会被反向；
- 这一步不会把不同路径合并成一条打印路径，也不会在 JSON 中增加 Core 运动点。

所以，如果外部规划器要求绝对保持最终路径顺序和方向，当前 JSON 接口并不承诺这一点；应提供几何路径模板，由 UI/Core 负责层内路径顺序和后续运动处理。

## 7. 推荐发送前校验

外部生成器发送前至少应检查：

```python
import json
import math


def validate_xy_path(path):
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError("each fiber path needs at least two points")

    for point in path:
        if isinstance(point, dict):
            x, y = point["x"], point["y"]
        else:
            x, y = point[0], point[1]
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError("fiber coordinates must be finite")


def validate_fiber_json(path_name):
    with open(path_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    if isinstance(data, dict) and "final_paths" in data:
        paths = [item["points"] for item in data["final_paths"]]
    elif isinstance(data, dict):
        paths = [path for family in data.values() for path in family]
    elif isinstance(data, list):
        paths = data
    else:
        raise ValueError("fiber JSON must be a list or supported object")

    for fiber_path in paths:
        validate_xy_path(fiber_path)
```

## 8. 与最终 NPZ 的关系

上传纤维 JSON 后，UI 会先生成 Prusa/外部源 NPZ，再调用当前 `process_core` 导出最终 Core NPZ。最终源数据使用稳定的层字段：

```text
meta
layer_0000_R
layer_0000_F
layer_0001_R
layer_0001_F
...
```

空材料层也会保留对应的空数组字段。JSON 输入不会直接成为机器人消费文件；机器人侧应使用 UI 导出的最终 Core NPZ。

## 9. 一句话接口约定

外部只需要提供与打印平面同坐标系的纤维 XY 路径模板：简单格式使用顶层路径列表或路径族对象；优化结果格式使用 `final_paths[].points`。不要在 JSON 中提供层号、Z、E、姿态、偏置或 Core 动作，这些由当前 UI、Prusa 内核和 process_core 负责。
