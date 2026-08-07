# 曲面映射器

`surface_mapper` 是独立于切片器、曲面预览器和 Core 的中间路径工具。

它读取：

- 平面切片结果：`external_layer_paths_v1` 格式的 `flat.npz`；
- 曲面预览器导出的几何定义：`graded_surface_v1.json`。

它输出可直接送入既有 Core 前处理器的 `curved.npz`。逻辑层键
`layer_XXXX_R/F/T`、XY、材料分组、路径/点顺序与已有 E 数值均原样保留；
只有逐点 Z 会按照映射器配置的逻辑层完成度改变：

```text
z_mapped = z_flat + alpha(logical_layer) * H(x, y) + z_offset
```

启动本地界面：

```powershell
python -m kuka_slicer surface-map
```

默认地址为 <http://127.0.0.1:8767>。映射策略（起止逻辑层、线性/平滑渐变、
自动或手动 Z 抬升）只在本包中配置，不会写入目标曲面 JSON。

本版本刻意不做 E 重算、姿态 ABC 生成或 Core 内部轨迹修改。
