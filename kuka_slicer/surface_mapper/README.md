# 曲面映射器

`surface_mapper` 是独立于切片器、曲面预览器和 Core 的中间路径工具。

它读取：

- 平面切片结果：`external_layer_paths_v1` 格式的 `flat.npz`；
- 曲面预览器导出的几何定义：`graded_surface_v1.json`。

它输出可直接送入既有 Core 前处理器的 `curved.npz`。逻辑层键
`layer_XXXX_R/F/T`、XY、材料分组、逻辑层号及路径/点顺序均原样保留；
只有逐点 Z 会按照映射器配置的逻辑层完成度改变：

```text
z_mapped = z_flat + alpha(logical_layer) * H(x, y)
```

启动本地界面：

```powershell
python -m kuka_slicer surface-map
```

默认地址为 <http://127.0.0.1:8767>。映射器只接收一个“曲面起始层”：它会围绕
整个 NPZ 的中间逻辑层自动生成镜像的平—曲—平完成度，顶部恢复为平面。该策略
只在本包中配置，不会写入目标曲面 JSON。预览与导出前均会拒绝包含负 Z 的结果。

导出时，已有 `layer_XXXX_R_E` / `layer_XXXX_F_E` 会按每段曲面三维弧长
替换为新的累计 E；没有 E 的路径不会被新增 E。该补偿仅覆盖路径弧长，不包含
喷嘴姿态、层间间隙或条带横截面积的体积模型。姿态 ABC 生成和 Core 内部轨迹
修改仍不属于本包。
