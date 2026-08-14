# 曲面可打印性验证器

`surface_validator` 是只读模块。它配对读取 `flat.npz`、`curved.npz` 与
`graded_surface_v1.json`，生成 JSON（可选 HTML）报告；不会改变路径、E、Prusa
配置或 Core。

```powershell
python -m kuka_slicer surface-validate flat.npz curved.npz graded_surface_v1.json report.json --html report.html
```

报告检查路径/逻辑层/XY/NaN 契约、曲面与源模型身份、逐点 Z 映射、相邻逻辑层可
配对点的间隙、局部坡度与采样密度、已有 `_E` 的弧长补偿、以及 T 空走风险。

生产蜂窝 NPZ 已包含 `path_roles` 和 `honeycomb_centerline_pathing`，可说明哪条路径是
外轮廓、哪些路径是蜂窝墙，并声明生产规划器采用“原始 STL 墙边一次沉积”的契约。
但验证器当前没有把 STL 实体作为第四个输入重新构造孔洞拓扑，因此它只能证明
“映射前后 XY 未改变且生产契约未丢失”，不能独立证明任意外部 NPZ 都没有新增封孔
线段。该项保留 `warning` 是证据范围提示，不表示已经发现蜂窝孔洞被封闭。

当前 T 风险检查同样只是路径点级筛查，不是实体喷头的扫掠体碰撞检测。后续应在
三维预览接入喷头装配实体后，以喷头实体、姿态和已沉积几何执行完整扫掠碰撞检查。
