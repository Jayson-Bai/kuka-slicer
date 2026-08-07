# 曲面可打印性验证器

`surface_validator` 是只读模块。它配对读取 `flat.npz`、`curved.npz` 与
`graded_surface_v1.json`，生成 JSON（可选 HTML）报告；不会改变路径、E、Prusa
配置或 Core。

```powershell
python -m kuka_slicer surface-validate flat.npz curved.npz graded_surface_v1.json report.json --html report.html
```

报告检查路径/逻辑层/XY/NaN 契约、曲面与源模型身份、逐点 Z 映射、相邻逻辑层可
配对点的间隙、局部坡度与采样密度、已有 `_E` 的弧长补偿、以及 T 空走风险。NPZ
没有外轮廓或蜂窝孔洞的语义标记，因此该部分会明确标为需要人工确认的 `warning`。
