# 图层 PNG 预览查看器 — 设计文档

## 概述

在 NPZ 导出完成后，提供一个按钮弹出图片查看器，显示每一层的 XY 轨迹 PNG，支持左右切换、滚轮缩放和拖拽平移。

## 变更清单

| 文件 | 变更内容 |
|------|----------|
| `npz_exporter.py` | 修改 `_plot_single_layer()` PNG 输出路径 |
| `ui_panel.py` | 新增 `_LayerViewerDialog`，添加"查看图层"按钮，连接导出完成信号 |

## 1. PNG 导出目录变更

### 当前
`_plot_single_layer()` 保存到 `{data_root}/layer_{n:04d}/layer_{n:04d}.png`

### 改为
`{data_root}/layer_previews/layer_{n:04d}.png`

- `layer_previews/` 目录在首次写入时自动创建（`os.makedirs(exist_ok=True)`）
- 不影响现有的 `layer_NNNN/` NPZ 数据目录

## 2. UI 按钮

### 位置
GCode Export 面板内，Export NPZ 按钮下方。

### 行为
- **初始状态**：`disabled`（无导出记录时不可点）
- **导出成功后**：`enabled`，且 `self._last_npz_dir` 记录 NPZ 输出目录路径
- **点击后**：弹出 `_LayerViewerDialog`

## 3. 弹窗查看器 `_LayerViewerDialog`

### 定义
`class _LayerViewerDialog(QtWidgets.QDialog)`，新增于 `ui_panel.py`

### 构造参数
- `npz_dir: str` — NPZ 输出目录路径

### 初始化流程
1. 扫描 `{npz_dir}/layer_previews/` 下所有 `layer_*.png`
2. 按层号数字排序（`sorted(key=lambda x: int(re.search(r'layer_(\d+)', x).group(1)))`）
3. 若无图片，显示提示信息
4. 初始化显示第一张

### 布局

```
┌──────────────────────────────────────────┐
│  ← Prev   图层 3/12   Next →    [X]      │  ← 顶部导航栏 (QHBoxLayout)
├──────────────────────────────────────────┤
│                                          │
│              QGraphicsView              │
│          (图片显示 + 缩放/平移)          │
│                                          │
├──────────────────────────────────────────┤
│  缩放: 100%          [重置视图]  [关闭]   │  ← 底部状态栏 (QHBoxLayout)
└──────────────────────────────────────────┘
```

### 图片显示
- 使用 `QGraphicsView + QGraphicsScene`
- 加载 PNG 为 `QPixmap`，添加到 scene
- 初始调用 `fitInView()` 适配窗口
- 拖拽模式：`ScrollHandDrag`（按住拖动平移）

### 滚轮缩放
- 重写 `wheelEvent`：
  - 向上滚：放大（scale 1.15x）
  - 向下滚：缩小（scale 1/1.15x）
  - 缩放以鼠标位置为中心（`setTransformationAnchor(AnchorUnderMouse)`）
- 缩放比例限制：0.1x ~ 20x
- 实时更新底部缩放比例显示

### 切换逻辑
- "← Prev" 按钮：切换到上一张（若已在第一张则不做任何事）
- "Next →" 按钮：切换到下一张（若已在最后一张则不做任何事）
- 切换时更新图片、更新"图层 X/Y"文字、重置缩放
- 快捷键：`←` / `→` 键也触发切换（`keyPressEvent`）

### 重置按钮
- 调用 `fitInView()` 恢复适配窗口
- 重置缩放比例为 100%

### 窗口属性
- 初始大小：900x700
- 标题："图层预览 - {npz 文件名}"

## 4. 数据流

```
导出完成
  │
  ├─ _on_export_finished(success, msg)
  │     └─ if success: self._last_npz_dir = npz 输出目录
  │                     self._view_layers_btn.setEnabled(True)
  │
  └─ 用户点击"查看图层"
        │
        ├─ _on_view_layers()
        │     └─ dlg = _LayerViewerDialog(self._last_npz_dir)
        │           dlg.exec_()
        │
        └─ 弹窗内:
              ├─ 扫描 layer_previews/*.png
              ├─ 显示当前图片
              ├─ ←/→ 切换
              └─ 滚轮缩放 + 拖拽平移
```

## 5. 边界情况

| 场景 | 行为 |
|------|------|
| `layer_previews/` 不存在 | 弹窗显示"未找到图层图片" |
| 目录为空（0 张 PNG） | 弹窗显示"未找到图层图片" |
| 只有 1 张图 | Prev/Next 按钮 disabled |
| PNG 文件损坏 | `QPixmap.load()` 失败，弹窗显示"图片加载失败" |
| 导出未完成就点按钮 | 按钮 disabled，无法点击 |
