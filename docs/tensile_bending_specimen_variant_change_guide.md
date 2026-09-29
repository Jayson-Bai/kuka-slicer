# 拉伸版 / 弯曲版双正弦蜂窝试件适配指导

## 1. 文档目的

本文只定义后续代码修改边界，不实施代码修改，也不锁定零件长、宽、高、跨距或六边形边长。

目标是在现有蜂窝网格共形设计器中增加一个明确的“试件版本”选项：

- **拉伸版本**：完整保留当前两端夹持区、夹持区实心树脂填充、夹持分界墙，以及中央蜂窝工作段的现有构型和 slicer 处理逻辑。
- **弯曲版本**：只取消两端夹持区；蜂窝骨架、树脂连续路径和纤维连续路径的有效 X 区域从原中央工作段扩展为整个矩形零件区域；仍保留每层一条闭合外轮廓。

本次需求中的“铺满整个零件区域”应理解为：蜂窝有效裁剪域由 `[grip, length-grip] × [0, width]` 变为 `[0, length] × [0, width]`，不存在两端空白夹持段。它不要求蜂窝壁与 `X=0` 或 `X=length` 完全重合，也不取消为单层外轮廓预留的半线宽几何净空。

## 2. 需求分析与关键判断

### 2.1 试件版本必须与其他设计选项解耦

建议新增独立字段：

```text
specimen_variant = tensile | bending
```

该字段只决定“是否存在夹持区域”，不能隐式改变以下参数：

- 双正弦曲面的幅值、波数、波长或相位；
- `surface_parameter_mode`；
- 蜂窝边长、方向或相位对齐；
- 三点弯曲加载线对齐和检验点；
- 是否铺设纤维及纤维层位置；
- 零件长、宽、高；
- 外轮廓数量和线宽；
- Core 工艺参数。

特别注意：现有 `surface_parameter_mode=tensile_centered_wave_count` 是曲面参数化策略，不是试件几何版本。不能把它直接复用为“拉伸版/弯曲版”选择器，否则会把曲面相位规则和夹持区几何错误耦合。

### 2.2 后端已经具备零夹持区语义

当前实现已经将 `part.symmetric_grip_end_length_mm` 视为可选字段，并在缺失时按 `0.0 mm` 处理：

- `pipeline._symmetric_grip_end_length()` 缺省返回 `0.0`；
- `pipeline._central_lattice_bounds()` 在夹持长度为零时返回 `None`，从而使用完整矩形曲面域；
- `pipeline.run_conformal_lattice_pipeline()` 只在夹持长度大于零时生成夹持分界墙和夹持区之字填充；
- `continuous_course.build_continuous_course_plan()` 在夹持长度为零时自然得到 `(0, 0, length, width)` 的连续路径工作域；
- `fiber_reinforcement._continuous_work_bounds()` 明确兼容不含夹持字段的旧设计，并使用矩形外轮廓作为安全 Travel 边界；
- `path_bridge.to_external_base_source_job()` 已能只保留外轮廓，再追加连续蜂窝路径。

因此，弯曲版本不应复制或改写一套 slicer 算法。正确做法是让设计器导出“零夹持”的设计契约，并继续复用现有统一流水线。

### 2.3 单层外轮廓继续保留

当前设计器固定树脂喷嘴名义线宽为 `2 mm`，`wall_width_mm=2 mm` 时导出 `wall_bead_count=1`。`pipeline._layered_outer_boundary()` 每层生成一条闭合矩形外轮廓，`path_bridge` 将其标记为：

```text
conformal_outer_boundary
```

弯曲版本应继续保持每层恰好一条该角色路径，不增加内轮廓，也不能因为去除夹持分界墙而去掉外轮廓。

## 3. 目标数据契约

### 3.1 新导出的拉伸版本

建议新导出的 JSON 在 `part` 中显式记录版本，并继续输出夹持长度：

```json
{
  "part": {
    "boundary": "rectangle",
    "specimen_variant": "tensile",
    "length_mm": "待定",
    "width_mm": "待定",
    "final_height_mm": "待定",
    "symmetric_grip_end_length_mm": "待定且必须大于 0"
  }
}
```

### 3.2 新导出的弯曲版本

弯曲版本显式记录版本，但不写入夹持区字段：

```json
{
  "part": {
    "boundary": "rectangle",
    "specimen_variant": "bending",
    "length_mm": "待定",
    "width_mm": "待定",
    "final_height_mm": "待定"
  }
}
```

不建议在弯曲版本中写入一个仍为正数、但要求下游忽略的夹持长度。夹持字段缺失是当前流水线已经支持的零夹持表示，可使几何行为保持单一来源。

### 3.3 兼容旧 JSON

为了保留现有拉伸构型和历史设计文件，合同加载器应保持以下兼容规则：

| JSON 情况 | 处理规则 |
|---|---|
| 无 `specimen_variant`，夹持长度大于 0 | 完全沿用现有拉伸几何行为 |
| 无 `specimen_variant`，夹持字段缺失或为 0 | 完全沿用现有零夹持几何行为 |
| `specimen_variant=tensile` | 要求夹持长度存在、有限、为正且两端之和小于零件长度 |
| `specimen_variant=bending` | 要求夹持字段缺失或为 0 |
| 显式版本与夹持长度冲突 | 拒绝导出/加载，不能静默猜测 |

`specimen_variant` 是可追溯元数据和一致性校验字段；正式路径计算仍只根据有效夹持长度进入现有几何分支，不在 slicer 中新增第二套版本路由。

这是向 `conformal_lattice_spec_v1` 添加可选字段，可保持格式版本不变；前提是所有本项目消费者继续容忍 `part` 中的附加字段。若后续发现外部消费者使用严格白名单，再单独评估格式升级，不能在本修改中顺带扩展范围。

## 4. 设计器需要修改的位置

主要文件：`kuka_slicer/surface_preview/server.py`

### 4.1 HTML 控件

在“矩形实体”区域、尺寸输入之后、夹持长度之前增加试件版本选择器，例如：

```html
<select id="specimen_variant">
  <option value="tensile">拉伸版本（保留两端夹持区）</option>
  <option value="bending">弯曲版本（蜂窝铺满整个矩形）</option>
</select>
```

将现有 `grip_end_length_mm` 输入和提示包入可整体显示/隐藏的容器：

- 拉伸版本：显示并启用；保留当前默认值和用户已填写值。
- 弯曲版本：隐藏并禁用，但不要覆盖输入框中保存的拉伸夹持长度。

版本切换时保留拉伸长度非常重要。若切换到弯曲版时直接把输入值写成 `0`，用户切回拉伸版会丢失现有构型，不满足“保留拉伸版本”的要求。

### 4.2 状态保存和旧状态迁移

把 `specimen_variant` 加入 `persistedInputIds`，继续使用现有浏览器 `localStorage` 和本地状态文件保存机制。

旧的 `conformal_designer_state_v1.json` 没有该字段。恢复旧状态时建议：

- 若状态中没有 `specimen_variant` 且保存的 `grip_end_length_mm > 0`，界面选择拉伸版本；
- 若夹持字段缺失或为 0，可显示弯曲版本，但只改变界面解释，不重写历史文件；
- 当前默认和“重置”行为继续选择拉伸版本，以保持现有用户体验和回归兼容。

不需要仅为增加一个可选状态字段就更换状态文件名或 `localStorage` key。

### 4.3 统一计算“有效夹持长度”

新增一个前端辅助函数作为唯一入口，例如：

```text
effectiveGripEndLengthMm():
    tensile -> 读取并校验 grip_end_length_mm
    bending -> 0
```

随后让以下逻辑使用“有效夹持长度”，而不是直接读取输入框：

- `honeycombActiveXBounds()`；
- `updateConformalDesignSummary()`；
- `latticePreviewParameters()`；
- `continuousCoursePreview()`；
- 连续纤维预览的裁剪范围；
- 导出参数组装。

预期结果：

| 版本 | `honeycombActiveXBounds()` |
|---|---|
| 拉伸 | `[grip, length-grip]` |
| 弯曲 | `[0, length]` |

设计器中的蜂窝母板锚点继续取有效工作域中心。弯曲版本下，该中心就是整个矩形中心，因此黄色蜂窝骨架和红色连续路径会自然向两端顺延，而不是把原中央图案简单拉伸。

### 4.4 预览缓存和刷新

版本切换时必须执行：

- 同步夹持控件显示状态；
- 失效 `latticePreviewCache`；
- 失效 `continuousCoursePreviewCache`；
- 更新设计摘要和路径长度摘要；
- 重新渲染画布；
- 保存设计器状态。

当前连续路径缓存键已包含实际 `bounds`，只要有效边界正确变化，原则上不必再把版本名加入缓存键；显式加入也可以提高可读性，但不能代替边界更新。

### 4.5 导出请求

`conformalParameters()` 应显式提交 `specimen_variant`，并按版本提交有效夹持值：

- 拉伸版本：提交当前正的 `grip_end_length_mm`；
- 弯曲版本：提交 `0`，或不提交夹持参数；
- 不能把隐藏输入框内保留的拉伸长度误发给弯曲版本。

共形和对应的平面基线导出都应使用同一个版本选择，确保第一章“平面/双正弦”对比只有曲面形态不同，不会一边带夹持区、另一边不带夹持区。

导出文件名可以增加 `_tensile` 或 `_bending` 后缀，降低实验文件混用风险，但这只是易用性改进，不能成为 slicer 判定版本的依据。

### 4.6 界面文案

以下现有文案包含固定的拉伸/夹持假设，应改为按版本显示或使用中性表述：

- 页面标题下“默认采用拉伸标距段策略”的说明；
- 夹持区说明；
- 蜂窝预览中“夹持分界树脂带”的说明；
- 纤维“延伸到左右夹持区”的说明；
- “连续路径工作段”摘要；
- 与夹持区域有关的错误信息。

`applyTensilePreset` 目前只应被理解为曲面参数预设。它不应自动切换试件版本。后续可以把按钮文案改成“应用中心对称曲面参数组”，但不改变其数学行为。

## 5. 配置生成与合同校验需要修改的位置

### 5.1 `kuka_slicer/surface_preview/server.py`

在 `_rectangular_lattice_config_payload()` 中：

1. 读取并校验可选的 `specimen_variant`；
2. 对新请求执行版本与夹持长度一致性检查；
3. 在导出的 `part` 中记录 `specimen_variant`；
4. 仅在拉伸版本下写入正的 `symmetric_grip_end_length_mm`；
5. 对没有版本字段的旧调用继续沿用当前“按夹持长度决定”的行为。

不要把弯曲版理解为新的 `source_provider`。它仍可分别导出：

- `double_sine` 共形弯曲试件；
- `planar` 平面弯曲基线试件。

### 5.2 `kuka_slicer/conformal_lattice/contracts.py`

在 `_validate_rectangular_part()` 中增加对可选 `part.specimen_variant` 的校验，并执行第 3.3 节中的一致性规则。

`ConformalLatticeSpec.metadata()` 已经完整携带 `part`，无需增加另一份重复元数据。

## 6. slicer 路径流水线的处理原则

以下文件需要在实施时复核和补测试，但不应新增一套弯曲版算法。

### 6.1 `kuka_slicer/conformal_lattice/pipeline.py`

保留现有统一控制流：

```text
grip = _symmetric_grip_end_length(spec)
grip > 0:
    中央蜂窝域 + 两条分界墙 + 两端实心夹持填充
grip == 0:
    完整矩形蜂窝域 + 无分界墙 + 无夹持填充
```

弯曲版的预期状态为：

- `domain is full_domain`；
- `lattice_bounds is None`；
- `partition_connection.mode == "not_partitioned"`；
- `auxiliary_paths is None`；
- `_layered_outer_boundary()` 仍生成每层一条闭合外轮廓。

可修改注释和报告文案，将“central working region”等固定拉伸措辞改为“effective honeycomb region”，但不修改几何控制流。

### 6.2 `kuka_slicer/conformal_lattice/continuous_course.py`

`build_continuous_course_plan()` 已使用：

```text
bounds = (grip, 0, length-grip, width)
```

弯曲版在 `grip=0` 时自然变为完整矩形。保持以下逻辑不变：

- 外轮廓半线宽只用于孔洞/通道可打印性净空；
- 父蜂窝母板仍以有效工作域中心定位；
- 蜂窝按完整矩形裁剪；
- 边界截断的连续路径片段继续作为独立路径保留。

不要在弯曲版中把课程路径额外延伸到矩形之外，也不要重新引入“穿过夹持区”的扩展函数。

### 6.3 `kuka_slicer/conformal_lattice/path_bridge.py`

`to_external_base_source_job()` 已允许辅助路径为空。弯曲版每层在追加连续课程前应只保留：

```text
1 × conformal_outer_boundary
0 × conformal_partition_wall
0 × conformal_grip_zigzag_x_one_stroke
```

无需增加弯曲专用路径角色。

### 6.4 `kuka_slicer/conformal_lattice/fiber_reinforcement.py`

当前排序和安全 Travel 逻辑已将夹持墙/夹持填充当作可选集合，并允许只使用外轮廓作为边界。实施时应保持：

- `_continuous_work_bounds()` 在弯曲版返回 `(0, 0, length, width)`；
- `_order_resin_records_for_continuous_courses()` 在分界墙和夹持路径为空时仍正常排序；
- 纤维路径使用与树脂课程相同的完整矩形 XY 几何；
- 纤维路径之间的非挤出 Travel 仍通过外轮廓安全路由，不能用穿过蜂窝孔洞的直线弦连接。

现有函数说明中的“Only the central ...”和“preserve ... grips”等措辞应泛化，但不改变功能。

### 6.5 `kuka_slicer/ui_server.py`

主界面的 JSON 导入、统一 pipeline 调用、连续课程替换、纤维策略和 Core 导出不需要按 `specimen_variant` 新增分支。它们继续从设计合同中的有效夹持长度得到几何结果。

可以在结果报告中透传 `part.specimen_variant`，便于实验归档和界面确认；不得根据导出文件名猜测版本。

## 7. 明确不应修改的行为

- 不删除或重命名现有 `symmetric_grip_end_length_mm`。
- 不移除夹持区之字填充和分界墙生成函数。
- 不把当前拉伸版迁移成弯曲版默认值。
- 不复制 `run_conformal_lattice_pipeline()` 或 `build_continuous_course_plan()`。
- 不在弯曲版增加第二层内轮廓。
- 不把弯曲版本强制绑定到某组零件尺寸或蜂窝边长。
- 不自动启用 `align_load_line` 或弯曲检验点。
- 不自动改变双正弦波数、波长和相位。
- 不把夹持区实心填充替换成蜂窝后仍称为拉伸版本。
- 不修改 Core、上位机或外部 NPZ 格式；现有路径角色已经足以表达两个版本。

## 8. 必须增加或调整的测试

### 8.1 设计器和导出测试：`tests/test_surface_preview.py`

增加以下测试：

1. HTML 包含 `specimen_variant`，选项值严格为 `tensile` 和 `bending`。
2. 旧设计器状态缺少版本字段但夹持长度为正时恢复为拉伸版。
3. 切到弯曲版后夹持输入隐藏/禁用，但其原始拉伸数值仍保留，可切回恢复。
4. 拉伸导出的 `part` 同时包含版本和正的夹持长度。
5. 弯曲导出的 `part` 包含 `specimen_variant=bending`，且不包含正的夹持长度。
6. 弯曲版前端 `honeycombActiveXBounds()` 为 `[0, length]`。
7. 平面和双正弦导出使用同一试件版本。
8. 显式版本与夹持值冲突时返回清晰错误。
9. 版本切换会刷新蜂窝、连续路径和路径长度摘要。

### 8.2 pipeline 测试：`tests/test_conformal_lattice_pipeline.py`

为零夹持弯曲构型增加关系型断言，不依赖尚未确定的最终实验尺寸：

- 生成域 XY 边界等于完整零件边界；
- `partition_connection.mode == "not_partitioned"`；
- 每层恰好一条闭合 `conformal_outer_boundary`；
- 每层没有 `conformal_partition_wall`；
- 每层没有 `conformal_grip_zigzag_x_one_stroke`；
- 蜂窝几何和连续课程都位于 `[0, length] × [0, width]`；
- 至少生成一条可制造连续课程。

### 8.3 连续路径和 Core 回归：`tests/test_ui_core_export.py`

增加成对测试：

| 断言 | 拉伸版 | 弯曲版 |
|---|---:|---:|
| `course_bounds_mm` | `(grip,0,L-grip,W)` | `(0,0,L,W)` |
| 外轮廓角色/层 | 1 | 1 |
| 分界墙角色/层 | 2 | 0 |
| 夹持填充角色/层 | 2 | 0 |
| 连续蜂窝路径 | 存在 | 存在并覆盖完整有效 X 域 |
| 纤维路径 | 与对应树脂课程 XY 一致 | 与对应树脂课程 XY 一致 |
| 最终 Core NPZ | 可生成 | 可生成 |

现有 `test_production_continuous_courses_replace_only_legacy_honeycomb_and_keep_grips()` 必须保留，作为拉伸版不被破坏的核心回归。

不要把最终论文零件尺寸、最终蜂窝边长或固定路径数量写死在新增的版本语义测试中；这些研究参数尚待确定，版本测试应验证边界关系和路径角色。

## 9. 验收标准

实现完成后必须同时满足：

1. 设计器可以明确选择“拉伸版本”或“弯曲版本”。
2. 切换版本不会丢失用户已经填写的拉伸夹持长度。
3. 拉伸版本的预览、导出 JSON、树脂路径、纤维路径和 Core 输出与当前行为一致。
4. 弯曲版本不存在两端夹持实心填充和夹持分界墙。
5. 弯曲版本的蜂窝骨架、树脂连续课程和纤维连续课程使用完整矩形有效域。
6. 两个版本每层都只有一条闭合外轮廓；弯曲版不增加内轮廓。
7. 设计器预览的有效边界与生产 `ContinuousCoursePlan.course_bounds_mm` 一致。
8. 弯曲版本不因选择版本而改变曲面参数、蜂窝参数、纤维开关或工艺参数。
9. 旧 JSON 和旧设计器状态继续可用。
10. 共形版本和平面基线版本都支持相同的拉伸/弯曲试件语义。

## 10. 建议实施顺序

1. 先为 `specimen_variant` 和旧 JSON 兼容补合同测试。
2. 在配置生成器中加入显式版本元数据和一致性校验。
3. 在设计器中增加版本选择、有效夹持长度函数和状态迁移。
4. 让所有预览、摘要和导出统一使用有效夹持长度。
5. 增加零夹持 pipeline、连续树脂、连续纤维和 Core 端到端测试。
6. 最后只做注释/文案泛化，确认没有为弯曲版本复制路径算法。

## 11. 本文暂不决定的参数

以下内容继续留待后续讨论，不应在本次版本适配中写死：

- 零件长度；
- 零件宽度；
- 零件厚度；
- 三点弯曲支承跨距和压头/支承半径；
- 六边形边长水平；
- 双正弦曲面幅值、波数、波长和相位水平；
- 纤维层数和具体层间位置；
- 是否启用加载线蜂窝特征对齐。

这些参数改变实验设计，但不改变本文定义的拉伸版/弯曲版软件架构。
