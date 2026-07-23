# 离线规划源码交接与外部 Codex 工作指令

## 1. 你的身份与唯一目标

你是该离线规划仓库的接收、审计和后续重构执行者。

当前阶段的目标不是立即修改算法，而是先确认源码完整、能够复现，并完整理解两条离线规划链路及其输出契约：

1. GCode 到系统 NPZ。
2. 外部源 NPZ 到系统 NPZ。

实时上位机继续负责加载、预览、排队、事件触发以及机器人和挤出机实时执行。离线仓库只负责在打印前生成可被现有上位机消费的文件。

## 2. 原始来源与可追溯基线

- 原始仓库：Jayson-Bai/kuka_ram_ws
- 独立交接仓库：https://github.com/Jayson-Bai/offline-path-planner.git
- 原始分支：main
- 原始提交：eb9091bcf405eaca2dddb07f2998bb3f25c12601
- 原始标签：offline-planner-source-v1
- 交接日期：2026-07-23
- 原始工作区在交接时为干净状态。

本仓库由原仓库副本通过路径过滤生成。过滤重写了提交哈希，但保留文件的内容、文件模式和 Git blob 完全不变。

## 3. 硬约束

在完成第一阶段审计并得到仓库所有者书面确认之前：

1. 不重构目录。
2. 不修改算法。
3. 不修改系统 NPZ schema、字段类型、单位、词表或 sidecar 命名。
4. 不删除 gcode_planner 中的兼容包装模块。
5. 不修改 handoff 目录中的清单和黄金产物。
6. 不修改标记为“只读消费契约参考”的上位机文件。
7. 不从原实时仓库自行复制额外文件。发现缺失依赖时只报告。
8. 不把实时通信、RSI、UART、center_node 或启动系统引入离线规划器。
9. 不以“测试未运行”为“测试通过”。
10. 不覆盖原始标签或强推主分支。

## 4. 仓库所有权边界

### 4.1 离线规划器拥有并可在后续阶段修改

- src/my_project/gcode_planner/
- src/my_project/external_npz_preprocessor/
- src/my_project/path_processing_core/
- scripts/estimate_npz_rows.py
- scripts/plot_gcode_xy.py
- scripts/plot_layers_from_manifest.py
- scripts/plot_npz_xy.py
- scripts/split_gcode_by_layer.py
- test/scripts/test_plot_npz_xy.py
- data/input_gcode/
- data/external_npz_preprocessor/
- 随仓库携带的离线设计文档

path_processing_core 是两条离线链路共同使用的真实实现，包含公共数据类型、B 样条、插值、校准、RSI 时间累计和统一 NPZ exporter。不得误认为 gcode_planner 中的同名兼容包装层是唯一实现。

### 4.2 只读消费契约参考，不属于离线规划器所有权

- src/my_project/control_center/include/control_center/npz_loader.hpp
- src/my_project/control_center/include/control_center/queue_manager.hpp
- src/my_project/control_center/src/npz_loader.cpp
- src/my_project/control_center/src/queue_manager.cpp
- src/my_project/my_project_interfaces/msg/TrajectoryPoint.msg
- src/my_project/my_project_interfaces/msg/PlannedEvent.msg
- src/my_project/my_project_ui/my_project_ui/ui_panel.py

这些文件只用于确认现有上位机如何消费离线输出。不得在本仓库内修改它们。若输出契约确实需要变化，先提交接口变更提案，等待上位机仓库单独批准和实现。

## 5. 克隆与接收验真

私有交接仓库地址：

    https://github.com/Jayson-Bai/offline-path-planner.git

请先确保当前 GitHub 身份已获该私有仓库只读权限，然后执行：

    git clone https://github.com/Jayson-Bai/offline-path-planner.git offline-path-planner
    cd offline-path-planner
    git fetch --tags
    git checkout offline-planner-handoff-v1
    git status --short

如果仓库所有者交付的是 offline-path-planner-v1.bundle，则改用：

    git clone -b main offline-path-planner-v1.bundle offline-path-planner
    cd offline-path-planner
    git checkout offline-planner-handoff-v1
    git status --short

git status 必须为空。

核对来源记录：

    sed -n '1,220p' EXTERNAL_CODEX_HANDOFF.md
    wc -l handoff/HANDOFF_PATHS.txt
    wc -l handoff/SOURCE_TREE.tsv
    wc -l handoff/SOURCE_SHA256SUMS

验证原始 124 个文件逐字节未变：

    sha256sum -c handoff/SOURCE_SHA256SUMS

预期结果为 124 项全部 OK。任何一项失败都立即停止，不得继续分析或修改。

验证黄金产物：

    sha256sum -c handoff/GOLDEN_SHA256SUMS

## 6. 已建立的测试基线

交接环境中执行的功能和契约测试：

    PYTHONPATH=src/my_project/path_processing_core:src/my_project/gcode_planner:src/my_project/external_npz_preprocessor \
    python3 -m pytest -q \
      src/my_project/path_processing_core/test \
      src/my_project/gcode_planner/test \
      src/my_project/external_npz_preprocessor/test \
      test/scripts/test_plot_npz_xy.py \
      --ignore=src/my_project/gcode_planner/test/test_copyright.py \
      --ignore=src/my_project/gcode_planner/test/test_flake8.py \
      --ignore=src/my_project/gcode_planner/test/test_pep257.py

基线结果：

- 155 passed
- 0 failed
- 用时约 2.11 秒

以下三个静态规范测试在交接机器上缺少 ROS ament 插件，收集阶段会报 ModuleNotFoundError：

- ament_copyright
- ament_flake8
- ament_pep257

这属于环境缺依赖，不代表源码测试失败。接收方应安装对应 ROS 测试依赖后单独运行这三个测试，并如实记录结果。

GCode CLI 直接运行还需要 rclpy。交接机器的系统 Python 没有 rclpy，因此 GCode 真实 CLI 基线未在本次交接中运行；相关功能由现有单元测试覆盖。接收方必须在正确的 ROS 2 Python 环境中补跑。

## 7. 已建立的真实外部 NPZ 导出基线

输入：

    data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz

命令：

    PYTHONPATH=src/my_project/path_processing_core:src/my_project/external_npz_preprocessor \
    python3 -m external_npz_preprocessor.cli \
      --source data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz \
      --out /tmp/external-template-system.npz

交接基线：

- 输出行数：43231
- seq：0 到 43230
- total_layers：2
- planned_time_s 末值约 172.77200317382812
- 产物包括 NPZ、offset.json 和 timing.json

冻结产物位于 handoff/golden/。不要覆盖这些文件；重跑结果应写入新的临时目录并与冻结产物比较。

冻结 NPZ 字段：

- seq
- x, y, z
- a, b, c
- e
- tool_id
- move_type
- src_line
- event_flag
- event_type
- payload
- trigger_seq
- layer_index
- total_layers
- preview_layer_index
- path_id
- path_end_flag
- planned_time_s
- move_type_vocab_keys / move_type_vocab_vals
- event_type_vocab_keys / event_type_vocab_vals

## 8. 第一阶段：只读完整审计

必须逐个阅读仓库中的全部受版本控制文件，不得只阅读入口文件或 README。

完成以下任务：

1. 建立完整源码目录清单，说明每个模块、脚本、测试、配置、样例和文档的职责。
2. 搜索所有 import、文件路径、环境变量、ROS 依赖和运行时生成文件。
3. 确认是否存在指向原仓库中未携带文件的依赖。
4. 梳理 GCode 链路，从解析、primeline、命令类型、规划、插值到统一 NPZ 导出。
5. 梳理外部 NPZ 链路，从源格式加载、参数模型、轨迹转换到统一 NPZ 导出。
6. 说明两条链路如何共享 path_processing_core。
7. 梳理 head calibration、offset、timing、manifest、分层和预览数据的关系。
8. 对照 npz_loader、queue_manager 和消息定义，逐字段确认消费兼容性。
9. 识别代码中存在但主流程未调用的实现，不得仅凭函数存在就声称功能已启用。
10. 运行第 5、6、7 节的所有验证，并记录机器环境、Python、ROS、NumPy 和 pytest 版本。

第一阶段不得提交源代码修改。只允许提交审计文档。

## 9. 第一阶段必须交付的文档

在 docs/audit/ 下创建：

1. SOURCE_INVENTORY.md
   - 每个文件的职责和所有权分类。
2. DEPENDENCY_AUDIT.md
   - Python、ROS、Qt、NumPy、Matplotlib、文件系统及跨仓库依赖。
3. GCODE_PIPELINE.md
   - GCode 到系统 NPZ 的逐阶段数据流。
4. EXTERNAL_NPZ_PIPELINE.md
   - 外部源 NPZ 到系统 NPZ 的逐阶段数据流。
5. SHARED_CORE.md
   - path_processing_core 的公共类型、算法和 exporter。
6. NPZ_OUTPUT_CONTRACT.md
   - 字段、dtype、shape、单位、词表、sidecar、manifest、兼容规则。
7. TEST_BASELINE.md
   - 完整命令、环境、通过/失败/跳过数量和失败原文。
8. MISSING_OR_AMBIGUOUS.md
   - 缺失依赖、行为歧义、死代码、未覆盖分支和待所有者确认的问题。
9. SEPARATION_PROPOSAL.md
   - 只提出拆分后的目标结构和迁移步骤，不实施重构。

所有事实必须引用具体文件和行号。推断必须明确标记为推断。

## 10. 重点审计事项

外部 NPZ converter 中存在样条相关实现。必须确认当前主流程实际返回样条还是 POLYLINE，不能根据函数名推断。

检查以下兼容风险：

- trigger_seq 与事件实际执行时机。
- layer_index、preview_layer_index 和物理层高的区别。
- path_id 与 path_end_flag 的连续性。
- planned_time_s 单调性和 timing sidecar 总时间。
- manifest 分片顺序和 loader 的文件解析规则。
- float 精度，尤其 e 和 planned_time_s。
- 空路径、零长度打印路径、首层参数和多材料路径。
- prime、retract、reset-E、wait、cut 和 tool-change 事件。
- gcode_planner 兼容包装层与 path_processing_core 实现是否漂移。

## 11. 报告发现缺失依赖的格式

不要自行补文件。创建问题记录：

    标题：
    发现位置：
    直接证据：
    缺失对象：
    对测试或运行的影响：
    建议由所有者提供的最小文件：
    是否阻塞第一阶段：

只有仓库所有者确认后，才能追加交接文件并生成新的交接版本。

## 12. 提交和分支规则

第一阶段使用分支：

    audit/offline-planner-baseline

允许的提交仅限 docs/audit/ 文档。推荐一个主题一个提交。

禁止：

- 修改 main。
- 强制推送。
- 重写或移动 handoff 标签。
- 把生成的临时 NPZ、缓存、日志、build、install、log 或虚拟环境提交进仓库。
- 在审计 PR 中夹带格式化或源码清理。

第一阶段结束时创建一个 PR，标题：

    audit: document offline planner baseline and separation boundary

## 13. 第一阶段完成标准

只有同时满足以下条件才算完成：

1. 124/124 源文件 SHA256 验证通过。
2. 黄金文件 SHA256 验证通过。
3. 155 项功能/契约测试至少保持通过。
4. 三个 ament 静态测试已运行，或明确给出仍无法安装的环境证据。
5. GCode CLI 在 ROS 2 环境中至少完成一次代表性真实输入导出。
6. 外部 NPZ 模板导出结果已复现。
7. 九份审计文档全部完成。
8. 所有缺失和歧义均已列出，没有静默猜测。
9. 没有修改任何现有源代码和黄金产物。
10. 仓库所有者审阅并明确批准进入第二阶段。

## 14. 第二阶段仅在获得批准后执行

批准前只提交方案。批准后才可以：

1. 建立独立安装和运行方式。
2. 减少不必要的 ROS 运行时依赖。
3. 将上位机 UI 调用替换为文件导入、任务接口或外部进程接口。
4. 为 NPZ 契约建立版本号和双仓库契约测试。
5. 清理兼容包装层或调整目录。
6. 最终从实时上位机仓库删除离线生产逻辑。

任何 schema 变化都必须先实现向后兼容、版本迁移和上位机消费者测试。

## 15. 给仓库所有者的首次回复模板

完成接收验真后，先回复：

    已检出 offline-planner-handoff-v1。
    原始来源提交：eb9091bcf405eaca2dddb07f2998bb3f25c12601。
    SOURCE_SHA256SUMS：通过 N/N。
    GOLDEN_SHA256SUMS：通过 N/N。
    功能测试：通过 N，失败 N，跳过 N。
    静态测试：通过 N；缺依赖或失败项为……
    当前未修改任何源代码。
    下一步将执行只读完整审计并提交 docs/audit/ 文档。

如果验真失败，停止并只报告失败项。
