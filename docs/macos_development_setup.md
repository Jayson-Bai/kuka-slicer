# KUKA Slicer 与 Core：Windows/macOS 双端开发环境配置指南

> 审计基准：`kuka_slicer` 主仓库提交 `213a63c4afe116244ce37260c750da0c44c6b4ad`（2026-09-14）。
> 适用范围：当前仓库根包 `kuka-slicer`、仓库内 `packages/offline_path_planner`（本文简称 Core）。
> 不包含：上位机 `kuka_ram_ws`、ROS 2 实时运行环境、机器人控制器与硬件联调。

## Mac 接收方：从这里开始

这是一份可直接交给 Mac 开发人员执行的交付文档。Windows 侧已经完成且不会影响现有环境的准备项包括：Prusa 定制补丁入库、macOS 原生产物忽略规则和本指南。第 3 节保留为 Windows 侧审计记录，Mac 接收方不需要执行其中的 PowerShell 命令。

开始前只需确认：

- 这台 Mac 能访问 GitHub 仓库 `Jayson-Bai/kuka-slicer`；私有仓库需由仓库管理员提前授权账号。
- Mac 有管理员权限安装 Xcode Command Line Tools、Homebrew 和编译依赖。
- 至少预留 30 GB 磁盘空间，并允许较长时间的 Prusa 依赖编译。
- 不从 Windows 复制 `.venv`、`deps`、`.pyd`、构建缓存或整个工作目录。

拿到仓库后，先执行以下交付完整性检查：

```bash
git switch main
git pull --ff-only origin main
git status --short --branch
test -f native/prusa_bridge/patches/prusa-2.9.6-kuka-raft-controls.patch
shasum -a 256 native/prusa_bridge/patches/prusa-2.9.6-kuka-raft-controls.patch
```

补丁 SHA256 必须是：

```text
AB1DE888580A2534D2C4B41C29585F9D456EC9547A7BD8F8C04421EBAC45627D
```

若文件缺失或哈希不符，停止 Prusa 编译并联系 Windows 侧确认，不要从聊天记录、网盘或 Windows `deps` 目录另取一份源码覆盖。校验通过后按第 4 → 5 → 6 → 7 → 8 节顺序执行；第 8.4 节的双端结果对照完成前，只能称为“Mac 开发环境可运行”，不能称为“离线端生产验收完成”。

## Windows 准备状态（2026-09-15）

本机必要准备已在不重建、不替换现有生产环境的前提下完成：

- 已将现有 5 个 PrusaSlicer 定制文件导出为 `native/prusa_bridge/patches/prusa-2.9.6-kuka-raft-controls.patch`。
- 补丁 SHA256 为 `AB1DE888580A2534D2C4B41C29585F9D456EC9547A7BD8F8C04421EBAC45627D`。
- 补丁已同时通过：对干净 Prusa Git index 的正向 `git apply --check --cached`，以及对当前定制工作树的反向 `git apply --reverse --check`。
- `.gitignore` 已补充 macOS native bridge 的 `*.so`、`*.dylib`、`*.a`、`*.dSYM/`。
- 准备前备份位于 `.runtime/backups/macos-prep-20260915-110546`，包含 19 个文件及校验记录。
- 准备过程中没有删除、移动、restore 或 stash；Windows `.venv`、Prusa 源码、依赖构建目录和现有 `.pyd` 均未改动。
- 交付提交只包含 `.gitignore`、本文档和上述 Prusa 补丁；Windows 本地备份不会上传，其他业务代码改动也不属于本次交付。

Mac 接收方能够从 GitHub 读取本文时，表示这些必要准备文件已进入云端；无需访问 Windows 本地备份。

## 1. 先说结论

当前项目可以在 Mac 上建立独立源码工作区，运行 Python UI、Legacy/PySLM 路径和纯 Python Core；Mac 原生 Prusa 内核从源码编译在结构上也是可行的。但是，若目标是与现有 Windows 生产结果一致，不能只执行 `git clone` 和 `pip install`。需要同时处理以下三项可复现性问题：

1. Windows 使用的 PrusaSlicer 2.9.6 源码基于提交 `b028299c770b8380ee81c921a2867d522f288123`，另有 5 个定制文件、40 行新增和 2 行修改。该差异现已固化为仓库内补丁，Mac 必须在同一官方提交上应用它。
2. 仓库暂时没有 macOS 一键构建脚本，Mac 必须按第 5.4/5.5 节本地编译；生成的 `*.so`/`*.dylib`/`*.a` 已被忽略，不能提交。
3. UI 会自动把 Core/Prusa 设置写回两个受 Git 跟踪的 JSON。Mac 首次配置必须按第 6 节隔离本机设置，避免双端配置漂移或覆盖。

因此，推荐的目标状态是：

```text
GitHub origin/main
      |
      +-- Windows 独立 clone + Windows .venv + Windows deps + .pyd
      |
      +-- macOS 独立 clone + macOS .venv + macOS deps + .so

仅同步源码、受控默认配置和 Prusa 补丁；
绝不同步 .venv、deps 构建目录、原生二进制、outputs 或用户本地设置。
```

文档中的“生产测试”表示离线端生成物达到可比较、可回归的状态，不表示已完成上位机、ROS 2、实时 RSI 或真实机器人硬件验收。

## 2. 本次配置审计结果

| 项目 | 当前事实 | 对 Mac 的影响 |
| --- | --- | --- |
| 主包 | 根目录 `pyproject.toml`，Python `>=3.10`，依赖 NumPy、SciPy、Shapely | 纯 Python 部分可跨平台 |
| Core | `packages/offline_path_planner/pyproject.toml`，Python `>=3.10`，基础依赖仅 NumPy | 外部 NPZ → 系统 NPZ 可在 Mac 纯 Python 运行 |
| 两包边界 | 生产代码不互相静态导入；通过 `external_layer_paths_v1` NPZ 集成 | 两包应分别安装，但必须来自同一仓库提交 |
| 主 UI | `python -m kuka_slicer ui`，默认 `127.0.0.1:8765` | Web UI 本身可运行 |
| 辅助 UI | `surface-preview` 为 8766，`surface-map` 为 8767 | 可分别从终端运行 |
| Prusa 内核 | pybind11 + CMake + PrusaSlicer `libslic3r` | 每个平台必须本地编译，Windows `.pyd` 不能复制到 Mac |
| Windows 原生桥 | 当前报告 `PrusaSlicer-2.9.6`，Windows Python 3.13.1 x64 | Mac 应使用相同 Python 小版本和相同 Prusa 补丁进行对照 |
| Windows CMake | 全局 `PATH` 当前没有 `cmake`；PowerShell 构建脚本会定位 Visual Studio 2022 自带 CMake/Ninja | 不应根据全局 `cmake` 缺失误判 Windows 原生桥不可重建 |
| 浏览器独立窗口 | 自动搜索路径只写了 Windows Chrome/Edge；但 `KUKA_SLICER_BROWSER` 可指定任意实际浏览器可执行文件 | Mac 设置 Chrome 路径后通常可用；也可直接浏览器打开端口 |
| 原生文件选择器 | 映射 NPZ/Core NPZ 两个服务器端 picker 明确限制 `sys.platform == "win32"` | Mac 上主 STL 上传、共形 JSON 上传、曲面 NPZ 的浏览器上传可用；“导入 Core NPZ 预览”与本地碰撞检查的原生选择流程不可用 |
| 持久配置 | UI 自动修改 `print_params.json` 与 `prusa_params.json`，两者均被 Git 跟踪 | 需要本机隔离或后续代码改造 |
| 输出/缓存 | `.venv/`、`deps/`、`outputs/`、`.runtime/`、`build/`、`dist/` 已忽略 | 两台机器可安全各自生成 |
| 打包 | 现有 wheel 是旧产物，只含 14 个文件，不含当前全部模块、`assets/`、Core 或原生桥 | Mac 开发/测试必须使用源码 editable 安装，不能把现有 wheel 当生产包 |
| 依赖锁定 | 没有 lock/constraints 文件 | 两端依赖可能随时间漂移；首次对齐后应补锁定文件 |
| CI | 当前 `.github` 没有 workflow | 暂无云端 Windows/macOS 自动回归保护 |
| Git 换行 | Windows `core.autocrlf=true`；项目仅对 golden 文件设 `-text` | Mac 建议 `core.autocrlf=input`，不要在首次配置时全仓库 renormalize |

本次 Windows 只读/测试基线：

- Windows：Python 3.13.1 x64、NumPy 2.5.1、SciPy 1.18.1、Shapely 2.1.2、pytest 9.1.1、pybind11 3.0.4。
- 根包测试在限定 `tests/` 并提供 Core 源路径后：`490 passed, 2 skipped`。
- Core 功能测试：`177 passed, 1 failed`；唯一失败是绘图测试要求生成 PNG，但当前环境和包元数据均未包含 Matplotlib。
- README 中的裸 `python -m pytest` 不适合当前 Windows 工作区，因为 `deps` 是指向 `D:\kuka_slicer_prusa_deps` 的 junction，pytest 会误收集 Boost 自带测试。应始终使用本文的定向测试命令。

## 3. Windows 侧一次性准备（已完成，Mac 只需阅读）

本节用于说明云端交付物如何从 Windows 生产基线获得，便于审计和以后重建。本次交付已经完成这些操作；Mac 接收方不要执行本节 PowerShell 命令，也不需要连接 Windows 文件系统。

### 3.1 固化主仓库的云端基线

在 Windows PowerShell 中执行：

```powershell
Set-Location F:\CodeX_ws\kuka_slicer
git status --short --branch
git fetch origin
git log -1 --oneline
git push origin main
```

要求：开始 Mac 配置前，Windows 的业务源码改动应已提交并推送，或者明确保留为 Windows 本地未完成分支。不要把 `.venv`、`deps`、`kuka_slicer/_native` 中的二进制、`outputs`、日志或真实生产数据强行加入 Git。

当前仓库还有名为 `offline-planner` 的 remote，指向旧的独立 Core 仓库。Mac 同步本工作区时只需要 `origin=https://github.com/Jayson-Bai/kuka-slicer.git`；不要把第二个 remote 误认为必须再克隆一次的 Core。

### 3.2 将 Windows 的 Prusa 定制补丁加入云端

这是生产一致性的硬前置。建议把补丁存在主仓库内，而不是提交整个 PrusaSlicer 源码或把 `deps/` 解除忽略。

先检查基线和改动范围：

```powershell
Set-Location F:\CodeX_ws\kuka_slicer
git -C deps\PrusaSlicer-version_2.9.6 rev-parse HEAD
git -C deps\PrusaSlicer-version_2.9.6 status --short
git -C deps\PrusaSlicer-version_2.9.6 diff --check
```

第一条必须是：

```text
b028299c770b8380ee81c921a2867d522f288123
```

当前预期仅有以下 5 个修改文件：

```text
src/libslic3r/Preset.cpp
src/libslic3r/PrintConfig.cpp
src/libslic3r/PrintConfig.hpp
src/libslic3r/Slicing.cpp
src/libslic3r/Support/SupportParameters.cpp
```

导出补丁到不会被 `.gitignore` 排除的位置：

```powershell
New-Item -ItemType Directory -Force native\prusa_bridge\patches | Out-Null
git -C deps\PrusaSlicer-version_2.9.6 diff --binary -- `
  src/libslic3r/Preset.cpp `
  src/libslic3r/PrintConfig.cpp `
  src/libslic3r/PrintConfig.hpp `
  src/libslic3r/Slicing.cpp `
  src/libslic3r/Support/SupportParameters.cpp `
  > native\prusa_bridge\patches\prusa-2.9.6-kuka-raft-controls.patch

git -C deps\PrusaSlicer-version_2.9.6 apply --reverse --check `
  ..\..\native\prusa_bridge\patches\prusa-2.9.6-kuka-raft-controls.patch
Get-FileHash native\prusa_bridge\patches\prusa-2.9.6-kuka-raft-controls.patch -Algorithm SHA256
```

当前 Windows 树已经包含这些改动，所以这里使用 `--reverse --check` 验证导出的补丁确实能还原当前差异。最可靠的正向验收仍是在一个全新 Prusa clone 上执行第 5.2 节的 `git apply --check`。

补丁和本文档应作为独立交付提交推送。不要修改或清理当前 Windows 的 dirty Prusa 工作树。

### 3.3 补充 Mac 原生产物忽略规则

在主仓库 `.gitignore` 的 native bridge 区域补充：

```gitignore
kuka_slicer/_native/*.so
kuka_slicer/_native/*.dylib
kuka_slicer/_native/*.a
kuka_slicer/_native/*.dSYM/
```

这一步只改变 Git 的忽略规则，不删除或替换 Windows 的 `.pyd`。补充后再提交并推送。

### 3.4 记录 Windows 对照环境

不要把 Windows `.venv` 复制到 Mac。只记录版本：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip list --format=freeze |
  Select-String '^(numpy|scipy|shapely|pytest|pybind11|matplotlib|kuka-)=='
.\.venv\Scripts\python.exe -c "from kuka_slicer.prusa_bridge import bridge_info; print(bridge_info())"
git rev-parse HEAD
```

若要达到长期可重建状态，应后续新增一个跨平台 constraints/lock 文件，并在 Windows 与 Mac 同一提交上重新验证。不要直接把包含 Windows 专属包的完整 `pip freeze` 当跨平台锁文件。

### 3.5 Windows 保持不受影响的规则

- Mac 使用全新 clone，不使用 SMB、iCloud Drive、OneDrive 或 Dropbox 共享同一个工作目录。
- 两端分别拥有自己的 `.venv`、`deps`、CMake build 目录与原生扩展。
- 不复制 Windows 的 junction `deps -> D:\kuka_slicer_prusa_deps`；Mac 创建普通本地目录 `deps/`。
- 不复制 Windows `.pyd` 到 Mac；也不把 Mac `.so` 复制回 Windows。
- 功能开发使用独立分支并通过 GitHub 合并；不要两台机器同时在同一未推送分支上修改。
- 合并前始终检查两个受跟踪参数 JSON 是否被 UI 自动改动。

## 4. Mac 硬件与系统前置

先执行：

```bash
uname -m
sw_vers
```

`uname -m` 通常是 Apple Silicon 的 `arm64` 或 Intel Mac 的 `x86_64`。Python、Prusa 依赖和 bridge 必须全部使用相同架构。不要在一个原生 arm64 虚拟环境中混用 Rosetta/x86_64 Homebrew，反之亦然。

推荐条件：

- 受 Homebrew 支持的 macOS；当前 Homebrew 官方支持基线是 macOS 15+，Intel Mac 属于较低支持等级。
- 至少 30 GB 可用空间。Prusa 依赖源码和 Release 构建可能占用数 GB，编译也可能持续较长时间。
- Xcode Command Line Tools；Prusa 官方 macOS 指南建议完整 Xcode，若只装 CLT 构建失败再安装完整 Xcode。
- Chrome。Safari 可打开普通 UI，但项目的一键独立窗口逻辑使用 Chromium `--app` 参数。

官方参考：[Homebrew 安装与平台要求](https://docs.brew.sh/Installation)、[Python `venv`](https://docs.python.org/3/library/venv.html)、[PrusaSlicer 2.9.6 macOS 构建指南](https://github.com/prusa3d/PrusaSlicer/blob/version_2.9.6/doc/How%20to%20build%20-%20Mac%20OS.md)。

## 5. Mac 从零安装

### 5.1 安装系统工具与 GitHub 凭据

```bash
xcode-select --install
```

安装 Homebrew 时使用 [brew.sh](https://brew.sh/) 当时显示的官方命令。安装完成后，严格执行安装器打印的 `brew shellenv` 指令，然后安装依赖：

```bash
brew update
brew install git git-lfs cmake ninja automake gettext libtool texinfo m4 zlib python@3.13
brew install --cask google-chrome
```

确认工具与架构：

```bash
git --version
cmake --version
"$(brew --prefix python@3.13)/bin/python3.13" --version
"$(brew --prefix python@3.13)/bin/python3.13" -c 'import platform; print(platform.machine())'
```

若主仓库为私有库，推荐为这台 Mac 单独创建 SSH key，不复制 Windows 私钥：

```bash
ssh-keygen -t ed25519 -C "你的 GitHub 邮箱"
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
pbcopy < ~/.ssh/id_ed25519.pub
```

将公钥加入 GitHub 后执行：

```bash
ssh -T git@github.com
```

参考：[GitHub 添加 SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) 与 [克隆仓库](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)。

### 5.2 克隆主仓库和准确的 Prusa 源码

```bash
mkdir -p ~/Developer
cd ~/Developer
git clone git@github.com:Jayson-Bai/kuka-slicer.git
cd kuka-slicer
git switch main
git pull --ff-only origin main
git status --short --branch
```

确认输出中没有意外修改，然后创建 Mac 专用开发分支：

```bash
git switch -c codex/macos-bootstrap
```

克隆准确版本的 PrusaSlicer：

```bash
git clone --branch version_2.9.6 --depth 1 \
  https://github.com/prusa3d/PrusaSlicer.git \
  deps/PrusaSlicer-version_2.9.6

git -C deps/PrusaSlicer-version_2.9.6 rev-parse HEAD
```

提交必须是 `b028299c770b8380ee81c921a2867d522f288123`。若不是，不要继续生产对齐，应检查官方 tag 是否被错误选择。

应用 Windows 已版本化的 KUKA 补丁：

```bash
git -C deps/PrusaSlicer-version_2.9.6 apply --check \
  ../../native/prusa_bridge/patches/prusa-2.9.6-kuka-raft-controls.patch
git -C deps/PrusaSlicer-version_2.9.6 apply \
  ../../native/prusa_bridge/patches/prusa-2.9.6-kuka-raft-controls.patch
git -C deps/PrusaSlicer-version_2.9.6 diff --check
git -C deps/PrusaSlicer-version_2.9.6 diff --stat
```

预期仍是 5 个文件、40 行新增、2 行修改。不要自行“顺手修复”该第三方源码中的其他内容，否则 Windows/Mac 不再是同一内核。

### 5.3 创建 Mac 独立 Python 环境

在仓库根目录执行：

```bash
"$(brew --prefix python@3.13)/bin/python3.13" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[test]'
python -m pip install -e './packages/offline_path_planner[test]'
python -m pip install pybind11 matplotlib build
```

为什么使用 editable 安装：当前 UI 会从源码相对路径读取 `assets/` 和 Core 的 `data/`，现有 wheel 又不包含完整的新模块、资产、Core 和 native bridge。此阶段“从源码 clone + editable 安装”才是受支持的开发方式。

确认双包与关键依赖：

```bash
python -c 'import kuka_slicer, external_npz_preprocessor, path_processing_core; print("imports ok")'
python -c 'import numpy, scipy, shapely, matplotlib, pybind11; print(numpy.__version__, scipy.__version__, shapely.__version__, matplotlib.__version__, pybind11.__version__)'
python -m pip check
```

默认生产验收先使用 Prusa 内核。PySLM 是 `PythonSLM==0.6.1` 的可选 extra，不是 Core 的必要依赖；只有确实要在 Mac 验证 PySLM 时才额外执行：

```bash
python -m pip install -e '.[pyslm,test]'
python -c 'import pyslm; print("PySLM import ok")'
```

如果 PythonSLM 在当前 Python 3.13/macOS 架构上无法安装，不要因此更换整个项目 Python 或跳过 Prusa 基线。应为 PySLM 建立另一套独立虚拟环境进行兼容性验证，并把它标记为“可选内核未验收”，而不是误判主 Prusa/Core 环境失败。

### 5.4 编译 Prusa 依赖

保持虚拟环境激活，从仓库根目录执行。先使用较保守的并行度 2，避免编译占满内存：

```bash
cmake -S deps/PrusaSlicer-version_2.9.6/deps \
  -B deps/prusa-deps-build-macos \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
  -DDESTDIR="$PWD/deps/prusa-deps-macos"

cmake --build deps/prusa-deps-build-macos --parallel 2
```

构建完成后必须存在：

```bash
test -d "$PWD/deps/prusa-deps-macos/usr/local"
```

Prusa 官方特别说明：依赖 `destdir` 内含绝对安装路径，构建后不要移动 `deps/prusa-deps-macos`。若需要换目录，应删除 Mac 自己的 build/dependency 目录并重新构建，不要动 Windows 的 `D:` 盘依赖。

若出现 `CMath::CMath target not found`，这是 Prusa 2.9.6 官方文档记录的 CMake 版本兼容问题。先记录完整错误和 `cmake --version`，再按官方指南使用 CMake 3.27.9；不要同时更换 Prusa tag、编译器和依赖版本，否则无法定位变量。

若提示无法确定 deployment target，只在依赖和 bridge 两次 CMake 配置中使用相同的值，例如当前系统为 macOS 15 时都加：

```text
-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0
```

不要照抄与本机不一致的系统版本。

### 5.5 编译 Mac 原生 bridge

```bash
PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir)"

cmake -S native/prusa_bridge \
  -B deps/prusa-bridge-build-macos \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
  -DPRUSA_SOURCE_DIR="$PWD/deps/PrusaSlicer-version_2.9.6" \
  -DKUKA_NATIVE_OUTPUT_DIR="$PWD/kuka_slicer/_native" \
  -DCMAKE_PREFIX_PATH="$PWD/deps/prusa-deps-macos/usr/local" \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"

cmake --build deps/prusa-bridge-build-macos --target prusa_bridge --parallel 2
```

成功时，`kuka_slicer/_native/` 下会出现类似 `prusa_bridge.cpython-313-darwin.so` 的文件。验证：

```bash
find kuka_slicer/_native -maxdepth 1 -name 'prusa_bridge*.so' -print
python -c 'from kuka_slicer.prusa_bridge import bridge_info; print(bridge_info())'
```

期望：

```text
{'available': True, 'reason': '', 'native_version': 'PrusaSlicer-2.9.6'}
```

再确认原生产物没有进入待提交列表：

```bash
git status --short
git check-ignore -v kuka_slicer/_native/prusa_bridge*.so
```

## 6. 隔离每台机器的本地配置

### 6.1 当前会自动写回仓库的文件

主 UI 在控件变化后约 120 ms 自动保存：

```text
packages/offline_path_planner/data/external_npz_preprocessor/print_params.json
packages/offline_path_planner/data/external_npz_preprocessor/prusa_params.json
```

它们现在既是默认配置，又承担用户本地设置，职责混合。这是跨机器同步的主要冲突源。

### 6.2 不改代码时的临时隔离方案

在 Mac 上备份后标记 `skip-worktree`：

```bash
mkdir -p "$HOME/Library/Application Support/KukaSlicer/config-backup"
cp packages/offline_path_planner/data/external_npz_preprocessor/print_params.json \
  "$HOME/Library/Application Support/KukaSlicer/config-backup/print_params.json"
cp packages/offline_path_planner/data/external_npz_preprocessor/prusa_params.json \
  "$HOME/Library/Application Support/KukaSlicer/config-backup/prusa_params.json"

git update-index --skip-worktree \
  packages/offline_path_planner/data/external_npz_preprocessor/print_params.json \
  packages/offline_path_planner/data/external_npz_preprocessor/prusa_params.json
```

检查标记，行首应为 `S`：

```bash
git ls-files -v \
  packages/offline_path_planner/data/external_npz_preprocessor/print_params.json \
  packages/offline_path_planner/data/external_npz_preprocessor/prusa_params.json
```

风险：`skip-worktree` 会隐藏本地修改，也会让上游对这两个默认文件的更新不易察觉。因此每次发布或确知默认配置已变化时，先备份本地文件，再取消标记、同步和人工合并：

```bash
git update-index --no-skip-worktree \
  packages/offline_path_planner/data/external_npz_preprocessor/print_params.json \
  packages/offline_path_planner/data/external_npz_preprocessor/prusa_params.json
git status --short
```

不要在有未备份的本地参数时直接覆盖或 restore 这两个文件。

### 6.3 正确的长期方案

后续应单独修改代码，将“仓库默认值”与“每用户运行设置”分离：

- 仓库 JSON 只读，作为默认模板。
- Windows 用户设置写到 `%LOCALAPPDATA%\KukaSlicer\...`。
- macOS 用户设置写到 `~/Library/Application Support/KukaSlicer/...`。
- 启动时先读默认模板，再覆盖用户设置。
- 增加配置 schema/version 与迁移测试。

`surface_preview` 已采用“Windows 用 `LOCALAPPDATA`、其他平台回退用户主目录”的思路，但 Mac 最终仍应使用标准 Application Support 目录。完成这项代码改造后，应取消 `skip-worktree` 临时措施。

## 7. 启动 UI 和 Core

### 7.0 运行时环境变量

当前代码读取的项目环境变量如下：

| 变量 | 用途 | Mac 行为 |
| --- | --- | --- |
| `KUKA_SLICER_BROWSER` | 指定 Chrome/Chromium 可执行文件 | 推荐设为 `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` |
| `KUKA_SLICER_MAX_CPU_CORES` | 把数值库线程数限制到指定值，但代码仍不会允许超过逻辑 CPU 的 70% | 有效；应用导入时也会为未显式设置的 OMP/OpenBLAS/MKL/NumExpr/VecLib 线程变量写默认值 |
| `KUKA_SLICER_MAX_MEMORY_PERCENT` | 请求不高于 70% 的 working-set 上限 | 当前只通过 Windows API 实施，Mac 上不会形成硬内存上限 |
| `KUKA_SLICER_LOW_PRIORITY` | 以 `1/true/yes/on` 请求后台优先级 | 当前只在 Windows 生效 |
| `KUKA_SLICER_HONEYCOMB_REFERENCE_ROOT` | 指向可选的真实蜂窝回归数据集 | 数据不在可移植仓库内；不设置时相关测试应 skip，不影响普通 UI/Core |

macOS 没有当前代码使用的 `sched_getaffinity`/`sched_setaffinity` 路径，因此 UI 元数据中的 `affinity_applied` 可能为 `false`。这不等于线程限制完全失效，但也不能把它解释为 Windows 式进程 CPU affinity 已应用。首次大模型切片建议显式从较低值开始：

```bash
export KUKA_SLICER_MAX_CPU_CORES=4
```

不要在项目中提交个人 `.env`。若写入 `~/.zshrc`，该配置只影响 Mac 用户环境，不会同步到 Windows。

### 7.1 主 UI

```bash
cd ~/Developer/kuka-slicer
source .venv/bin/activate
python -m kuka_slicer ui --host 127.0.0.1 --port 8765 --output-dir outputs/mac-ui
```

浏览器打开 `http://127.0.0.1:8765`。只绑定 `127.0.0.1`，不要为了方便改成 `0.0.0.0`；当前服务器不是面向局域网或公网加固的服务。

主流程验收：上传一个已在 Windows 使用过的小型 STL，选择 Prusa 内核，完成切片，确认页面能够展示最终 Core 轨迹并下载 Core NPZ。

### 7.2 蜂窝网格共形设计器与 surface-map

最稳定的 Mac 启动方式是分别开终端：

```bash
source .venv/bin/activate
python -m kuka_slicer surface-preview --host 127.0.0.1 --port 8766
```

```bash
source .venv/bin/activate
python -m kuka_slicer surface-map --host 127.0.0.1 --port 8767
```

对应打开 `http://127.0.0.1:8766` 和 `http://127.0.0.1:8767`。

若希望主 UI 的“启动蜂窝网格共形设计器”按钮打开 Chrome 独立窗口：

```bash
export KUKA_SLICER_BROWSER='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
python -m kuka_slicer ui --host 127.0.0.1 --port 8765 --output-dir outputs/mac-ui
```

可将该环境变量加入个人 `~/.zshrc`，不要提交含机器路径的项目 `.env`。

### 7.3 Mac 当前不可用的两个 UI 动作

以下后端函数明确拒绝非 Windows 平台：

- “导入 Core NPZ 预览”的服务器端原生文件选择器。
- “导入曲面/共形 NPZ 预览”后依赖服务器本地路径的碰撞检查。

普通 STL 上传、共形 JSON 上传和曲面 NPZ 的浏览器上传是 HTML file input，不受该限制；但 Core NPZ 没有等价的浏览器上传 endpoint。若 Mac 必须完整使用这两项能力，需要另开代码任务：增加浏览器上传接口，或把 picker 实现改为 macOS 可用且保持安全边界。本文不把这项未实现功能算作“已通过”。

### 7.4 单独运行 Core

```bash
mkdir -p outputs/mac-core-smoke
kuka-offline-npz \
  --source packages/offline_path_planner/data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz \
  --out outputs/mac-core-smoke/two-layer-system.npz \
  --dt 0.02
```

验证系统 NPZ 契约：

```bash
python -c 'from path_processing_core.npz_contract import validate_system_npz_contract; import sys; print(validate_system_npz_contract(sys.argv[1]))' \
  outputs/mac-core-smoke/two-layer-system.npz
```

预期输出 `True` 或 `1`。Core 外部 NPZ CLI 不需要 ROS 2。`kuka-offline-gcode` 的纯 CLI 也能在没有 ROS 时导入；只有遗留 ROS node 入口需要 `rclpy`。

## 8. Mac 验收测试

### 8.1 环境与导入

```bash
python --version
python -m pip check
python -c 'from kuka_slicer.prusa_bridge import bridge_info; assert bridge_info()["available"], bridge_info()'
python -c 'import kuka_slicer, external_npz_preprocessor, path_processing_core'
```

### 8.2 根包测试

安装两个 editable 包后，使用定向路径，避免扫描 `deps/`：

```bash
python -m pytest -q tests
```

Windows 审计基线为 `490 passed, 2 skipped`。Mac 的 exact 数量会随代码提交变化；判断标准是无失败，并记录 skip 原因。原生 bridge 测试不能因 bridge 缺失而整批跳过，否则不算 Prusa 生产内核验收。

### 8.3 Core 功能测试

```bash
cd packages/offline_path_planner
PYTHONPATH="src/my_project/path_processing_core:src/my_project/gcode_planner:src/my_project/external_npz_preprocessor" \
python -m pytest -q \
  src/my_project/path_processing_core/test \
  src/my_project/gcode_planner/test \
  src/my_project/external_npz_preprocessor/test \
  test/scripts/test_plot_npz_xy.py \
  --ignore=src/my_project/gcode_planner/test/test_copyright.py \
  --ignore=src/my_project/gcode_planner/test/test_flake8.py \
  --ignore=src/my_project/gcode_planner/test/test_pep257.py
cd ../..
```

安装 Matplotlib 后，Windows 当前那一个 PNG 绘图失败应消失。三个被显式忽略的 ament/style 测试属于遗留 ROS/代码风格基线，不应伪装成通过；它们与本次 Mac 离线生产环境无关。

### 8.4 Windows/Mac 结果对照

选择一个小型、固定、可合法同步的 STL 和固定 UI 参数，在同一主仓库提交、同一 Prusa 基线提交、同一补丁下分别生成：

```text
Windows: *_source.npz、*_prusa.gcode、*_core.npz、sidecars
macOS:   *_source.npz、*_prusa.gcode、*_core.npz、sidecars
```

对照原则：

- 先比较 metadata、字段名、dtype、shape、层数、路径数、行数和事件序列。
- 浮点数组使用明确容差的 `numpy.testing.assert_allclose`；不要只比较 ZIP/NPZ 文件 SHA256，因为压缩容器元数据和浮点架构差异可能造成字节不同。
- 离散字段、序号、事件类型与字符串应完全相等。
- 若路径拓扑、行数或事件顺序不同，不要用放宽浮点容差掩盖。
- 记录 Python/NumPy/SciPy/Shapely、编译器、macOS、CPU 架构、Prusa commit/patch hash 和主仓库 commit。

只有以下全部通过，才可把 Mac 标记为离线端生产开发环境：

- `bridge_info().available == True`。
- 根包与 Core 定向测试无失败。
- Core smoke NPZ 通过契约验证。
- 已知 STL 能在主 UI 完成 Prusa → Core → 下载闭环。
- Windows/Mac 对照的离散结构一致，浮点差异在事先约定容差内。
- `git status --short` 中没有 `.venv`、`deps`、`.so`、输出、真实数据或本机配置泄漏。

## 9. 双端日常同步流程

每项功能单独使用分支。开始工作：

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/<功能名>
```

提交前：

```bash
git status --short
git diff --check
python -m pytest -q tests
git diff -- \
  packages/offline_path_planner/data/external_npz_preprocessor/print_params.json \
  packages/offline_path_planner/data/external_npz_preprocessor/prusa_params.json
```

确认没有环境文件和意外参数变更，再提交并推送该分支，通过 GitHub PR 合并。另一台机器只在工作区干净时执行：

```bash
git fetch origin
git switch main
git pull --ff-only origin main
```

不要使用以下方式同步：

- 两台机器共享同一个网络盘/云盘 checkout。
- 提交 `.venv`、`deps`、CMake cache 或编译产物。
- 在 Mac 上使用 Windows `.pyd`，或在 Windows 使用 Mac `.so`。
- 为解决冲突而直接覆盖 Core 参数 JSON。
- 使用 `git reset --hard`、强制 push 或未经检查的全仓库换行标准化。

## 10. 发布前仍建议补齐的项目配置

以下不是完成本指南首次 Mac source checkout 的阻塞项，但属于真正可维护的双平台生产工程缺口：

1. 新增 `scripts/build_prusa_bridge_macos.sh`，把第 5.4/5.5 节命令固化并校验架构、commit 与补丁。
2. 为测试/绘图声明明确 optional dependencies，例如 `dev` 或 `plot`，避免 Matplotlib 静默缺失。
3. 增加跨平台 constraints/lock 文件，固定两端验证过的 Python 依赖。
4. 将 UI 用户设置迁出受 Git 跟踪的默认 JSON。
5. 为 Core NPZ 预览增加浏览器上传路径，移除对 Windows picker 的硬依赖。
6. 修复 package data/构建配置并重新构建 wheel；当前 wheel 不能代表当前源码，也没有 macOS 原生扩展。
7. 增加 GitHub Actions `windows-latest`/`macos-14` 的纯 Python测试矩阵；原生 Prusa 构建可做缓存后的单独 workflow，避免每次提交都全量编译。
8. 在 `.gitattributes` 中只对新加入的 shell 脚本明确 `eol=lf`。若要全仓库统一换行，单独提交并先评估大规模 diff，不在环境搭建提交中执行。

## 11. 常见失败定位

| 现象 | 优先检查 | 不要做 |
| --- | --- | --- |
| `No module named external_npz_preprocessor` | 是否安装了 `-e './packages/offline_path_planner[test]'` | 不要把 Core 源码复制进根包 |
| `PrusaBridgeUnavailable` | `.so` 是否生成、Python 架构是否一致、`bridge_info()` 原始 reason | 不要复制 Windows `.pyd` |
| `bad CPU type in executable` | Homebrew/Python/CMake 产物是否混用 arm64 与 x86_64 | 不要同时用 Rosetta 与原生工具链 |
| `CMath::CMath target not found` | Prusa 官方记录的 CMake 兼容问题 | 不要顺便换 Prusa 版本 |
| `pytest` 收集 Boost 测试并 `SystemExit` | 是否使用了 `python -m pytest -q tests` | 不要在仓库根裸跑无路径 pytest |
| Core 绘图测试不生成 PNG | Matplotlib 是否安装、`python -c 'import matplotlib'` | 不要把静默跳过当成功 |
| UI 能开但 Prusa 切片失败 | Prusa 补丁、bridge、Core 双包导入和输出目录权限 | 不要先怀疑上位机 |
| 主 UI 的 Core NPZ 选择失败 | 这是当前非 Windows 明确不支持的 picker | 不要反复重装 tkinter/浏览器 |
| `git pull` 被两个参数 JSON 阻止 | 先备份本机设置，检查 skip-worktree 状态，人工合并 | 不要直接丢弃未备份的参数 |

## 12. 上位机边界

本指南不配置上位机。只有出现以下情况时，才需要另开任务检查 live `kuka_ram_ws`：

- Core 系统 NPZ 契约字段、dtype、shape、事件语义或分片规则发生变化。
- Mac 生成物在离线契约验证通过，但上位机消费者拒绝或解释不同。
- 需要做真实机器人、ROS 2、RSI、控制中心或实时队列联调。

届时应修改和测试 live 上位机仓库，不应把 `packages/offline_path_planner` 中保留的只读消费者摘录当成上位机生产代码。
