# Codex project instructions

## Project and cross-repository terminology

- The current repository root is `F:\CodeX_ws\kuka_slicer`. Treat “本项目”、
  “切片器”、“离线端” and `kuka_slicer` as this repository unless the user
  explicitly says otherwise.
- In this project, “上位机” means the live `kuka_ram_ws` repository, not the
  vendored or copied code under this repository.
- The live upper-computer repository is available to Windows Codex Desktop at:
  `\\wsl.localhost\Ubuntu2204_ros2\home\jayson\kuka_ram_ws`
- The same repository path inside WSL and the development container is:
  `/home/jayson/kuka_ram_ws`
- Its development container is `kuka_drivers_mounted`.

## Routing work to the upper-computer repository

- When the user explicitly asks to inspect, change, test, or debug “上位机”,
  operate in the live repository path above.
- Do not implement an upper-computer change in
  `packages/offline_path_planner` or another copy inside `kuka_slicer` as a
  substitute for changing the live repository.
- Use the Windows UNC path for direct file inspection and edits from Codex
  Desktop. Run Ubuntu/ROS 2 commands and tests inside `kuka_drivers_mounted`
  with `docker exec`, using `/home/jayson/kuka_ram_ws`.
- Before editing the upper-computer repository, verify that the UNC path is
  accessible and that `kuka_drivers_mounted` is the intended running
  container. If either is unavailable, report the problem instead of silently
  falling back to a copied directory.
- Changes in the upper-computer repository are outside the `kuka_slicer` Git
  worktree. When a task touches both repositories, inspect and report the Git
  status of each repository separately.
- Do not copy or synchronize upper-computer changes back into
  `kuka_slicer/packages/offline_path_planner` unless the user explicitly asks
  for that synchronization.

## Token-efficient workflow

- For a targeted file, symbol, or small change, use `rg` and read only the
  necessary files or ranges; do not load broad directories without a reason.
- Use Graphify only for cross-module dependencies, call paths, ownership, or
  change-impact questions. Before querying, if relevant source changed since
  the last index update or freshness is unknown, run
  `graphify update . --no-cluster`; then query with a bounded budget. Add
  `--force` after a refactor that deletes or moves code.
- Do not automatically run Graphify extraction, clustering, HTML generation,
  Git hooks, or load all of `graphify-out/graph.json`. The source code remains
  the ground truth.
- Follow `@RTK.md` for high-output read-only commands. On a failure, retain
  the complete error using the native command or `rtk proxy`; do not filter
  diagnostic output away.
- Summarize findings and reuse them within the task instead of re-reading or
  reprinting already resolved logs and diffs.

@RTK.md
