# Workspace packages

This repository contains two independently installable Python distributions:

| Distribution | Location | Responsibility |
| --- | --- | --- |
| `kuka-slicer` | repository root | STL geometry to `external_layer_paths_v1` source NPZ |
| `kuka-offline-planner` | `packages/offline_path_planner` | GCode or source NPZ to the existing system NPZ contract |

The packages communicate only through an NPZ file written to disk. Production
code in either package must not import the other package. This keeps installation,
runtime dependencies, release cadence, and failures independent while the two
packages temporarily share one Git repository.

The imported planner subtree also retains handoff evidence and read-only
upper-computer consumer excerpts. Those files are audit material and are not
installed by the `kuka-offline-planner` Python distribution.
