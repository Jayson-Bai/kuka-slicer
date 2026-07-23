# Missing or ambiguous items

This document distinguishes observed absence from inference. None of the items
below was silently reconstructed, copied from another repository, or repaired
during phase one.

## 1. Upstream source commit object is not present

- **Title:** documented upstream SHA differs from the local source-tag target.
- **Location:** `EXTERNAL_CODEX_HANDOFF.md:19`;
  tag metadata for `offline-planner-source-v1`.
- **Direct evidence:** the handoff names
  `eb9091bcf405eaca2dddb07f2998bb3f25c12601`. The local repository does not
  contain that object. The annotated source tag dereferences to
  `927f47e46b52a342c63b688a1ffbe557a932d94b`.
- **Missing object:** original commit object or an explicit provenance mapping
  from upstream SHA `eb9091...` to imported commit `927f47e...`.
- **Impact:** exact commit-history provenance cannot be independently proven
  from this repository alone. File content integrity is still verifiable.
- **Minimum owner-provided evidence:** a bundle containing `eb9091...`, or a
  signed/committed note explaining the import mapping.
- **Blocks phase one:** no. `SOURCE_SHA256SUMS` passed 124/124 and the golden
  passed 3/3.
- **Blocks phase two:** only if historical commit identity is required for
  compliance/release provenance.

## 2. Upper-computer UI handoff is an incomplete runtime excerpt

- **Title:** supplied `ui_panel.py` cannot run as a standalone package from this
  repository.
- **Location:** `src/my_project/my_project_ui/my_project_ui/ui_panel.py:14`,
  `src/my_project/my_project_ui/my_project_ui/ui_panel.py:5231`,
  `src/my_project/my_project_ui/my_project_ui/ui_panel.py:5908`.
- **Direct evidence:** it imports `UiStatus` and `ExtruderLatencyStatus`, but the
  handoff contains only `PlannedEvent.msg` and `TrajectoryPoint.msg`
  (`handoff/SOURCE_TREE.tsv:100`–`handoff/SOURCE_TREE.tsv:101`). It references
  `my_project_ui.vtk_path_preview` and `startup.launch.py`, neither of which is
  tracked.
- **Missing objects:** complete UI package metadata/modules, message package,
  launch package, and runtime nodes.
- **Impact:** no end-to-end UI/launch/hardware test can be executed here.
- **Minimum owner-provided files:** the complete upper-computer repository at a
  pinned commit, or a consumer CI result against this handoff's NPZ golden.
- **Blocks phase one:** no; the file is read-only consumer evidence.
- **Blocks phase two:** yes for migrating and validating the UI invocation.

## 3. Read-only C++ consumer is not buildable in this repository

- **Title:** `control_center` has source excerpts but no build/package closure.
- **Location:** `handoff/SOURCE_TREE.tsv:36`–`handoff/SOURCE_TREE.tsv:39`.
- **Direct evidence:** the loader includes/uses `cnpy`
  (`src/my_project/control_center/src/npz_loader.cpp:1`), and the queue uses ROS
  messages, but no `control_center/CMakeLists.txt` or `package.xml` is included.
- **Missing objects:** CMake/package metadata, exact cnpy dependency version,
  remaining control-center source, and consumer tests.
- **Impact:** C++ compatibility is based on static inspection rather than an
  executable consumer test.
- **Minimum owner-provided files:** complete consumer build tree and one test
  that loads `handoff/golden/external-template-system.npz`.
- **Blocks phase one:** no.
- **Blocks phase two:** yes before any NPZ schema change.

## 4. NPZ schema has no version identifier

- **Title:** the system NPZ is a de facto, unversioned contract.
- **Location:**
  `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:257`.
- **Direct evidence:** 25 arrays and vocabularies are written, but no
  `schema_version`/format array is present. Package versions are all `0.0.0`
  (`src/my_project/path_processing_core/setup.py:7`).
- **Missing object:** agreed schema version, compatibility policy, and dual
  producer/consumer contract tests.
- **Impact:** a field/dtype/vocabulary change can break C++ at load time because
  fields 1–15 are accessed with `npz.at`
  (`src/my_project/control_center/src/npz_loader.cpp:406`).
- **Minimum owner decision:** approve a versioning and backward-compatibility
  design before implementation.
- **Blocks phase one:** no; current schema is frozen and documented.
- **Blocks phase two:** yes for schema evolution.

## 5. `preview_layer_index` has asymmetric consumer support

- **Title:** Python preview prefers a field the supplied C++ loader ignores.
- **Location:**
  `src/my_project/gcode_planner/gcode_planner/path_preview.py:339`;
  `src/my_project/control_center/src/npz_loader.cpp:449`.
- **Direct evidence:** Python reads `preview_layer_index`; C++ loads
  `layer_index`, `total_layers`, `path_id`, and `path_end_flag` but has no
  preview-layer member or read.
- **Missing decision:** whether preview grouping is intentionally UI-only or
  should be propagated into realtime messages.
- **Impact:** consumers can display different layer groupings without violating
  the current runtime trajectory contract.
- **Minimum owner decision:** declare this field UI-only or supply the intended
  runtime behavior.
- **Blocks phase one:** no.
- **Blocks phase two:** only if layer semantics are changed.

## 6. External UI dependency is undeclared

- **Title:** advertised Qt entry point lacks dependency metadata.
- **Location:** `src/my_project/external_npz_preprocessor/setup.py:24`,
  `src/my_project/external_npz_preprocessor/external_npz_preprocessor/app.py:7`.
- **Direct evidence:** `python_qt_binding` is imported, while `setup.py` and
  `package.xml` list only NumPy/shared-core runtime dependencies
  (`src/my_project/external_npz_preprocessor/setup.py:14`,
  `src/my_project/external_npz_preprocessor/package.xml:10`).
- **Missing object:** an explicit optional UI dependency or documented ROS/Qt
  environment prerequisite.
- **Impact:** the CLI may install and work while the UI entry point fails.
- **Minimum owner decision:** choose separate UI package vs optional extra.
- **Blocks phase one:** no.
- **Blocks phase two:** yes for a supported standalone UI distribution.

## 7. GCode offline CLI is coupled to `rclpy`

- **Title:** pure parse/export imports ROS node support at module load.
- **Location:** `src/my_project/gcode_planner/gcode_planner/cli.py:10`,
  `src/my_project/gcode_planner/gcode_planner/gcode_parser.py:8`.
- **Direct evidence:** the CLI imports parser functions from a module that
  imports `rclpy` before defining pure `parse_gcode_lines`.
- **Missing decision:** whether standalone operation is a release requirement
  and which module owns the ROS adapter.
- **Impact:** plain Python cannot run the GCode CLI without ROS or a test shim.
- **Minimum owner decision:** approve a pure-parser/ROS-adapter split.
- **Blocks phase one:** no; representative export passed in ROS 2 Humble.
- **Blocks phase two:** yes for non-ROS standalone packaging.

## 8. Optional plotting dependencies are not declared

- **Title:** plot features depend on Matplotlib without an optional dependency
  contract.
- **Location:**
  `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1834`;
  tracked plot scripts at `handoff/SOURCE_TREE.tsv:32`–`handoff/SOURCE_TREE.tsv:34`.
- **Direct evidence:** Matplotlib is imported only at plotting time and is not
  listed by any `setup.py`.
- **Missing object:** `plot` optional-extra or documented developer
  environment.
- **Impact:** export works, but plot flags/scripts can silently skip or fail.
- **Minimum owner decision:** define whether plotting is supported product
  functionality or developer tooling.
- **Blocks phase one:** no.

## 9. Dormant alternative implementations need intent clarification

- **Title:** two private implementations have no production call site.
- **Location:**
  `src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:685`
  and
  `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1946`.
- **Direct evidence:** `_fit_validated_spline` and `_npz_exporter` are defined.
  **Inference:** repository-wide reference search found no production caller.
- **Missing decision:** future feature, obsolete path, or retained fallback.
- **Impact:** maintainers may wrongly assume external paths are spline-fitted
  or may update the wrong NPZ writer.
- **Minimum owner decision:** identify intended algorithm/writer; then add a
  test before activation or deletion.
- **Blocks phase one:** no.
- **Blocks phase two:** no, unless refactoring these areas.

## 10. Static quality baseline is red

- **Title:** ament style tests do not pass at handoff.
- **Location:** detailed in `TEST_BASELINE.md`.
- **Direct evidence:** flake8 reports 44 issues; pep257 reports 3 D213 issues;
  copyright skips.
- **Missing object:** none; these are existing source-quality issues.
- **Impact:** CI that treats these tests as required will be red.
- **Minimum owner decision:** approve a source-formatting-only phase after the
  audit, isolated from functional refactoring.
- **Blocks phase one:** no; source fixes are forbidden in this phase.
- **Blocks phase two:** policy decision.

## 11. Hardware/realtime behavior is outside the handoff

- **Title:** RSI, UART, startup, and hardware control are referenced but not
  supplied as executable scope.
- **Location:** upper-computer launch call
  `src/my_project/my_project_ui/my_project_ui/ui_panel.py:5903`; historical
  design records under `docs/superpowers/`.
- **Direct evidence:** the handoff path list contains only selected consumer
  excerpts (`handoff/HANDOFF_PATHS.txt:23`–`handoff/HANDOFF_PATHS.txt:29`).
- **Missing objects:** realtime repositories and hardware test environment.
- **Impact:** no claim can be made about live KUKA/RSI/event execution.
- **Minimum owner-provided evidence:** pinned realtime repository plus existing
  integration-test instructions/results.
- **Blocks phase one:** no.
- **Blocks phase two:** yes before changing event timing or launch integration.

## Phase-one status

All observed missing/ambiguous items are now explicit. None prevents completion
of the read-only baseline audit because integrity, functional tests, and both
offline export paths were reproduced. Owner approval is still required before
phase two, exactly as required by `EXTERNAL_CODEX_HANDOFF.md:193`.
