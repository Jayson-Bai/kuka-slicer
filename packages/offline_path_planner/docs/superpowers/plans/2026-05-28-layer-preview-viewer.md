# Layer PNG Preview Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After NPZ export, show a button to view per-layer XY path PNGs in a popup dialog with left/right navigation and mouse-wheel zoom.

**Architecture:** Modify `_plot_single_layer()` in `npz_exporter.py` to save PNGs to a centralized `layer_previews/` folder. Add `_LayerViewerDialog` class and a "查看图层" button in `ui_panel.py`, wired to the export-complete signal.

**Tech Stack:** Python, PyQt5 (python_qt_binding), matplotlib, numpy

---

### Task 1: Centralize PNG output path in `npz_exporter.py`

**Files:**
- Modify: `src/my_project/gcode_planner/gcode_planner/npz_exporter.py:1058-1060`

- [ ] **Step 1: Change `_plot_single_layer()` output path**

Current (lines 1058-1060):
```python
    out_dir = Path(base_root) / f"layer_{layer:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer_{layer:04d}.png"
```

Replace with:
```python
    out_dir = Path(base_root) / "layer_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer_{layer:04d}.png"
```

### Task 2: Add `_LayerViewerDialog` class to `ui_panel.py`

**Files:**
- Modify: `src/my_project/my_project_ui/my_project_ui/ui_panel.py` — add new class before `_UiStatusWidget`

- [ ] **Step 1: Add `_LayerViewerDialog` class**

Insert this new class just before `class _UiStatusWidget(QtWidgets.QWidget):` (line 251):

```python
import re
from pathlib import Path


class _LayerViewerDialog(QtWidgets.QDialog):
    def __init__(self, npz_dir: str, parent=None):
        super().__init__(parent)
        self._npz_dir = npz_dir
        self._images: list[Path] = []
        self._index = 0
        self._zoom = 1.0

        self.setWindowTitle(f"Layer Preview - {Path(npz_dir).name}")
        self.resize(900, 700)

        self._scan_images()
        self._build_ui()
        self._show_current()

    def _scan_images(self):
        preview_dir = Path(self._npz_dir) / "layer_previews"
        if not preview_dir.is_dir():
            return
        pattern = re.compile(r"layer_(\d+)\.png$")
        files = []
        for f in sorted(preview_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                files.append((int(m.group(1)), f))
        files.sort(key=lambda x: x[0])
        self._images = [f[1] for f in files]

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Top bar: prev / label / next
        top_bar = QtWidgets.QHBoxLayout()
        self._btn_prev = QtWidgets.QPushButton("←  Prev")
        self._btn_prev.setFixedWidth(100)
        self._btn_prev.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_next = QtWidgets.QPushButton("Next  →")
        self._btn_next.setFixedWidth(100)
        self._btn_next.setCursor(QtCore.Qt.PointingHandCursor)
        self._label_index = QtWidgets.QLabel("")
        self._label_index.setAlignment(QtCore.Qt.AlignCenter)
        top_bar.addWidget(self._btn_prev)
        top_bar.addStretch()
        top_bar.addWidget(self._label_index)
        top_bar.addStretch()
        top_bar.addWidget(self._btn_next)
        layout.addLayout(top_bar)

        # Image area
        self._scene = QtWidgets.QGraphicsScene(self)
        self._view = _ZoomableGraphicsView(self._scene)
        self._view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self._view.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )
        layout.addWidget(self._view, 1)

        # Bottom bar: zoom label + reset + close
        bottom_bar = QtWidgets.QHBoxLayout()
        self._label_zoom = QtWidgets.QLabel("Zoom: 100%")
        btn_reset = QtWidgets.QPushButton("Reset View")
        btn_reset.setCursor(QtCore.Qt.PointingHandCursor)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setCursor(QtCore.Qt.PointingHandCursor)
        bottom_bar.addWidget(self._label_zoom)
        bottom_bar.addStretch()
        bottom_bar.addWidget(btn_reset)
        bottom_bar.addWidget(btn_close)
        layout.addLayout(bottom_bar)

        # Connections
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        btn_reset.clicked.connect(self._on_reset)
        btn_close.clicked.connect(self.accept)

    def _show_current(self):
        if not self._images:
            self._label_index.setText("No images found")
            self._btn_prev.setEnabled(False)
            self._btn_next.setEnabled(False)
            return
        total = len(self._images)
        self._label_index.setText(f"Layer {self._index + 1} / {total}")
        self._btn_prev.setEnabled(self._index > 0)
        self._btn_next.setEnabled(self._index < total - 1)

        pixmap = QtGui.QPixmap(str(self._images[self._index]))
        if pixmap.isNull():
            self._label_index.setText("Failed to load image")
            return
        self._scene.clear()
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self._on_reset()

    def _on_prev(self):
        if self._index > 0:
            self._index -= 1
            self._show_current()

    def _on_next(self):
        if self._index < len(self._images) - 1:
            self._index += 1
            self._show_current()

    def _on_reset(self):
        self._view.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self._zoom = 1.0
        self._label_zoom.setText("Zoom: 100%")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self._on_prev()
        elif event.key() == QtCore.Qt.Key_Right:
            self._on_next()
        else:
            super().keyPressEvent(event)


class _ZoomableGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)

    def wheelEvent(self, event):
        factor = 1.15
        if event.angleDelta().y() > 0:
            zoom = factor
        else:
            zoom = 1.0 / factor
        self.scale(zoom, zoom)
```

- [ ] **Step 2: Update zoom label on scale changed**

The `_ZoomableGraphicsView` needs to notify the dialog when zoom changes. Replace the `wheelEvent` approach with a signal:

In `_ZoomableGraphicsView`:
```python
class _ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoom_changed = QtCore.pyqtSignal(float)

    def wheelEvent(self, event):
        factor = 1.15
        if event.angleDelta().y() > 0:
            zoom = factor
        else:
            zoom = 1.0 / factor
        self.scale(zoom, zoom)
        self.zoom_changed.emit(self.transform().m11())
```

In `_LayerViewerDialog._build_ui()`, after creating the view:
```python
        self._view.zoom_changed.connect(self._on_zoom_changed)
```

Add method:
```python
    def _on_zoom_changed(self, scale):
        self._zoom = scale
        self._label_zoom.setText(f"Zoom: {scale * 100:.0f}%")
```

### Task 3: Add "查看图层" button and wire up signals

**Files:**
- Modify: `src/my_project/my_project_ui/my_project_ui/ui_panel.py`

- [ ] **Step 1: Add `_last_npz_dir` and `_view_layers_btn` in `_UiStatusWidget.__init__`**

After line 262 (`self._extrude_scale_current = 1.0`), add:
```python
        self._last_npz_dir: str | None = None
```

- [ ] **Step 2: Add "查看图层" button after Export NPZ button**

After the Export NPZ button block (after line 828 `export_btn_row.addWidget(self._btn_export_npz)`), add:
```python
        # View Layer Images button
        self._btn_view_layers = QtWidgets.QPushButton("View Layer Images")
        self._btn_view_layers.setObjectName("btnViewLayers")
        self._btn_view_layers.setMinimumHeight(28)
        self._btn_view_layers.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_view_layers.setEnabled(False)
        export_btn_row.addWidget(self._btn_view_layers)
```

Note: This adds the button in the same `export_btn_row` as Export NPZ. If the user wants it on a separate row, change to a new layout. Based on the requirement "below Export NPZ", let's use a separate row:

Actually, looking at the code more carefully, `export_btn_row` contains only the Export NPZ button. Adding the View button there would put them side by side. To put it below, add a new row:

```python
        # Export button + progress
        export_btn_row = QtWidgets.QHBoxLayout()
        export_btn_row.setSpacing(8)
        self._btn_export_npz = QtWidgets.QPushButton("Export NPZ")
        ...
        export_btn_row.addWidget(self._btn_export_npz)
        export_layout.addLayout(export_btn_row)

        # View Layer Images button (below Export NPZ)
        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(8)
        self._btn_view_layers = QtWidgets.QPushButton("View Layer Images")
        self._btn_view_layers.setObjectName("btnViewLayers")
        self._btn_view_layers.setMinimumHeight(28)
        self._btn_view_layers.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_view_layers.setEnabled(False)
        view_row.addWidget(self._btn_view_layers)
        export_layout.addLayout(view_row)
```

- [ ] **Step 3: Connect button click and enable on export success**

In `_UiStatusWidget.__init__`, after the existing signal connections (after line 847 `self.export_progress.connect(...)`), add:
```python
        self._btn_view_layers.clicked.connect(self._on_view_layers)
```

In `_on_export_finished` (line 1623), add the enable logic:
```python
    def _on_export_finished(self, success, message):
        self._btn_export_npz.setEnabled(True)
        self._export_progress.setVisible(False)
        if success:
            self._export_status.setText(message)
            self._export_status.setStyleSheet("color: #1b6e3c;")
            # Store NPZ directory and enable view button
            npz_path = self._npz_out_input.text().strip()
            if npz_path:
                self._last_npz_dir = os.path.dirname(npz_path)
                self._btn_view_layers.setEnabled(True)
        else:
            self._export_status.setText(f"Export failed: {message}")
            self._export_status.setStyleSheet("color: #b42318;")
```

Add the handler method:
```python
    def _on_view_layers(self):
        if not self._last_npz_dir or not os.path.isdir(self._last_npz_dir):
            self._export_status.setText("No NPZ export directory found.")
            self._export_status.setStyleSheet("color: #b42318;")
            return
        dlg = _LayerViewerDialog(self._last_npz_dir, self)
        dlg.exec_()
```

- [ ] **Step 4: Verify imports**

Ensure `re` and `pathlib` are importable (they are stdlib, so no changes needed). `_LayerViewerDialog` uses `re` and `Path` internally — no global import changes needed.

### Task 4: Verify

- [ ] **Step 1: Check file for syntax errors**

Run: `python3 -c "import ast; ast.parse(open('src/my_project/my_project_ui/my_project_ui/ui_panel.py').read()); print('OK')"`
Run: `python3 -c "import ast; ast.parse(open('src/my_project/gcode_planner/gcode_planner/npz_exporter.py').read()); print('OK')"`

Expected: Both print `OK`.
