"""Simple Qt UI for external NPZ preprocessing."""

from __future__ import annotations

import traceback

from python_qt_binding import QtCore, QtWidgets

from .export_runner import (
    convert_external_npz,
    default_output_npz_dir,
    default_output_path_for_source,
    default_source_npz_template_dir,
    ensure_default_data_dirs,
    load_shared_export_offsets,
    resolve_output_path,
)
from path_processing_core.head_calibration import DEFAULT_HEAD_CALIBRATION_PATH

from .param_config import default_print_params_path, load_print_params, save_print_params
from .process_params import (
    FiberProcessParams,
    ProcessParams,
    RESIN_FIXED_BEAD_WIDTH_MM,
    ResinProcessParams,
)
from .source_npz import load_source_npz


class ExternalNpzPreprocessorWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("External NPZ Preprocessor")
        self.setMinimumSize(720, 560)
        self.resize(960, 760)
        ensure_default_data_dirs()
        self._last_auto_output = ""
        self._build_ui()
        self._load_saved_params()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.source_edit = QtWidgets.QLineEdit()
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("为空时默认保存到 data/output_npz/<源文件名>/<源文件名>.npz")
        form.addRow("源 NPZ 位置", self._path_row(self.source_edit, self._choose_source))
        form.addRow("输出 NPZ 位置", self._path_row(self.output_edit, self._choose_output))
        self.source_edit.textChanged.connect(self._on_source_path_changed)
        layout.addLayout(form)

        self.offset_info_label = QtWidgets.QLabel()
        self.offset_info_label.setWordWrap(True)
        layout.addWidget(self.offset_info_label)
        self._refresh_offset_info()

        params_group = QtWidgets.QGroupBox("工艺参数")
        params_layout = QtWidgets.QGridLayout(params_group)
        self.resin_layer_height = self._spin(0.5)
        self.resin_extrusion_scale = self._spin(1.0)
        self.resin_feed = self._spin(10.0)
        self.first_layer_resin_feed = self._spin(10.0, minimum=0.001)
        self.resin_temp = self._spin(250.0, maximum=500.0)
        self.resin_prime_length = self._spin(18.0)
        self.resin_prime_speed = self._spin(15.0)
        self.resin_retract_length = self._spin(15.0)
        self.resin_retract_speed = self._spin(30.0)
        self.resin_fan = QtWidgets.QCheckBox()
        self.resin_fan.setChecked(True)
        self.fiber_layer_height = self._spin(0.1)
        self.fiber_extrusion_scale = self._spin(1.0)
        self.fiber_feed = self._spin(10.0)
        self.first_layer_fiber_feed = self._spin(10.0, minimum=0.001)
        self.fiber_start_accel = self._spin(2.0, minimum=0.001)
        self.fiber_temp = self._spin(250.0, maximum=500.0)
        self.fiber_prime_length = self._spin(12.0)
        self.fiber_prime_speed = self._spin(5.0)
        self.fiber_retract_length = self._spin(10.0)
        self.fiber_retract_speed = self._spin(5.0)
        self.fiber_fan = QtWidgets.QCheckBox()
        self.fiber_fan.setChecked(True)
        self.travel_feed = self._spin(10.0)
        self.first_layer_travel_feed = self._spin(10.0, minimum=0.001)
        self.prime_settle_s = self._spin(0.5)
        self.default_a = self._spin(0.0, minimum=-360.0, maximum=360.0)
        self.default_b = self._spin(0.0, minimum=-360.0, maximum=360.0)
        self.default_c = self._spin(0.0, minimum=-360.0, maximum=360.0)
        self.start_x = self._spin(0.0, minimum=-100000.0, maximum=100000.0)
        self.start_y = self._spin(0.0, minimum=-100000.0, maximum=100000.0)
        fixed_resin_width = QtWidgets.QLabel(f"固定树脂线宽: {RESIN_FIXED_BEAD_WIDTH_MM:.1f} mm")
        params_layout.addWidget(fixed_resin_width, 0, 0, 1, 4)
        rows = [
            (
                "首层树脂打印速度 mm/s",
                self.first_layer_resin_feed,
                "首层纤维打印速度 mm/s",
                self.first_layer_fiber_feed,
            ),
            (
                "首层空走速度 mm/s",
                self.first_layer_travel_feed,
                "",
                QtWidgets.QLabel(""),
            ),
            ("树脂层高 mm", self.resin_layer_height, "纤维层高 mm", self.fiber_layer_height),
            ("树脂挤出倍率", self.resin_extrusion_scale, "纤维挤出倍率", self.fiber_extrusion_scale),
            ("树脂非首层打印速度 mm/s", self.resin_feed, "纤维非首层打印速度 mm/s", self.fiber_feed),
            (
                "",
                QtWidgets.QLabel(""),
                "纤维起步加速时间 s",
                self.fiber_start_accel,
            ),
            ("树脂温度 C", self.resin_temp, "纤维温度 C", self.fiber_temp),
            ("树脂预挤出长度 mm", self.resin_prime_length, "纤维预挤出长度 mm", self.fiber_prime_length),
            ("树脂预挤出速度 mm/s", self.resin_prime_speed, "纤维预挤出速度 mm/s", self.fiber_prime_speed),
            ("树脂回抽长度 mm", self.resin_retract_length, "纤维回抽长度 mm", self.fiber_retract_length),
            ("树脂回抽速度 mm/s", self.resin_retract_speed, "纤维回抽速度 mm/s", self.fiber_retract_speed),
            ("树脂风扇", self.resin_fan, "纤维风扇", self.fiber_fan),
            ("预挤出稳定等待 s", self.prime_settle_s, "非首层空走速度 mm/s", self.travel_feed),
            ("左下角 X mm", self.start_x, "左下角 Y mm", self.start_y),
            ("", QtWidgets.QLabel(""), "默认 A", self.default_a),
            ("", QtWidgets.QLabel(""), "默认 B", self.default_b),
            ("", QtWidgets.QLabel(""), "默认 C", self.default_c),
        ]
        for row, (left_label, left_widget, right_label, right_widget) in enumerate(rows, start=1):
            if left_label:
                params_layout.addWidget(QtWidgets.QLabel(left_label), row, 0)
                params_layout.addWidget(left_widget, row, 1)
            if right_label:
                params_layout.addWidget(QtWidgets.QLabel(right_label), row, 2)
                params_layout.addWidget(right_widget, row, 3)
        params_scroll = QtWidgets.QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        params_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        params_scroll.setWidget(params_group)
        layout.addWidget(params_scroll, 2)

        buttons = QtWidgets.QHBoxLayout()
        self.validate_button = QtWidgets.QPushButton("校验源 NPZ")
        self.save_params_button = QtWidgets.QPushButton("保存打印参数json文件")
        self.convert_button = QtWidgets.QPushButton("开始处理")
        self.validate_button.clicked.connect(self._validate_source)
        self.save_params_button.clicked.connect(self._save_params)
        self.convert_button.clicked.connect(self._convert)
        buttons.addWidget(self.validate_button)
        buttons.addWidget(self.save_params_button)
        buttons.addWidget(self.convert_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)


    def _refresh_offset_info(self):
        try:
            offset, resin_z = load_shared_export_offsets()
            self.offset_info_label.setText(
                "偏置数据源: %s\n树脂Z: %.3f mm, 纤维XYZ: (%.3f, %.3f, %.3f) mm"
                % (DEFAULT_HEAD_CALIBRATION_PATH, resin_z, offset[0], offset[1], offset[2])
            )
        except Exception as exc:
            self.offset_info_label.setText(f"偏置数据读取失败: {exc}")

    def _path_row(self, edit, callback):
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QtWidgets.QPushButton("选择")
        button.clicked.connect(callback)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    def _spin(self, value, minimum=0.0, maximum=100000.0):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _choose_source(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择源 NPZ 位置",
            str(default_source_npz_template_dir()),
            "NPZ (*.npz)",
        )
        if path:
            self.source_edit.setText(path)

    def _choose_output(self):
        source = self.source_edit.text().strip()
        start_path = (
            default_output_path_for_source(source)
            if source
            else default_output_npz_dir() / "output.npz"
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择输出 NPZ 位置", str(start_path), "NPZ (*.npz)"
        )
        if path:
            self.output_edit.setText(path)
            self._last_auto_output = ""

    def _on_source_path_changed(self, text):
        source = text.strip()
        if not source:
            return
        current_output = self.output_edit.text().strip()
        if current_output and current_output != self._last_auto_output:
            return
        auto_output = str(default_output_path_for_source(source))
        self.output_edit.setText(auto_output)
        self._last_auto_output = auto_output


    def _load_saved_params(self):
        path = default_print_params_path()
        try:
            if path.is_file():
                self._apply_params(load_print_params(path))
                self._append_log(f"已读取打印参数json文件: {path}")
        except Exception as exc:
            self._append_log(f"读取打印参数json文件失败: {exc}")

    def _apply_params(self, params: ProcessParams):
        self.resin_layer_height.setValue(params.resin.layer_height_mm)
        self.resin_extrusion_scale.setValue(params.resin.extrusion_scale)
        self.resin_feed.setValue(params.resin.feed_mm_s)
        self.first_layer_resin_feed.setValue(params.resin.first_layer_feed_mm_s)
        self.resin_temp.setValue(params.resin.temperature_c)
        self.resin_prime_length.setValue(params.resin.prime_length_mm)
        self.resin_prime_speed.setValue(params.resin.prime_speed_mm_s)
        self.resin_retract_length.setValue(params.resin.retract_length_mm)
        self.resin_retract_speed.setValue(params.resin.retract_speed_mm_s)
        self.resin_fan.setChecked(params.resin.fan_enabled)
        self.fiber_layer_height.setValue(params.fiber.layer_height_mm)
        self.fiber_extrusion_scale.setValue(params.fiber.extrusion_scale)
        self.fiber_feed.setValue(params.fiber.feed_mm_s)
        self.first_layer_fiber_feed.setValue(params.fiber.first_layer_feed_mm_s)
        self.fiber_start_accel.setValue(params.fiber.start_accel_s)
        self.fiber_temp.setValue(params.fiber.temperature_c)
        self.fiber_prime_length.setValue(params.fiber.prime_length_mm)
        self.fiber_prime_speed.setValue(params.fiber.prime_speed_mm_s)
        self.fiber_retract_length.setValue(params.fiber.retract_length_mm)
        self.fiber_retract_speed.setValue(params.fiber.retract_speed_mm_s)
        self.fiber_fan.setChecked(params.fiber.fan_enabled)
        self.travel_feed.setValue(params.travel_feed_mm_s)
        self.first_layer_travel_feed.setValue(params.first_layer_travel_feed_mm_s)
        self.prime_settle_s.setValue(params.prime_settle_s)
        self.default_a.setValue(params.default_a)
        self.default_b.setValue(params.default_b)
        self.default_c.setValue(params.default_c)
        self.start_x.setValue(params.start_x_mm)
        self.start_y.setValue(params.start_y_mm)

    def _params(self) -> ProcessParams:
        return ProcessParams(
            resin=ResinProcessParams(
                layer_height_mm=self.resin_layer_height.value(),
                extrusion_scale=self.resin_extrusion_scale.value(),
                feed_mm_s=self.resin_feed.value(),
                first_layer_feed_mm_s=self.first_layer_resin_feed.value(),
                temperature_c=self.resin_temp.value(),
                fan_enabled=self.resin_fan.isChecked(),
                prime_length_mm=self.resin_prime_length.value(),
                prime_speed_mm_s=self.resin_prime_speed.value(),
                retract_length_mm=self.resin_retract_length.value(),
                retract_speed_mm_s=self.resin_retract_speed.value(),
            ),
            fiber=FiberProcessParams(
                layer_height_mm=self.fiber_layer_height.value(),
                extrusion_scale=self.fiber_extrusion_scale.value(),
                feed_mm_s=self.fiber_feed.value(),
                first_layer_feed_mm_s=self.first_layer_fiber_feed.value(),
                start_accel_s=self.fiber_start_accel.value(),
                temperature_c=self.fiber_temp.value(),
                fan_enabled=self.fiber_fan.isChecked(),
                prime_length_mm=self.fiber_prime_length.value(),
                prime_speed_mm_s=self.fiber_prime_speed.value(),
                retract_length_mm=self.fiber_retract_length.value(),
                retract_speed_mm_s=self.fiber_retract_speed.value(),
            ),
            travel_feed_mm_s=self.travel_feed.value(),
            first_layer_travel_feed_mm_s=self.first_layer_travel_feed.value(),
            prime_settle_s=self.prime_settle_s.value(),
            default_a=self.default_a.value(),
            default_b=self.default_b.value(),
            default_c=self.default_c.value(),
            start_x_mm=self.start_x.value(),
            start_y_mm=self.start_y.value(),
        )

    def _save_params(self):
        try:
            path = save_print_params(self._params())
            self._append_log(f"已保存打印参数json文件: {path}")
        except Exception as exc:
            self._append_log(f"保存打印参数json文件失败: {exc}")

    def _validate_source(self):
        try:
            job = load_source_npz(self.source_edit.text().strip(), default_abc=self._params().default_abc)
            path_count = sum(len(layer.resin_paths) + len(layer.fiber_paths) for layer in job.layers)
            self._append_log(f"校验通过: layers={len(job.layers)}, paths={path_count}")
        except Exception as exc:
            self._append_log(f"校验失败: {exc}")

    def _convert(self):
        try:
            source = self.source_edit.text().strip()
            output = resolve_output_path(source, self.output_edit.text().strip())
            offset, resin_z = load_shared_export_offsets()
            self._refresh_offset_info()
            stats = convert_external_npz(source, self.output_edit.text().strip(), self._params())
            self._append_log(f"输出 NPZ 位置: {output}")
            self._append_log(
                "使用偏置: 树脂Z=%.3f, 纤维XYZ=(%.3f, %.3f, %.3f)"
                % (resin_z, offset[0], offset[1], offset[2])
            )
            self._append_log(
                "处理完成: rows=%d, parts=%d, total=%.3fs"
                % (stats.get("rows", 0), stats.get("parts", 0), stats.get("total_s", 0.0))
            )
        except Exception:
            self._append_log("处理失败:\n" + traceback.format_exc())

    def _append_log(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
