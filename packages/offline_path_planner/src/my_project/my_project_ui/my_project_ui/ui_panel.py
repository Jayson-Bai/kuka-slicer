from python_qt_binding import QtCore, QtWidgets, QtGui
from rqt_gui_py.plugin import Plugin
import rclpy
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
import subprocess
import os
import signal
import json
import threading
import time
import math

from my_project_interfaces.msg import UiStatus, ExtruderLatencyStatus, TrajectoryPoint

from std_msgs.msg import String as StringMsg
import re
from pathlib import Path

from path_processing_core.head_calibration import (
    DEFAULT_DATA_ROOT,
    DEFAULT_HEAD_CALIBRATION_PATH,
    HeadCalibration,
    load_head_calibration,
    save_head_calibration,
)


# Share the planner calibration data root so save/load stay stable across rqt restarts.
_DEFAULT_DATA_ROOT = DEFAULT_DATA_ROOT
_DEFAULT_GCODE_INPUT_DIR = str(_DEFAULT_DATA_ROOT / "input_gcode")
_DEFAULT_NPZ_OUTPUT_DIR = str(_DEFAULT_DATA_ROOT / "output_npz")
_DEFAULT_NPZ_PATH = os.path.join(_DEFAULT_NPZ_OUTPUT_DIR, "test.npz")
_TEST_TOOL_CHANGE_SAFE_LIFT_DEFAULT_MM = 10.0
_PRINT_TEST_ZERO_CORRECTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_PRINT_TEST_FIBER_TOOL_ID = 1
_PRINT_TEST_RESIN_TOOL_ID = 2


def _ensure_default_data_dirs():
    for path in (
        _DEFAULT_DATA_ROOT / "input_gcode",
        _DEFAULT_DATA_ROOT / "output_npz",
        _DEFAULT_DATA_ROOT / "print_test" / "tmp",
        _DEFAULT_DATA_ROOT / "head_calibration_offsets",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _format_print_duration(seconds):
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0.0:
        return "--"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


LAUNCH_PARAMS = [
    # (param_name, default_value, description, group)
    ("center_start_delay_s", "1.0", "中心节点启动延迟（秒）", "中心节点"),
    ("npz_preload_chunks", "2", "NPZ 预加载块数", "中心节点"),
    ("queue_low", "1000", "轨迹队列低水位", "中心节点"),
    ("queue_high", "2000", "轨迹队列高水位", "中心节点"),
    ("plan_qos_depth", "2000", "Plan 话题 QoS 深度", "中心节点"),
    ("traj_prefill", "1000", "启动时轨迹预填充数量", "中心节点"),
    ("traj_low", "500", "轨迹积压低阈值", "中心节点"),
    ("traj_high", "1500", "轨迹积压高阈值", "中心节点"),
    ("xyzabc_decimals", "6", "位姿小数精度", "中心节点"),
    ("e_decimals", "6", "挤出小数精度", "中心节点"),
    ("kuka_status_raw", "false", "打印 KUKA 原始 XML 长度", "中心节点"),
    ("summary_period_ms", "200", "控制中心发布周期（ms）", "中心节点"),
    ("sen_type", "PythonDemo", "RSI XML SEN 类型", "RSI 节点"),
    ("decimal_precision", "4", "RSI 数据小数精度", "RSI 节点"),
    ("local_ip", "192.168.1.1", "RSI 本地监听 IP", "RSI 节点"),
    ("local_port", "49152", "RSI 本地监听端口", "RSI 节点"),
    ("abort_lift_mm", "100.0", "ABORT 时 Z 轴抬升距离（mm）", "RSI 节点"),
    ("abort_lift_speed_mm_s", "10.0", "ABORT 时 Z 轴抬升速度（mm/s）", "RSI 节点"),
    ("pause_safe_lift_mm", "20.0", "暂停安全抬升距离（mm）", "RSI 节点"),
    ("pause_lift_speed_mm_s", "10.0", "暂停抬升/返回速度（mm/s）", "RSI 节点"),
    ("pause_retract_mm", "2.0", "暂停树脂回抽距离（mm）", "RSI 节点"),
    ("pause_retract_speed_mm_s", "2.0", "暂停树脂回抽速度（mm/s）", "RSI 节点"),
    ("port", "/dev/ttyUSB0", "UART 串口设备路径", "UART 节点"),
    ("baudrate", "115200", "UART 波特率", "UART 节点"),
    ("extrude_scale", "1.0", "UART 挤出倍率因子", "UART 节点"),
    ("ui_publish_period_ms", "100", "UI 状态发布周期（ms）", "系统管理器"),
    ("heartbeat_timeout_s", "1.0", "心跳超时（秒）", "系统管理器"),
    ("traj_queue_limit", "5000", "UI 轨迹队列上限", "系统管理器"),
    ("event_queue_limit", "2000", "UI 事件队列上限", "系统管理器"),
    ("print_time_update_period_ms", "500", "预计时间更新周期（ms）", "系统管理器"),
    ("latency_publish_period_ms", "200", "延迟状态发布周期（ms）", "延迟监控"),
    ("latency_history_limit", "5000", "RSI 心跳缓存数量", "延迟监控"),
    ("latency_stats_window_limit", "5000", "延迟统计窗口样本数", "延迟监控"),
    ("rsi_period_ms", "4.0", "RSI 控制周期（ms）", "延迟监控"),
    ("robot_match_cache_back", "8000", "机械臂匹配缓存后向序号", "延迟监控"),
    ("robot_match_cache_forward", "1000", "机械臂匹配缓存前向序号", "延迟监控"),
    ("robot_match_search_back", "5000", "机械臂匹配搜索后向序号", "延迟监控"),
    ("robot_match_search_forward", "300", "机械臂匹配搜索前向序号", "延迟监控"),
    ("robot_match_max_error_mm", "1.0", "机械臂匹配最大空间误差（mm）", "延迟监控"),
    ("robot_match_uncertainty_min_band_mm", "0.10", "匹配不确定度最小误差带（mm）", "延迟监控"),
    ("robot_match_uncertainty_spacing_multiplier", "3.0", "匹配不确定度轨迹间距倍率", "延迟监控"),
    ("robot_match_nozzle_lever_mm", "401.68", "TCP 姿态误差等效臂长（mm）", "延迟监控"),
]

_LAUNCH_DEFAULTS = {p[0]: p[1] for p in LAUNCH_PARAMS}
_LAUNCH_GROUPS_ORDER = ["中心节点", "RSI 节点", "UART 节点", "系统管理器", "延迟监控"]
_MODE_PAGE_SELECT = "select"
_MODE_PAGE_TEST = "test"
_MODE_PAGE_PRINT = "print"
_SYSTEM_STATUS_MIN_HEIGHT = 84
_SYSTEM_STATUS_TARGET_HEIGHT = 145
_LOG_BOX_MIN_HEIGHT = 0


class _LaunchSettingsDialog(QtWidgets.QDialog):
    """Dialog for editing all launch parameters grouped by node."""

    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)
        self.setWindowTitle("启动参数设置")
        self.setMinimumSize(620, 520)
        self._inputs = {}
        self._build_ui(current_params)

    def _build_ui(self, current_params):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("启动参数配置")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #2b2b2b;")
        title_row.addWidget(title, 1)
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.setFixedWidth(54)
        close_btn.clicked.connect(self.reject)
        title_row.addWidget(close_btn)
        main_layout.addLayout(title_row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(0, 0, 6, 0)

        # Group params by node
        groups = {}
        for name, default, desc, group in LAUNCH_PARAMS:
            groups.setdefault(group, []).append((name, default, desc))

        group_colors = {
            "中心节点": "#1a73e8",
            "RSI 节点": "#b15e00",
            "UART 节点": "#1b6e3c",
            "系统管理器": "#7b1fa2",
            "延迟监控": "#0f766e",
        }

        for group_name in _LAUNCH_GROUPS_ORDER:
            if group_name not in groups:
                continue
            group_box = QtWidgets.QGroupBox(group_name)
            color = group_colors.get(group_name, "#333333")
            group_box.setStyleSheet(
                "QGroupBox { font-weight: 600; margin-top: 4px;"
                " padding: 10px 8px 8px 8px;"
                " border: 1px solid #d0d0d0; border-radius: 6px;"
                " background: #ffffff; }"
                "QGroupBox::title { subcontrol-origin: margin;"
                " subcontrol-position: top left;"
                f" padding: 0 6px; color: {color}; }}"
            )
            form = QtWidgets.QFormLayout(group_box)
            form.setLabelAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
            )
            form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.AllNonFixedFieldsGrow
            )
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)

            for param_name, default_val, description in groups[group_name]:
                current_val = current_params.get(param_name, default_val)
                label = QtWidgets.QLabel(param_name)
                label.setToolTip(description)
                label.setStyleSheet("color: #444444; font-size: 12px;")

                if param_name == "kuka_status_raw":
                    widget = QtWidgets.QCheckBox(description)
                    widget.setChecked(current_val.lower() == "true")
                    form.addRow(label, widget)

                else:
                    widget = QtWidgets.QLineEdit(current_val)
                    widget.setToolTip(description)
                    widget.setStyleSheet(
                        "border: 1px solid #d0d0d0; border-radius: 4px;"
                        " padding: 4px 6px; background: #ffffff;"
                    )
                    form.addRow(label, widget)

                self._inputs[param_name] = widget

            scroll_layout.addWidget(group_box)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, 1)

        # Bottom buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        btn_reset = QtWidgets.QPushButton("恢复默认")
        btn_reset.setMinimumHeight(32)
        btn_reset.setCursor(QtCore.Qt.PointingHandCursor)
        btn_reset.setStyleSheet(
            "font-weight: 600; border: 1px solid #c0c0c0;"
            " border-radius: 5px; background: #ffffff;"
            " color: #666666; padding: 4px 16px;"
        )
        btn_reset.clicked.connect(self._reset_defaults)
        btn_ok = QtWidgets.QPushButton("确定")
        btn_ok.setMinimumHeight(32)
        btn_ok.setCursor(QtCore.Qt.PointingHandCursor)
        btn_ok.setStyleSheet(
            "font-weight: 600; border: 1px solid #1a73e8;"
            " border-radius: 5px; background: #1a73e8;"
            " color: #ffffff; padding: 4px 20px;"
        )
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_cancel.setMinimumHeight(32)
        btn_cancel.setCursor(QtCore.Qt.PointingHandCursor)
        btn_cancel.setStyleSheet(
            "font-weight: 600; border: 1px solid #c0c0c0;"
            " border-radius: 5px; background: #ffffff;"
            " color: #333333; padding: 4px 16px;"
        )
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        main_layout.addLayout(btn_row)

        self.setStyleSheet("QWidget { background: #f7f7f7; }")

    def _reset_defaults(self):
        for name, default_val in _LAUNCH_DEFAULTS.items():
            widget = self._inputs.get(name)
            if widget is None:
                continue
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(default_val.lower() == "true")
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(default_val)

    def get_params(self):
        result = {}
        for name, widget in self._inputs.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                result[name] = "true" if widget.isChecked() else "false"
            elif isinstance(widget, QtWidgets.QLineEdit):
                val = widget.text().strip()
                result[name] = val if val else _LAUNCH_DEFAULTS.get(name, "")
        return result


class _AutoScaleLabel(QtWidgets.QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self._color = "#a0a0a0"

    def set_color(self, color):
        self._color = color
        self.update_style()

    def update_style(self):
        self.setStyleSheet(f"color: {self._color}; font-weight: 900;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        rect = self.rect()
        if rect.height() > 0:
            font = self.font()
            pixel_size = max(12, min(32, int(rect.height() * 0.42)))
            font.setPixelSize(pixel_size)
            self.setFont(font)


class _AdaptiveHeightGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title="", min_height=0, target_height=0, parent=None):
        super().__init__(title, parent)
        self._min_height = int(min_height)
        self._target_height = max(self._min_height, int(target_height))
        self.setMinimumHeight(self._min_height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

    def set_target_height(self, height):
        height = max(self._min_height, int(height))
        if self._target_height != height:
            self._target_height = height
            self.updateGeometry()

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(self._target_height)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setHeight(self._min_height)
        return hint


_OFFSET_CONFIG_DIR = os.path.expanduser("~/.config/my_project")
_OFFSET_CONFIG_PATH = os.path.join(_OFFSET_CONFIG_DIR, "tool_offset.json")
_OFFSET_DEFAULTS = {
    "tool_offset_x": 0.64,
    "tool_offset_y": -1.29,
    "tool_offset_z": 0.17,
    "resin_z_print_compensation_mm": 0.0}
_NPZ_OFFSET_TOLERANCE_MM = 0.005
_NPZ_RELATED_LAUNCH_PARAMS = (
    "npz_path",
    "npz_preload_chunks",
    "queue_low",
    "queue_high",
    "traj_prefill",
    "traj_low",
    "traj_high",
    "xyzabc_decimals",
    "e_decimals",
)

_PID_PARAM_KEYS = ["kp", "ki", "kd", "max_output", "min_output", "max_integral", "min_integral"]
_PID_PARAM_LABELS = {
    "kp": "Kp", "ki": "Ki", "kd": "Kd",
    "max_output": "max_output", "min_output": "min_output",
    "max_integral": "max_integral", "min_integral": "min_integral",
}
_PID_DEFAULTS = {
    "cf": {
        "kp": 2.0,
        "ki": 0.85,
        "kd": 40.0,
        "max_output": 100.0,
        "min_output": 82.0,
        "max_integral": 95.0,
        "min_integral": 0.0},
    "resin": {
        "kp": 1.782,
        "ki": 0.6,
        "kd": 150.0,
        "max_output": 100.0,
        "min_output": 63.18,
        "max_integral": 105.3,
        "min_integral": 0.0},
}


def _load_offset_config():
    try:
        with open(_OFFSET_CONFIG_PATH, "r") as f:
            data = json.load(f)
        return {
            "tool_offset_x": float(
                data.get("tool_offset_x", _OFFSET_DEFAULTS["tool_offset_x"])
            ),
            "tool_offset_y": float(
                data.get("tool_offset_y", _OFFSET_DEFAULTS["tool_offset_y"])
            ),
            "tool_offset_z": float(
                data.get("tool_offset_z", _OFFSET_DEFAULTS["tool_offset_z"])
            ),
            "resin_z_print_compensation_mm": float(
                data.get(
                    "resin_z_print_compensation_mm",
                    _OFFSET_DEFAULTS["resin_z_print_compensation_mm"],
                )
            ),
        }
    except Exception:
        return dict(_OFFSET_DEFAULTS)


def _save_offset_config(x, y, z, resin_z_print_compensation_mm=0.0):
    os.makedirs(_OFFSET_CONFIG_DIR, exist_ok=True)
    with open(_OFFSET_CONFIG_PATH, "w") as f:
        json.dump({"tool_offset_x": x, "tool_offset_y": y, "tool_offset_z": z,
                  "resin_z_print_compensation_mm": resin_z_print_compensation_mm}, f, indent=2)


def _format_tool_offset(offset):
    return f"({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f})"


def _dialog_start_dir(current_path, default_dir):
    current_path = (current_path or "").strip()
    if current_path:
        p = Path(current_path).expanduser()
        candidate = p if p.is_dir() else p.parent
        if candidate.is_dir():
            return str(candidate)
    return default_dir


def _normalize_npz_launch_path(npz_path):
    p = Path(npz_path)
    if p.suffix.lower() != ".npz":
        return None
    stem = re.sub(r"_part\d+$", "", p.stem)
    return str(p.with_name(stem + ".npz"))


def _npz_layer_dir_from_launch_path(npz_path):
    launch_path = _normalize_npz_launch_path(npz_path)
    if not launch_path:
        return None
    p = Path(launch_path)
    if p.parent.name == p.stem:
        return str(p.parent)
    return str(p.with_suffix(""))


def _npz_preview_root_from_path(npz_path):
    launch_path = _normalize_npz_launch_path(npz_path)
    if not launch_path:
        return None
    p = Path(launch_path)
    if re.search(r"_part\d+$", Path(npz_path).stem):
        return str(Path(npz_path).parent)
    if p.parent.name == p.stem:
        return str(p.parent)
    layer_dir = p.with_suffix("")
    if layer_dir.is_dir():
        return str(layer_dir)
    if p.is_file():
        return str(p)
    return str(layer_dir)


def _resolve_npz_launch_path_from_dir(npz_dir):
    root = Path(npz_dir)
    if not root.is_dir():
        return None

    single_files = sorted(
        p for p in root.glob("*.npz")
        if not re.search(r"_part\d+$", p.stem)
    )
    if single_files:
        return str(single_files[0])

    part_files = sorted(root.glob("*_part*.npz"))
    if part_files:
        return _normalize_npz_launch_path(str(part_files[0]))

    return None


def _offset_sidecar_candidates(npz_source):
    p = Path(npz_source)
    candidates = []

    if p.is_dir():
        candidates.append(p.parent / f"{p.name}.offset.json")
        return candidates

    if p.name.endswith("_manifest.json"):
        base = p.name[:-len("_manifest.json")]
        candidates.extend([
            p.parent.parent / f"{base}.offset.json",
            p.parent / f"{base}.offset.json",
        ])
        return candidates

    if p.suffix == ".npz":
        stem = re.sub(r"_part\d+$", "", p.stem)
        candidates.extend([
            p.with_name(stem + ".offset.json"),
            p.parent.parent / f"{stem}.offset.json",
        ])
        return candidates

    candidates.append(p.with_suffix(".offset.json"))
    return candidates


def _read_npz_export_metadata(npz_source):
    for offset_file in _offset_sidecar_candidates(npz_source):
        if not offset_file.is_file():
            continue
        with open(offset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        saved_offset = data.get("tool_offset")
        offset = None
        if saved_offset and len(saved_offset) == 3:
            offset = tuple(float(v) for v in saved_offset)
        resin_z = data.get("resin_z_print_compensation_mm")
        resin_z_value = None if resin_z is None else float(resin_z)
        return offset, resin_z_value, str(offset_file)
    return None, None, None


def _read_npz_tool_offset(npz_source):
    saved_offset, _resin_z, offset_file = _read_npz_export_metadata(npz_source)
    return saved_offset, offset_file


def _format_rsi_xml_for_display(xml_text):
    text = (xml_text or "").strip()
    if not text:
        return ""
    try:
        import xml.dom.minidom
        parsed = xml.dom.minidom.parseString(text.encode("utf-8"))
        pretty = parsed.toprettyxml(indent="  ")
        lines = [line for line in pretty.splitlines() if line.strip()]
        if lines and lines[0].startswith("<?xml"):
            lines = lines[1:]
        return "\n".join(lines)
    except Exception:
        return text


class _NoWheelDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class _PanelDialog(QtWidgets.QDialog):
    """Project-styled popup dialog with an in-window title."""

    def __init__(self, title, parent=None, minimum_width=360, native_frame=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        if native_frame:
            self.setWindowFlags(
                QtCore.Qt.Dialog
                | QtCore.Qt.WindowTitleHint
                | QtCore.Qt.WindowSystemMenuHint
                | QtCore.Qt.WindowMinMaxButtonsHint
                | QtCore.Qt.WindowCloseButtonHint
            )
        else:
            self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)
        self.setModal(False)
        self.setMinimumWidth(minimum_width)
        self.setStyleSheet(
            "QDialog { background: #ffffff; border: 1px solid #c0c0c0; border-radius: 6px; }"
        )
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 12)
        self._layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: 700; color: #2b2b2b; font-size: 14px;")
        self._close_btn = QtWidgets.QPushButton("关闭")
        self._close_btn.setFixedWidth(54)
        self._close_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._close_btn.clicked.connect(self.close)
        header.addWidget(title_label, 1)
        header.addWidget(self._close_btn)
        self._layout.addLayout(header)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self._layout.addWidget(line)

    def body_layout(self):
        return self._layout


class _DecisionDialog(QtWidgets.QDialog):
    def __init__(self, title, message, buttons, default_button, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)
        self.setModal(True)
        self._result = None
        self.setMinimumWidth(460)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: 700; color: #2b2b2b; font-size: 15px;")
        layout.addWidget(title_label)

        body = QtWidgets.QLabel(message)
        body.setWordWrap(True)
        body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        body.setStyleSheet("color: #444444; font-size: 12px;")
        layout.addWidget(body)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        for result, text in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.setMinimumWidth(72)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            if result == default_button:
                btn.setDefault(True)
            btn.clicked.connect(lambda checked=False, r=result: self._finish(r))
            row.addWidget(btn)
        layout.addLayout(row)

    def _finish(self, result):
        self._result = result
        self.accept()

    def result_value(self):
        return self._result


def _show_warning(parent, title, message):
    dialog = _DecisionDialog(
        title,
        message,
        [(QtWidgets.QMessageBox.Ok, "确定")],
        QtWidgets.QMessageBox.Ok,
        parent,
    )
    dialog.exec_()
    return QtWidgets.QMessageBox.Ok


def _ask_yes_no(parent, title, message, default=QtWidgets.QMessageBox.No):
    dialog = _DecisionDialog(
        title,
        message,
        [(QtWidgets.QMessageBox.Yes, "是"), (QtWidgets.QMessageBox.No, "否")],
        default,
        parent,
    )
    dialog.exec_()
    return dialog.result_value() or default


class _ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoom_changed = QtCore.pyqtSignal(float)
    ZOOM_MIN = 0.1
    ZOOM_MAX = 20.0

    def wheelEvent(self, event):
        factor = 1.15
        if event.angleDelta().y() > 0:
            zoom = factor
        else:
            zoom = 1.0 / factor
        new_scale = self.transform().m11() * zoom
        if new_scale < self.ZOOM_MIN or new_scale > self.ZOOM_MAX:
            return
        self.scale(zoom, zoom)
        self.zoom_changed.emit(self.transform().m11())


class _LogDetailDialog(QtWidgets.QDialog):
    def __init__(self, title_text, log_text, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)
        self.setWindowTitle(title_text)
        self.resize(820, 520)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(title_text)
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #2b2b2b;")
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.setFixedWidth(54)
        btn_close.setCursor(QtCore.Qt.PointingHandCursor)
        btn_close.clicked.connect(self.accept)
        header.addWidget(title, 1)
        header.addWidget(btn_close)
        layout.addLayout(header)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self._text.setPlainText(log_text or "尚未收到日志。")
        self._text.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #1e1e1e;"
            "  color: #d4d4d4;"
            "  font-family: 'Courier New', 'Noto Mono', monospace;"
            "  font-size: 12px;"
            "  border: 1px solid #3c3c3c;"
            "  border-radius: 4px;"
            "  padding: 6px;"
            "}"
        )
        layout.addWidget(self._text, 1)


class _LayerViewerDialog(QtWidgets.QDialog):
    def __init__(self, npz_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)
        self._npz_dir = npz_dir
        self._images: list[Path] = []
        self._layer_numbers: list[int] = []
        self._index = 0
        self._zoom = 1.0

        self.setWindowTitle(f"层预览 - {Path(npz_dir).name}")
        self.resize(900, 700)

        self._scan_images()
        self._build_ui()
        self._show_current()

    def _scan_images(self):
        preview_dir = Path(self._npz_dir) / "layer_previews"
        if not preview_dir.is_dir():
            return
        pattern = re.compile(r"layer_(-?\d+)\.png$")
        files = []
        for f in sorted(preview_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                files.append((int(m.group(1)), f))
        files.sort(key=lambda x: x[0])
        self._images = [f[1] for f in files]
        self._layer_numbers = [f[0] for f in files]

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_row = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(f"层预览 - {Path(self._npz_dir).name}")
        title_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #2b2b2b;")
        btn_close_top = QtWidgets.QPushButton("关闭")
        btn_close_top.setFixedWidth(54)
        btn_close_top.clicked.connect(self.close)
        title_row.addWidget(title_label, 1)
        title_row.addWidget(btn_close_top)
        layout.addLayout(title_row)

        # Top bar: prev / label / next
        top_bar = QtWidgets.QHBoxLayout()
        self._btn_prev = QtWidgets.QPushButton("← 上一层")
        self._btn_prev.setFixedWidth(100)
        self._btn_prev.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_next = QtWidgets.QPushButton("下一层 →")
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
        self._view.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        layout.addWidget(self._view, 1)

        # Bottom bar: zoom label + reset + close
        bottom_bar = QtWidgets.QHBoxLayout()
        self._label_zoom = QtWidgets.QLabel("缩放: 100%")
        btn_reset = QtWidgets.QPushButton("重置视图")
        btn_reset.setCursor(QtCore.Qt.PointingHandCursor)
        btn_close = QtWidgets.QPushButton("关闭")
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
        self._view.zoom_changed.connect(self._on_zoom_changed)

    def _show_current(self):
        if not self._images:
            self._label_index.setText("未找到图像")
            self._btn_prev.setEnabled(False)
            self._btn_next.setEnabled(False)
            return
        total = len(self._images)
        layer_no = self._layer_numbers[self._index] if self._index < len(
            self._layer_numbers) else self._index
        self._label_index.setText(f"G-code层 {layer_no} ({self._index + 1} / {total})")
        self._btn_prev.setEnabled(self._index > 0)
        self._btn_next.setEnabled(self._index < total - 1)

        pixmap = QtGui.QPixmap(str(self._images[self._index]))
        if pixmap.isNull():
            self._label_index.setText("图像加载失败")
            return
        self._scene.clear()
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
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
        self._zoom = self._view.transform().m11()
        self._label_zoom.setText(f"缩放: {self._zoom * 100:.0f}%")

    def _on_zoom_changed(self, scale):
        self._zoom = scale
        self._label_zoom.setText(f"缩放: {scale * 100:.0f}%")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self._on_prev()
        elif event.key() == QtCore.Qt.Key_Right:
            self._on_next()
        else:
            super().keyPressEvent(event)


class _UiStatusWidget(QtWidgets.QWidget):
    status_received = QtCore.pyqtSignal(object)
    latency_received = QtCore.pyqtSignal(object)
    scale_submit = QtCore.pyqtSignal(float)
    command_submit = QtCore.pyqtSignal(str)
    uart_command_submit = QtCore.pyqtSignal(str)
    print_test_rsi_command_submit = QtCore.pyqtSignal(str)
    print_test_load_npz_submit = QtCore.pyqtSignal(str)
    print_test_initial_fiber_offset_dispatched = QtCore.pyqtSignal()
    current_correction_received = QtCore.pyqtSignal(object)
    print_test_status = QtCore.pyqtSignal(str, str)
    print_test_controls_enabled = QtCore.pyqtSignal(bool)
    export_finished = QtCore.pyqtSignal(bool, str)  # (success, message)
    export_progress = QtCore.pyqtSignal(str)  # status text
    export_progress_val = QtCore.pyqtSignal(int)  # percentage (0-100)
    rsi_xml_received = QtCore.pyqtSignal(str)  # RSI 发出 XML 日志
    uart_log_received = QtCore.pyqtSignal(str)  # UART 原始日志

    _DIAGNOSTIC_LOG_LIMIT = 200000

    def __init__(self):
        super().__init__()
        self._extrude_scale_current = 1.0
        self._current_tool_id = 0
        self._last_npz_dir = None
        self._print_test_current_correction = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._print_test_seen_correction = False
        self._print_test_busy = False
        self._print_test_target = None
        self._print_test_params = None
        self._print_test_resin_temp = None
        self._print_test_printhead_status = None
        self._head_calibration = load_head_calibration()
        self._print_test_resin_height_confirmed = False
        self._print_test_fiber_confirmed = False
        self._print_test_fiber_offset_initial_sent = False
        self._print_test_pending_initial_fiber_offset = None
        self._print_test_last_sent_fiber_offset = None
        self._print_test_waiting_for_tool = None
        self._print_test_pending_after_zero = None
        self._print_test_pending_after_tool_change = None
        self._print_test_requested_target_tool = None
        self._print_test_resin_z_floor = None
        self._uart_log_history = []
        self._diagnostic_log_history = []
        self._selected_npz_dir = None
        self._selected_npz_launch_path = None
        _ensure_default_data_dirs()
        self._build_ui()
        self.status_received.connect(self._update_ui)
        self.latency_received.connect(self._update_latency)
        self.rsi_xml_received.connect(self._on_rsi_xml)
        self.uart_log_received.connect(self._on_uart_log)
        self.current_correction_received.connect(self._on_current_correction)
        self.print_test_status.connect(self._set_print_test_status)
        self.print_test_controls_enabled.connect(self._set_print_test_controls_enabled)
        self.export_progress_val.connect(self._on_export_progress_val)

        self._align_timer = QtCore.QTimer(self)
        self._align_timer.timeout.connect(self._dynamic_align)
        self._align_timer.start(50)

    def _dynamic_align(self):
        if not hasattr(self, '_system_box'):
            return
        if self._active_mode == _MODE_PAGE_SELECT:
            return
        if not (self._rsi_log_box.isVisible() and self._uart_log_box.isVisible()):
            return
        if not (self._kuka_box.isVisible() and self._traj_box.isVisible()):
            return
        margins = self._content_layout.contentsMargins()

        def _top_in_content(widget):
            return widget.mapTo(self._mode_content_page, QtCore.QPoint(0, 0)).y()

        target_top = _top_in_content(self._uart_log_box)
        viewport = self._content_scroll.viewport() if hasattr(
            self, '_content_scroll') else self._mode_content_page
        target_bottom = viewport.height() - margins.bottom() - 1

        log_h = max(_LOG_BOX_MIN_HEIGHT, target_bottom - target_top + 1)

        def _set_box_target_height(widget, height):
            if hasattr(widget, "set_target_height"):
                widget.set_target_height(height)

        _set_box_target_height(self._rsi_log_box, log_h)
        _set_box_target_height(self._uart_log_box, log_h)

    def _build_mode_select_page(self):
        page = QtWidgets.QWidget()
        page.setObjectName("modeSelectPage")
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(24, 24, 24, 24)
        page_layout.setSpacing(20)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(24)
        self._btn_mode_test = QtWidgets.QPushButton("测试模式")
        self._btn_mode_test.setObjectName("btnModeTest")
        self._btn_mode_test.setMinimumSize(220, 120)
        self._btn_mode_test.setCursor(QtCore.Qt.PointingHandCursor)

        self._btn_mode_print = QtWidgets.QPushButton("正式打印")
        self._btn_mode_print.setObjectName("btnModePrint")
        self._btn_mode_print.setMinimumSize(220, 120)
        self._btn_mode_print.setCursor(QtCore.Qt.PointingHandCursor)

        btn_row.addStretch(1)
        btn_row.addWidget(self._btn_mode_test)
        btn_row.addWidget(self._btn_mode_print)
        btn_row.addStretch(1)

        page_layout.addStretch(1)
        page_layout.addLayout(btn_row)
        page_layout.addStretch(1)
        return page

    def _set_active_mode(self, mode):
        self._active_mode = mode
        mode_sections = {
            "test": [
                self._system_box,
                self._kuka_box,
                self._traj_box,
                self._rsi_log_box,
                self._ph_overview_box,
                self._ph_tools_box,
                self._uart_log_box,
                self._launch_box,
                self._print_test_box,
            ],
            "print": [
                self._system_box,
                self._kuka_box,
                self._traj_box,
                self._rsi_log_box,
                self._ph_overview_box,
                self._ph_tools_box,
                self._uart_log_box,
                self._export_box,
                self._launch_box,
                self._control_box,
                self._latency_box,
            ],
        }
        all_sections = {section for sections in mode_sections.values() for section in sections}
        for section in all_sections:
            section.setVisible(False)
        for section in mode_sections.get(mode, []):
            section.setVisible(True)

        if mode == _MODE_PAGE_TEST:
            self._title_label.setText("测试模式")
            self._launch_box.setTitle("启动通信")
        else:
            self._title_label.setText("正式打印")
            self._launch_box.setTitle("启动")
        self._print_progress_widget.setVisible(mode == _MODE_PAGE_PRINT)
        self._mode_stack.setCurrentWidget(self._content_scroll)

    def _show_mode_select(self):
        self._active_mode = _MODE_PAGE_SELECT
        self._mode_stack.setCurrentWidget(self._mode_select_page)

    def active_mode(self):
        return self._active_mode

    def _build_ui(self):
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._mode_stack = QtWidgets.QStackedWidget()
        root_layout.addWidget(self._mode_stack)
        self._mode_select_page = self._build_mode_select_page()
        self._mode_content_page = QtWidgets.QWidget()

        self._content_scroll = QtWidgets.QScrollArea()
        self._content_scroll.setWidgetResizable(True)
        self._content_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._content_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._content_scroll.setWidget(self._mode_content_page)

        self._mode_stack.addWidget(self._mode_select_page)
        self._mode_stack.addWidget(self._content_scroll)

        layout = QtWidgets.QGridLayout(self._mode_content_page)
        self._content_layout = layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        self._labels = {}
        value_min_width = QtWidgets.QLabel("0").fontMetrics().horizontalAdvance("0") * 5
        cf_labels = [
            ("Carbon Fiber State", "当前状态"),
            ("Carbon Fiber Fan OK", "风扇状态"),
            ("Carbon Fiber Current Temp", "当前温度"),
            ("Carbon Fiber Target Temp", "目标温度"),
        ]
        resin_labels = [
            ("Resin State", "当前状态"),
            ("Resin Fan OK", "风扇状态"),
            ("Resin Current Temp", "当前温度"),
            ("Resin Target Temp", "目标温度"),
        ]
        label_metrics = QtWidgets.QLabel("X").fontMetrics()
        cf_resin_label_titles = [title for _, title in (cf_labels + resin_labels)]
        cf_resin_label_min_width = max(
            label_metrics.horizontalAdvance(text) for text in cf_resin_label_titles
        )

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)
        self._title_label = QtWidgets.QLabel("系统控制面板")
        self._title_label.setObjectName("titleLabel")
        self._title_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._title_label.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred)
        self._btn_mode_back = QtWidgets.QPushButton("返回")
        self._btn_mode_back.setObjectName("btnModeBack")
        self._btn_mode_back.setMinimumHeight(28)
        self._btn_mode_back.setCursor(QtCore.Qt.PointingHandCursor)
        self._print_progress_widget = QtWidgets.QWidget()
        self._print_progress_widget.setObjectName("printProgressWidget")
        progress_layout = QtWidgets.QHBoxLayout(self._print_progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(6)
        self._print_progress_label = QtWidgets.QLabel("层进度 -- / --")
        self._print_progress_label.setObjectName("printProgressLabel")
        self._print_progress_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._print_progress_label.setMinimumWidth(92)
        self._print_progress_bar = QtWidgets.QProgressBar()
        self._print_progress_bar.setObjectName("printProgressBar")
        self._print_progress_bar.setRange(0, 100)
        self._print_progress_bar.setValue(0)
        self._print_progress_bar.setTextVisible(False)
        self._print_progress_bar.setFixedWidth(160)
        self._print_progress_bar.setFixedHeight(14)
        self._print_time_label = QtWidgets.QLabel("时间估计 --")
        self._print_time_label.setObjectName("printTimeLabel")
        self._print_time_label.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._print_time_label.setMinimumWidth(250)
        progress_layout.addWidget(self._print_progress_label)
        progress_layout.addWidget(self._print_progress_bar)
        progress_layout.addWidget(self._print_time_label)
        self._print_progress_widget.setVisible(False)

        header.addWidget(self._title_label, 1)
        header.addWidget(self._print_progress_widget)
        header.addWidget(self._btn_mode_back)
        layout.addLayout(header, 0, 0, 1, 3)

        def add_group(
            group_title,
            rows,
            parent_layout=None,
            object_name=None,
            add_to_layout=True,
            value_alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            label_min_width=None,
        ):
            group_box = QtWidgets.QGroupBox(group_title)
            if object_name:
                group_box.setObjectName(object_name)

            form = QtWidgets.QFormLayout(group_box)
            form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(6)
            for row_title in rows:
                if isinstance(row_title, tuple):
                    key, title = row_title
                else:
                    key, title = row_title, row_title
                label_title = QtWidgets.QLabel(title)
                label_title.setObjectName("fieldLabel")
                if label_min_width is not None:
                    label_title.setMinimumWidth(label_min_width)
                label_value = QtWidgets.QLabel("-")
                label_value.setObjectName("valueLabel")
                label_value.setAlignment(value_alignment)
                label_value.setMinimumWidth(value_min_width)
                label_value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                form.addRow(label_title, label_value)
                self._labels[key] = label_value

            if parent_layout is not None:
                parent_layout.addWidget(group_box)
            elif add_to_layout:
                layout.addWidget(group_box)
            return group_box

        col0_layout = QtWidgets.QVBoxLayout()
        col0_layout.setSpacing(6)
        col0_layout.setAlignment(QtCore.Qt.AlignTop)
        self._col0_layout = col0_layout
        col1_layout = QtWidgets.QVBoxLayout()
        col1_layout.setSpacing(6)
        col1_layout.setAlignment(QtCore.Qt.AlignTop)
        self._col1_layout = col1_layout
        col2_layout = QtWidgets.QVBoxLayout()
        col2_layout.setSpacing(6)

        system_box = QtWidgets.QGroupBox("系统状态")
        system_box.setObjectName("groupSystem")
        system_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self._system_box = system_box
        self._system_box.setFixedHeight(_SYSTEM_STATUS_TARGET_HEIGHT)
        system_layout = QtWidgets.QVBoxLayout(system_box)
        system_layout.setSpacing(0)
        system_layout.setContentsMargins(4, 12, 4, 12)

        sys_val = _AutoScaleLabel("离线")
        sys_val.setObjectName("valueLabel")
        sys_val.setAlignment(QtCore.Qt.AlignCenter)
        sys_val.setMinimumHeight(40)

        system_layout.addWidget(sys_val, 1)

        self._labels["System State"] = sys_val
        col0_layout.addWidget(system_box)

        kuka_box = QtWidgets.QGroupBox("KUKA 实时位姿")
        self._kuka_box = kuka_box
        kuka_box.setObjectName("groupKuka")
        kuka_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        kuka_layout = QtWidgets.QGridLayout(kuka_box)
        kuka_layout.setSpacing(6)
        for i, axis in enumerate(("X", "Y", "Z", "A", "B", "C")):
            axis_widget = QtWidgets.QWidget()
            axis_layout = QtWidgets.QVBoxLayout(axis_widget)
            axis_layout.setContentsMargins(2, 2, 2, 2)
            axis_layout.setSpacing(2)
            axis_label = QtWidgets.QLabel(axis)
            axis_label.setObjectName("axisLabel")
            axis_label.setProperty("axis", axis)
            axis_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            axis_value = QtWidgets.QLabel("-")
            axis_value.setObjectName("axisValue")
            axis_value.setProperty("axis", axis)
            axis_value.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            axis_value.setMinimumWidth(value_min_width)
            axis_value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            axis_layout.addWidget(axis_label)
            axis_layout.addWidget(axis_value)
            kuka_layout.addWidget(axis_widget, i // 3, i % 3)
            self._labels[f"KUKA {axis}"] = axis_value
        col0_layout.addWidget(kuka_box)

        traj_box = QtWidgets.QGroupBox("RSI 节点")
        self._traj_box = traj_box
        traj_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        traj_layout = QtWidgets.QVBoxLayout(traj_box)
        traj_layout.setSpacing(6)
        add_group("概览", [
            ("Traj Backlog", "待发送"),
            ("Next Traj Seq", "下一帧UDP发送"),
        ], parent_layout=traj_layout)
        traj_row = QtWidgets.QHBoxLayout()
        traj_row.setSpacing(8)
        add_group("当前帧UDP发送", [
            ("Traj Seq", "序号"),
            ("Traj Tool", "工具"),
            ("Traj X", "X"),
            ("Traj Y", "Y"),
            ("Traj Z", "Z"),
            ("Traj A", "A"),
            ("Traj B", "B"),
            ("Traj C", "C"),
            ("Traj E", "E"),
        ], parent_layout=traj_row)
        add_group("下一帧UDP发送", [
            ("Traj Seq (Next)", "序号"),
            ("Traj Tool (Next)", "工具"),
            ("Traj X (Next)", "X"),
            ("Traj Y (Next)", "Y"),
            ("Traj Z (Next)", "Z"),
            ("Traj A (Next)", "A"),
            ("Traj B (Next)", "B"),
            ("Traj C (Next)", "C"),
            ("Traj E (Next)", "E"),
        ], parent_layout=traj_row)
        traj_row.setStretch(0, 1)
        traj_row.setStretch(1, 1)
        traj_layout.addLayout(traj_row)
        col0_layout.addWidget(traj_box)

        # ======== RSI 日志区域 ========
        rsi_log_box = _AdaptiveHeightGroupBox("RSI 日志", _LOG_BOX_MIN_HEIGHT, _LOG_BOX_MIN_HEIGHT)
        rsi_log_box.setObjectName("groupRsiLog")
        rsi_log_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        rsi_log_layout = QtWidgets.QVBoxLayout(rsi_log_box)
        rsi_log_layout.setContentsMargins(4, 8, 4, 4)
        rsi_log_layout.setSpacing(2)

        rsi_log_header = QtWidgets.QHBoxLayout()
        rsi_log_header.setSpacing(6)
        self._rsi_log_summary = QtWidgets.QLabel("等待 XML")
        self._rsi_log_summary.setObjectName("fieldLabel")
        self._btn_rsi_log_detail = QtWidgets.QPushButton("放大查看")
        self._btn_rsi_log_detail.setObjectName("btnRsiLogDetail")
        self._btn_rsi_log_detail.setMinimumHeight(24)
        self._btn_rsi_log_detail.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_rsi_log_detail.setEnabled(False)
        rsi_log_header.addWidget(self._rsi_log_summary, 1)
        rsi_log_header.addWidget(self._btn_rsi_log_detail)
        rsi_log_layout.addLayout(rsi_log_header)

        self._rsi_log_text = QtWidgets.QPlainTextEdit()
        self._rsi_log_text.setReadOnly(True)
        self._rsi_log_text.setMaximumBlockCount(80)
        self._rsi_log_text.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self._rsi_log_text.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #1e1e1e;"
            "  color: #d4d4d4;"
            "  font-family: 'Courier New', 'Noto Mono', monospace;"
            "  font-size: 11px;"
            "  border: 1px solid #3c3c3c;"
            "  border-radius: 4px;"
            "  padding: 4px;"
            "}"
        )
        rsi_log_layout.addWidget(self._rsi_log_text)

        self._rsi_log_text.setMinimumHeight(0)

        self._rsi_log_last_xml = ""
        self._rsi_log_latest_display = ""
        self._rsi_log_dup_count = 0
        col0_layout.addWidget(rsi_log_box)
        self._rsi_log_box = rsi_log_box

        # ======== Column 1: Printhead ========

        ph_overview_box = QtWidgets.QGroupBox("Uart 节点")
        ph_overview_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum)
        ph_overview_layout = QtWidgets.QVBoxLayout(ph_overview_box)
        ph_overview_layout.setSpacing(6)

        events_summary_box = QtWidgets.QGroupBox("事件概览")
        events_summary_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum)
        events_summary_layout = QtWidgets.QHBoxLayout(events_summary_box)
        events_summary_layout.setSpacing(12)

        lbl1 = QtWidgets.QLabel("下一序号")
        lbl1.setObjectName("fieldLabel")
        val1 = QtWidgets.QLabel("-")
        val1.setObjectName("valueLabel")
        val1.setMinimumWidth(value_min_width)
        val1.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        events_summary_layout.addWidget(lbl1)
        events_summary_layout.addWidget(val1)

        lbl2 = QtWidgets.QLabel("待处理")
        lbl2.setObjectName("fieldLabel")
        val2 = QtWidgets.QLabel("-")
        val2.setObjectName("valueLabel")
        val2.setMinimumWidth(value_min_width)
        val2.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        events_summary_layout.addStretch(1)

        events_summary_layout.addWidget(lbl2)
        events_summary_layout.addWidget(val2)
        events_summary_layout.addStretch(1)
        ph_overview_layout.addWidget(events_summary_box)

        self._labels["Next Event Seq"] = val1
        self._labels["Events Pending"] = val2

        events_row = QtWidgets.QHBoxLayout()
        events_row.setSpacing(8)
        add_group("当前事件", [
            ("Event Type", "类型"),
            ("Event Payload", "载荷"),
            ("Event Src Line", "源码行"),
            ("Event Trigger Seq", "触发序号"),
        ], parent_layout=events_row)
        add_group("下一事件", [
            ("Event Type (Next)", "类型"),
            ("Event Payload (Next)", "载荷"),
            ("Event Src Line (Next)", "源码行"),
            ("Event Trigger Seq (Next)", "触发序号"),
        ], parent_layout=events_row)
        events_row.setStretch(0, 1)
        events_row.setStretch(1, 1)
        ph_overview_layout.addLayout(events_row)

        col1_layout.addWidget(ph_overview_box)
        self._ph_overview_box = ph_overview_box

        ph_tools_box = QtWidgets.QGroupBox("工具管理")
        ph_tools_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred)
        ph_tools_layout = QtWidgets.QVBoxLayout(ph_tools_box)
        ph_tools_layout.setSpacing(8)

        tool_row = QtWidgets.QHBoxLayout()
        tool_row.setSpacing(8)

        cur_tool_title = QtWidgets.QLabel("当前工具")
        cur_tool_title.setObjectName("fieldLabel")
        tool_row.addWidget(cur_tool_title)
        self._current_tool_value = QtWidgets.QLabel("-")
        self._current_tool_value.setObjectName("valueLabel")
        tool_row.addWidget(self._current_tool_value)

        sep = QtWidgets.QLabel("│")
        sep.setStyleSheet("color: #5a5a5a; font-size: 16px;")
        sep.setAlignment(QtCore.Qt.AlignCenter)
        tool_row.addWidget(sep)

        switch_title = QtWidgets.QLabel("切换工具")
        switch_title.setObjectName("fieldLabel")
        tool_row.addWidget(switch_title)
        self._btn_tool_cf = QtWidgets.QPushButton("碳纤维")
        self._btn_tool_cf.setObjectName("btnToolCF")
        self._btn_tool_cf.setMinimumHeight(24)
        self._btn_tool_cf.setMaximumHeight(24)
        self._btn_tool_cf.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_tool_resin = QtWidgets.QPushButton("树脂")
        self._btn_tool_resin.setObjectName("btnToolResin")
        self._btn_tool_resin.setMinimumHeight(24)
        self._btn_tool_resin.setMaximumHeight(24)
        self._btn_tool_resin.setCursor(QtCore.Qt.PointingHandCursor)
        tool_row.addWidget(self._btn_tool_cf)
        tool_row.addWidget(self._btn_tool_resin)
        tool_row.addStretch(1)

        ph_tools_layout.addLayout(tool_row)

        extrude_row_widget = QtWidgets.QWidget()
        extrude_row_widget.setFixedHeight(26)
        extrude_inner = QtWidgets.QHBoxLayout(extrude_row_widget)
        extrude_inner.setContentsMargins(0, 0, 0, 0)
        extrude_inner.setSpacing(10)
        extrude_cur_label = QtWidgets.QLabel("当前挤出倍率")
        extrude_cur_label.setObjectName("fieldLabel")
        self._extrude_scale_value = QtWidgets.QLabel("1.000")
        self._extrude_scale_value.setObjectName("valueLabel")
        self._extrude_scale_value.setMinimumWidth(value_min_width)
        extrude_set_label = QtWidgets.QLabel("设置")
        extrude_set_label.setObjectName("fieldLabel")
        self._extrude_scale_input = QtWidgets.QLineEdit()
        self._extrude_scale_input.setPlaceholderText("1.0")
        self._extrude_scale_input.setMaximumWidth(80)
        validator = QtGui.QDoubleValidator(0.001, 1000.0, 3, self._extrude_scale_input)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self._extrude_scale_input.setValidator(validator)
        self._extrude_scale_apply = QtWidgets.QPushButton("应用")
        self._extrude_scale_apply.setObjectName("btnTempApply_extrude")
        self._extrude_scale_apply.setCursor(QtCore.Qt.PointingHandCursor)
        self._extrude_scale_status = QtWidgets.QLabel("-")
        self._extrude_scale_status.setObjectName("valueLabel")
        extrude_inner.addWidget(extrude_cur_label)
        extrude_inner.addWidget(self._extrude_scale_value)
        extrude_inner.addSpacing(8)
        extrude_inner.addWidget(extrude_set_label)
        extrude_inner.addWidget(self._extrude_scale_input)
        extrude_inner.addWidget(self._extrude_scale_apply)
        extrude_inner.addSpacing(8)
        extrude_inner.addWidget(self._extrude_scale_status, 1)
        ph_tools_layout.addWidget(extrude_row_widget)

        head_panels_row = QtWidgets.QHBoxLayout()
        head_panels_row.setSpacing(12)

        for head_id, head_name in (("cf", "碳纤维"), ("resin", "树脂")):
            master_panel = QtWidgets.QGroupBox(head_name)
            master_panel.setObjectName(f"groupMaster{head_id}")
            master_panel.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Maximum)
            master_layout = QtWidgets.QVBoxLayout(master_panel)
            master_layout.setSpacing(6)

            status_box = add_group(
                "概况",
                cf_labels if head_id == "cf" else resin_labels,
                add_to_layout=False,
                label_min_width=cf_resin_label_min_width,
                value_alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            master_layout.addWidget(status_box)

            ctrl_box = QtWidgets.QGroupBox("控制")
            ctrl_layout = QtWidgets.QVBoxLayout(ctrl_box)
            ctrl_layout.setSpacing(6)

            fan_row = QtWidgets.QHBoxLayout()
            fan_row.setSpacing(8)
            fan_label = QtWidgets.QLabel("风扇")
            fan_label.setObjectName("fieldLabel")
            fan_label.setMinimumWidth(30)
            btn_fan_on = QtWidgets.QPushButton("开")
            btn_fan_on.setObjectName(f"btnFanOn_{head_id}")
            btn_fan_on.setMinimumHeight(24)
            btn_fan_on.setMaximumHeight(24)
            btn_fan_on.setCursor(QtCore.Qt.PointingHandCursor)
            btn_fan_off = QtWidgets.QPushButton("关")
            btn_fan_off.setObjectName(f"btnFanOff_{head_id}")
            btn_fan_off.setMinimumHeight(24)
            btn_fan_off.setMaximumHeight(24)
            btn_fan_off.setCursor(QtCore.Qt.PointingHandCursor)
            fan_row.addWidget(fan_label)
            fan_row.addWidget(btn_fan_on)
            fan_row.addWidget(btn_fan_off)
            ctrl_layout.addLayout(fan_row)

            temp_row = QtWidgets.QHBoxLayout()
            temp_row.setSpacing(8)
            temp_label = QtWidgets.QLabel("温度")
            temp_label.setObjectName("fieldLabel")
            temp_label.setMinimumWidth(30)
            temp_input = QtWidgets.QLineEdit()
            temp_input.setPlaceholderText("°C")
            temp_input.setMaximumWidth(80)
            temp_validator = QtGui.QDoubleValidator(0.0, 500.0, 1, temp_input)
            temp_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            temp_input.setValidator(temp_validator)
            btn_temp_apply = QtWidgets.QPushButton("设定")
            btn_temp_apply.setObjectName(f"btnTempApply_{head_id}")
            btn_temp_apply.setMinimumHeight(24)
            btn_temp_apply.setMaximumHeight(24)
            btn_temp_apply.setCursor(QtCore.Qt.PointingHandCursor)
            temp_row.addWidget(temp_label)
            temp_row.addWidget(temp_input, 1)
            temp_row.addWidget(btn_temp_apply)
            ctrl_layout.addLayout(temp_row)

            # ---- PID 参数弹窗 ----
            pid_toggle = QtWidgets.QPushButton("PID 参数")
            pid_toggle.setObjectName(f"btnPidToggle_{head_id}")
            pid_toggle.setMinimumHeight(28)
            pid_toggle.setCursor(QtCore.Qt.PointingHandCursor)
            ctrl_layout.addWidget(pid_toggle)

            pid_container = _PanelDialog(f"{head_name} PID 参数", self, 420)
            pid_container_layout = pid_container.body_layout()

            pid_form = QtWidgets.QFormLayout()
            pid_form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            pid_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            pid_form.setHorizontalSpacing(12)
            pid_form.setVerticalSpacing(6)

            pid_inputs = {}
            pid_defaults = _PID_DEFAULTS[head_id]
            for param_key in _PID_PARAM_KEYS:
                lbl = QtWidgets.QLabel(_PID_PARAM_LABELS[param_key])
                lbl.setStyleSheet("color: #666666;")
                inp = QtWidgets.QLineEdit(f"{pid_defaults[param_key]:.2f}")
                inp.setMinimumWidth(100)
                inp.setStyleSheet(
                    "border: 1px solid #d0d0d0; border-radius: 4px;"
                    " padding: 4px 6px; background: #ffffff;"
                )
                pid_validator = QtGui.QDoubleValidator(-1e6, 1e6, 4, inp)
                pid_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
                inp.setValidator(pid_validator)
                pid_form.addRow(lbl, inp)
                pid_inputs[param_key] = inp

            pid_container_layout.addLayout(pid_form)

            pid_btn_row = QtWidgets.QHBoxLayout()
            pid_btn_row.setSpacing(8)
            btn_pid_apply = QtWidgets.QPushButton("应用")
            btn_pid_apply.setObjectName(f"btnTempApply_pid_{head_id}")
            btn_pid_apply.setMinimumHeight(24)
            btn_pid_apply.setMaximumHeight(24)
            btn_pid_apply.setCursor(QtCore.Qt.PointingHandCursor)
            btn_pid_apply.setStyleSheet(
                "font-weight: 600; font-size: 12px;"
                " border: 1px solid #1a73e8; border-radius: 5px;"
                " background: #ffffff; color: #1a73e8; padding: 4px 14px;"
            )
            btn_pid_restore = QtWidgets.QPushButton("恢复默认")
            btn_pid_restore.setObjectName(f"btnPidRestore_{head_id}")
            btn_pid_restore.setMinimumHeight(24)
            btn_pid_restore.setMaximumHeight(24)
            btn_pid_restore.setCursor(QtCore.Qt.PointingHandCursor)
            btn_pid_restore.setStyleSheet(
                "font-weight: 600; font-size: 12px;"
                " border: 1px solid #c0c0c0; border-radius: 5px;"
                " background: #ffffff; color: #666666; padding: 4px 14px;"
            )
            pid_status = QtWidgets.QLabel("-")
            pid_status.setStyleSheet("color: #666666;")
            pid_btn_row.addWidget(btn_pid_apply)
            pid_btn_row.addWidget(btn_pid_restore)
            pid_btn_row.addWidget(pid_status, 1)
            pid_container_layout.addLayout(pid_btn_row)

            def _make_pid_show(container):
                def _show():
                    container.adjustSize()
                    container.show()
                    container.raise_()
                    container.activateWindow()
                return _show
            pid_toggle.clicked.connect(_make_pid_show(pid_container))

            setattr(self, f"_pid_inputs_{head_id}", pid_inputs)
            setattr(self, f"_pid_status_{head_id}", pid_status)
            setattr(self, f"_btn_pid_apply_{head_id}", btn_pid_apply)
            setattr(self, f"_btn_pid_restore_{head_id}", btn_pid_restore)

            master_layout.addWidget(ctrl_box)
            head_panels_row.addWidget(master_panel)

            setattr(self, f"_btn_fan_on_{head_id}", btn_fan_on)
            setattr(self, f"_btn_fan_off_{head_id}", btn_fan_off)
            setattr(self, f"_temp_input_{head_id}", temp_input)
            setattr(self, f"_btn_temp_apply_{head_id}", btn_temp_apply)

        ph_tools_layout.addLayout(head_panels_row)
        col1_layout.addWidget(ph_tools_box)
        self._ph_tools_box = ph_tools_box

        # ======== UART 日志区域 ========
        uart_log_box = _AdaptiveHeightGroupBox("UART 日志", _LOG_BOX_MIN_HEIGHT, _LOG_BOX_MIN_HEIGHT)
        uart_log_box.setObjectName("groupUartLog")
        uart_log_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        uart_log_layout = QtWidgets.QVBoxLayout(uart_log_box)
        uart_log_layout.setContentsMargins(4, 8, 4, 4)
        uart_log_layout.setSpacing(2)

        uart_log_header = QtWidgets.QHBoxLayout()
        uart_log_header.setSpacing(6)
        self._uart_log_summary = QtWidgets.QLabel("等待日志")
        self._uart_log_summary.setObjectName("fieldLabel")
        self._btn_uart_log_detail = QtWidgets.QPushButton("放大查看")
        self._btn_uart_log_detail.setObjectName("btnUartLogDetail")
        self._btn_uart_log_detail.setMinimumHeight(24)
        self._btn_uart_log_detail.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_uart_log_detail.setEnabled(False)
        uart_log_header.addWidget(self._uart_log_summary, 1)
        uart_log_header.addWidget(self._btn_uart_log_detail)
        uart_log_layout.addLayout(uart_log_header)

        self._uart_log_text = QtWidgets.QPlainTextEdit()
        self._uart_log_text.setReadOnly(True)
        self._uart_log_text.setMaximumBlockCount(80)
        self._uart_log_text.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self._uart_log_text.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #1e1e1e;"
            "  color: #d4d4d4;"
            "  font-family: 'Courier New', 'Noto Mono', monospace;"
            "  font-size: 11px;"
            "  border: 1px solid #3c3c3c;"
            "  border-radius: 4px;"
            "  padding: 4px;"
            "}"
        )
        uart_log_layout.addWidget(self._uart_log_text)
        self._uart_log_text.setMinimumHeight(0)
        self._uart_log_latest_display = ""

        col1_layout.addWidget(uart_log_box)
        self._uart_log_box = uart_log_box

        # ======== Print Control 区域 ========
        control_box = QtWidgets.QGroupBox("打印控制")
        control_box.setObjectName("groupControl")
        control_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        control_layout = QtWidgets.QVBoxLayout(control_box)
        control_layout.setSpacing(8)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)

        self._btn_pause = QtWidgets.QPushButton("暂停")
        self._btn_pause.setObjectName("btnPause")
        self._btn_pause.setMinimumHeight(36)
        self._btn_pause.setCursor(QtCore.Qt.PointingHandCursor)

        self._btn_resume = QtWidgets.QPushButton("继续")
        self._btn_resume.setObjectName("btnResume")
        self._btn_resume.setMinimumHeight(36)
        self._btn_resume.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_resume.setEnabled(False)

        self._btn_stop = QtWidgets.QPushButton("停止")
        self._btn_stop.setObjectName("btnStop")
        self._btn_stop.setMinimumHeight(36)
        self._btn_stop.setCursor(QtCore.Qt.PointingHandCursor)

        btn_row.addWidget(self._btn_pause)
        btn_row.addWidget(self._btn_resume)
        btn_row.addWidget(self._btn_stop)
        control_layout.addLayout(btn_row)

        latency_box = QtWidgets.QGroupBox("延迟匹配")
        latency_box.setObjectName("groupLatency")
        latency_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        latency_layout = QtWidgets.QVBoxLayout(latency_box)
        latency_layout.setSpacing(8)

        latency_row = QtWidgets.QHBoxLayout()
        latency_row.setSpacing(8)
        self._latency_link_labels = {}
        for key, title_text in (
            ("linux_mcu", "Linux -> MCU"),
            ("linux_robot", "Linux -> 机械臂"),
            ("mcu_robot", "MCU -> 机械臂"),
        ):
            panel = QtWidgets.QFrame()
            panel.setObjectName("latencyPanel")
            panel_layout = QtWidgets.QVBoxLayout(panel)
            panel_layout.setContentsMargins(6, 4, 6, 4)
            panel_layout.setSpacing(2)

            title_label = QtWidgets.QLabel(title_text)
            title_label.setObjectName("fieldLabel")
            value_label = QtWidgets.QLabel("-")
            value_label.setObjectName("valueLabel")
            value_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            p95_label = QtWidgets.QLabel("P95 -")
            p95_label.setObjectName("fieldLabel")

            panel_layout.addWidget(title_label)
            panel_layout.addWidget(value_label)
            panel_layout.addWidget(p95_label)
            latency_row.addWidget(panel, 1)
            self._latency_link_labels[key] = (value_label, p95_label)
        latency_layout.addLayout(latency_row)

        self._latency_summary = QtWidgets.QLabel("挤出相对机械臂：-")
        self._latency_summary.setObjectName("fieldLabel")
        self._latency_summary.setWordWrap(True)
        latency_layout.addWidget(self._latency_summary)

        self._latency_diag_toggle = QtWidgets.QPushButton("详细延迟")
        self._latency_diag_toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self._latency_diag_toggle.clicked.connect(self._show_latency_detail)
        latency_layout.addWidget(self._latency_diag_toggle)

        self._latency_diag_widget = _PanelDialog("详细延迟", self, 440)
        diag_grid = QtWidgets.QGridLayout()
        diag_grid.setContentsMargins(0, 0, 0, 0)
        diag_grid.setHorizontalSpacing(8)
        diag_grid.setVerticalSpacing(3)
        self._latency_diag_labels = {}
        latency_diag_rows = [
            ("linux_mcu_p99", "Linux-MCU P99"),
            ("linux_robot_p99", "Linux-机械臂 P99"),
            ("mcu_robot_avg", "MCU-机械臂平均"),
            ("mcu_robot_p99", "MCU-机械臂 P99 |偏移|"),
            ("robot_seq", "机械臂估计 Seq"),
            ("seqs", "Seq"),
            ("match_error", "匹配误差"),
            ("uncertainty", "不确定度"),
            ("warn", "告警"),
        ]
        for idx, (key, title_text) in enumerate(latency_diag_rows):
            title_label = QtWidgets.QLabel(title_text)
            title_label.setObjectName("fieldLabel")
            value_label = QtWidgets.QLabel("-")
            value_label.setObjectName("fieldLabel")
            value_label.setWordWrap(True)
            row = idx
            diag_grid.addWidget(title_label, row, 0)
            diag_grid.addWidget(value_label, row, 1)
            self._latency_diag_labels[key] = value_label
        self._latency_diag_widget.body_layout().addLayout(diag_grid)

        # Wait to add control_box until after launch_box

        # ======== GCode Export 区域 ========
        export_box = QtWidgets.QGroupBox("GCode 导出")
        export_box.setObjectName("groupExport")
        export_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        export_layout = QtWidgets.QVBoxLayout(export_box)
        export_layout.setSpacing(6)

        # Resin Z print compensation and fiber head Z offset
        resin_z_subtitle = QtWidgets.QLabel("树脂 Z 打印补偿 / 纤维 Z 偏置")
        resin_z_subtitle.setStyleSheet(
            "font-weight: bold; color: #1a73e8; font-size: 12px; margin-top: 2px;"
        )
        export_layout.addWidget(resin_z_subtitle)

        resin_z_desc = QtWidgets.QLabel(
            "树脂 Z 是整体 TCP/打印空间补偿；纤维 XYZ 是纤维头相对树脂头的物理偏置。"
            "切换到纤维头后，机械臂会按纤维偏置同号移动来补偿头间差异。"
        )
        offset_help = (
            "树脂 Z：整体 TCP/打印空间 Z 补偿，所有工具都会叠加。\n"
            "纤维 X/Y/Z：纤维头相对树脂头的物理偏置。填入正值表示纤维头相对树脂头在该轴正方向偏移；填入负值表示在负方向偏移。\n"
            "导出和切换到纤维头时，机械臂会按填入值同号移动补偿：例如纤维 Z=+6mm 表示纤维头物理上比树脂头低 6mm，切到纤维头后机械臂 Z 会上移 6mm，使纤维头末端回到同一打印空间高度。"
        )
        resin_z_subtitle.setToolTip(offset_help)
        resin_z_desc.setObjectName("fieldLabel")
        resin_z_desc.setWordWrap(True)
        resin_z_desc.setToolTip(offset_help)
        export_layout.addWidget(resin_z_desc)

        offset_cfg = _load_offset_config()
        resin_z_default = offset_cfg["resin_z_print_compensation_mm"]
        fiber_x_default = offset_cfg["tool_offset_x"]
        fiber_y_default = offset_cfg["tool_offset_y"]
        fiber_z_default = offset_cfg["tool_offset_z"]
        try:
            if DEFAULT_HEAD_CALIBRATION_PATH.is_file():
                json.loads(
                    DEFAULT_HEAD_CALIBRATION_PATH.read_text(encoding="utf-8")
                )
                head_calibration = load_head_calibration()
                self._head_calibration = head_calibration
                resin_z_default = float(
                    head_calibration.resin_z_print_compensation_mm
                )
                fiber_x_default = float(
                    head_calibration.fiber_x_print_compensation_mm
                )
                fiber_y_default = float(
                    head_calibration.fiber_y_print_compensation_mm
                )
                fiber_z_default = float(
                    head_calibration.fiber_z_print_compensation_mm
                )
        except Exception:
            resin_z_default = offset_cfg["resin_z_print_compensation_mm"]
            fiber_x_default = offset_cfg["tool_offset_x"]
            fiber_y_default = offset_cfg["tool_offset_y"]
            fiber_z_default = offset_cfg["tool_offset_z"]

        resin_z_row = QtWidgets.QHBoxLayout()
        resin_z_row.setSpacing(8)

        resin_z_w = QtWidgets.QWidget()
        resin_z_lay = QtWidgets.QVBoxLayout(resin_z_w)
        resin_z_lay.setContentsMargins(0, 0, 0, 0)
        resin_z_lay.setSpacing(2)
        resin_z_label = QtWidgets.QLabel("树脂 Z")
        resin_z_label.setObjectName("fieldLabel")
        resin_z_label.setAlignment(QtCore.Qt.AlignCenter)
        resin_z_tip = (
            "整体 TCP/打印空间 Z 补偿。\n"
            "正值：整体打印空间向机器人 Z 正方向偏移；负值：整体打印空间向机器人 Z 负方向偏移。\n"
            "该值会叠加到树脂头和纤维头的导出轨迹上。"
        )
        resin_z_label.setToolTip(resin_z_tip)
        self._resin_z_print_comp_spin = _NoWheelDoubleSpinBox()
        self._resin_z_print_comp_spin.setRange(-200.0, 200.0)
        self._resin_z_print_comp_spin.setDecimals(2)
        self._resin_z_print_comp_spin.setSingleStep(0.1)
        self._resin_z_print_comp_spin.setValue(resin_z_default)
        self._resin_z_print_comp_spin.setMinimumHeight(28)
        self._resin_z_print_comp_spin.setToolTip(resin_z_tip)
        self._resin_z_print_comp_spin.valueChanged.connect(self._on_offset_changed)
        resin_z_lay.addWidget(resin_z_label)
        resin_z_lay.addWidget(self._resin_z_print_comp_spin)
        resin_z_row.addWidget(resin_z_w)

        fiber_z_w = QtWidgets.QWidget()
        fiber_z_lay = QtWidgets.QVBoxLayout(fiber_z_w)
        fiber_z_lay.setContentsMargins(0, 0, 0, 0)
        fiber_z_lay.setSpacing(2)
        fiber_z_label = QtWidgets.QLabel("纤维 Z 偏置")
        fiber_z_label.setObjectName("fieldLabel")
        fiber_z_label.setAlignment(QtCore.Qt.AlignCenter)
        fiber_z_tip = (
            "纤维头相对树脂头的 Z 向物理偏置。\n"
            "正值：纤维头相对树脂头在物理上更低/需要机械臂 Z 正向上移补偿；负值：纤维头相对树脂头更高/切换后机械臂 Z 会向下补偿。\n"
            "导出时纤维轨迹 Z = 源路径 Z + 树脂整体 Z 补偿 + 纤维 Z 偏置。"
        )
        fiber_z_label.setToolTip(fiber_z_tip)
        self._fiber_z_print_comp_spin = _NoWheelDoubleSpinBox()
        self._fiber_z_print_comp_spin.setRange(-200.0, 200.0)
        self._fiber_z_print_comp_spin.setDecimals(2)
        self._fiber_z_print_comp_spin.setSingleStep(0.1)
        self._fiber_z_print_comp_spin.setValue(fiber_z_default)
        self._fiber_z_print_comp_spin.setMinimumHeight(28)
        self._fiber_z_print_comp_spin.setToolTip(fiber_z_tip)
        self._fiber_z_print_comp_spin.valueChanged.connect(self._on_offset_changed)
        fiber_z_lay.addWidget(fiber_z_label)
        fiber_z_lay.addWidget(self._fiber_z_print_comp_spin)
        resin_z_row.addWidget(fiber_z_w)
        resin_z_row.addStretch(1)
        export_layout.addLayout(resin_z_row)

        # Subtitle: Fiber XY Offset
        offset_subtitle = QtWidgets.QLabel("纤维头 XY 偏置")
        offset_subtitle.setStyleSheet(
            "font-weight: bold; color: #1a73e8; font-size: 12px; margin-top: 2px;")
        xy_offset_tip = (
            "纤维头相对树脂头的 X/Y 物理偏置。\n"
            "X 正值：纤维头相对树脂头在机器人 X 正方向偏移，切到纤维头后机械臂 X 同号移动补偿。\n"
            "Y 正值：纤维头相对树脂头在机器人 Y 正方向偏移，切到纤维头后机械臂 Y 同号移动补偿。\n"
            "负值表示对应轴负方向偏移，机械臂也按负方向补偿。"
        )
        offset_subtitle.setToolTip(xy_offset_tip)
        export_layout.addWidget(offset_subtitle)

        # Tool Offset Description & Input Fields (placed inside GCode Export)
        offset_desc = QtWidgets.QLabel(
            "纤维头相对树脂头的物理偏置；切换到纤维头后，"
            "机械臂按填入值同号移动来补偿头间差异。"
        )
        offset_desc.setObjectName("fieldLabel")
        offset_desc.setWordWrap(True)
        offset_desc.setToolTip(xy_offset_tip)
        export_layout.addWidget(offset_desc)

        offset_grid = QtWidgets.QHBoxLayout()
        offset_grid.setSpacing(8)
        self._offset_spins = {}
        for axis, default_val in [("X", fiber_x_default),
                                  ("Y", fiber_y_default)]:
            axis_w = QtWidgets.QWidget()
            axis_lay = QtWidgets.QVBoxLayout(axis_w)
            axis_lay.setContentsMargins(0, 0, 0, 0)
            axis_lay.setSpacing(2)
            lbl = QtWidgets.QLabel(f"{axis} (mm)")
            lbl.setObjectName("fieldLabel")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            axis_tip = (
                f"纤维头相对树脂头的 {axis} 向物理偏置。\n"
                f"正值表示纤维头相对树脂头在机器人 {axis} 正方向偏移；负值表示在 {axis} 负方向偏移。\n"
                f"导出/切换纤维头时，机械臂 {axis} 会按该值同号移动补偿。"
            )
            lbl.setToolTip(axis_tip)
            spin = _NoWheelDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.01)
            spin.setValue(default_val)
            spin.setMinimumHeight(28)
            spin.setToolTip(axis_tip)
            axis_lay.addWidget(lbl)
            axis_lay.addWidget(spin)
            offset_grid.addWidget(axis_w)
            self._offset_spins[axis] = spin
            spin.valueChanged.connect(self._on_offset_changed)
        export_layout.addLayout(offset_grid)

        self._offset_status = QtWidgets.QLabel("已加载配置。")
        self._offset_status.setObjectName("fieldLabel")
        export_layout.addWidget(self._offset_status)

        # Visual Separator Line
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        export_layout.addWidget(separator)

        # Subtitle: Source File
        gcode_subtitle = QtWidgets.QLabel("源文件")
        gcode_subtitle.setStyleSheet(
            "font-weight: bold; color: #1a73e8; font-size: 12px; margin-top: 4px;")
        export_layout.addWidget(gcode_subtitle)

        # Source file selector
        gcode_row = QtWidgets.QHBoxLayout()
        gcode_row.setSpacing(4)
        gcode_lbl = QtWidgets.QLabel("源文件")
        gcode_lbl.setObjectName("fieldLabel")
        gcode_lbl.setMinimumWidth(50)
        self._gcode_path_input = QtWidgets.QLineEdit()
        self._gcode_path_input.setPlaceholderText("可选择 GCode 或约定格式 NPZ，用于导出系统 NPZ")
        self._btn_browse_gcode = QtWidgets.QPushButton("…")
        self._btn_browse_gcode.setFixedWidth(32)
        self._btn_browse_gcode.setFixedHeight(28)
        self._btn_browse_gcode.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_browse_gcode.setObjectName("btnBrowseGcode")
        gcode_row.addWidget(gcode_lbl)
        gcode_row.addWidget(self._gcode_path_input, 1)
        gcode_row.addWidget(self._btn_browse_gcode)
        export_layout.addLayout(gcode_row)

        # NPZ output path (internal, hidden from UI)
        self._npz_out_input = QtWidgets.QLineEdit()
        self._npz_out_input.setPlaceholderText("根据源文件名自动生成")

        # Export settings popup
        planner_toggle = QtWidgets.QPushButton("导出设置")
        planner_toggle.setObjectName("btnPlannerToggle")
        planner_toggle.setMinimumHeight(28)
        planner_toggle.setCursor(QtCore.Qt.PointingHandCursor)
        export_layout.addWidget(planner_toggle)

        planner_container = _PanelDialog("导出设置", self, 640, native_frame=True)
        planner_container.setMinimumSize(640, 500)
        planner_container.resize(920, 720)
        planner_container.setSizeGripEnabled(True)
        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        settings_tabs = QtWidgets.QTabWidget()

        planner_tab = QtWidgets.QWidget()
        planner_form = QtWidgets.QFormLayout(planner_tab)
        planner_form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        planner_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        planner_form.setHorizontalSpacing(8)
        planner_form.setVerticalSpacing(4)

        _PLANNER_PARAMS = [
            ("dt", "0.004", "采样周期（s）"),
            ("default_feed_mm_s", "10.0", "默认进给速度（mm/s）"),
            ("corner_angle_deg", "10.0", "拐角角度阈值（°）"),
            ("corner_retreat_ratio", "0.2", "拐角回退比例"),
            ("density", "5", "点密度倍率"),
            ("degree", "3", "B 样条阶数"),
            ("max_fit_points_per_segment", "20000", "每段最大拟合点数"),
            ("export_sleep_ms", "0", "导出节流休眠（ms）"),
            ("export_yield_every", "0", "导出 yield 间隔"),
            ("split_by_layer_type", "false", "按层+类型拆分 NPZ"),
            ("plot_layer_xy", "true", "每层生成 XY 路径图"),
            ("plot_stride", "5", "绘图采样步长"),
            ("cut_lift_mm", "20.0", "剪切抬升距离（mm）"),
            ("cut_wait_s", "15.0", "剪切等待时间（s）"),
        ]
        self._planner_inputs = {}
        for param_name, default_val, desc in _PLANNER_PARAMS:
            lbl = QtWidgets.QLabel(param_name)
            lbl.setObjectName("fieldLabel")
            lbl.setToolTip(desc)
            if param_name in ("split_by_layer_type", "plot_layer_xy"):
                inp = QtWidgets.QCheckBox()
                inp.setChecked(default_val.lower() == "true")
                inp.setToolTip(desc)
                planner_form.addRow(lbl, inp)
            else:
                inp = QtWidgets.QLineEdit(default_val)
                inp.setToolTip(desc)
                planner_form.addRow(lbl, inp)
            self._planner_inputs[param_name] = inp
        settings_tabs.addTab(planner_tab, "规划器")

        external_tab = QtWidgets.QWidget()
        external_layout = QtWidgets.QVBoxLayout(external_tab)
        external_layout.setContentsMargins(0, 0, 0, 0)
        external_layout.setSpacing(8)

        external_defaults = {
            "resin_layer_height_mm": 0.5,
            "resin_extrusion_scale": 1.0,
            "resin_feed_mm_s": 10.0,
            "first_layer_resin_feed_mm_s": 10.0,
            "resin_temperature_c": 250.0,
            "resin_prime_length_mm": 18.0,
            "resin_prime_speed_mm_s": 15.0,
            "resin_retract_length_mm": 15.0,
            "resin_retract_speed_mm_s": 30.0,
            "resin_fan_enabled": True,
            "fiber_layer_height_mm": 0.1,
            "fiber_extrusion_scale": 1.0,
            "fiber_feed_mm_s": 10.0,
            "first_layer_fiber_feed_mm_s": 10.0,
            "fiber_start_accel_s": 2.0,
            "fiber_temperature_c": 250.0,
            "fiber_prime_length_mm": 12.0,
            "fiber_prime_speed_mm_s": 5.0,
            "fiber_retract_length_mm": 10.0,
            "fiber_retract_speed_mm_s": 5.0,
            "fiber_fan_enabled": True,
            "external_cut_lift_mm": 20.0,
            "external_cut_wait_s": 15.0,
            "travel_feed_mm_s": 10.0,
            "first_layer_travel_feed_mm_s": 10.0,
            "prime_settle_s": 0.5,
            "default_a": 0.0,
            "default_b": 0.0,
            "default_c": 0.0,
            "start_x_mm": 0.0,
            "start_y_mm": 0.0,
            "primeline_x_mm": 0.0,
            "primeline_y_mm": -10.0,
            "primeline_length_mm": 100.0,
            "corner_angle_deg": 45.0,
            "corner_retreat_ratio": 0.65,
            "spline_max_error_mm": 0.1,
            "spline_max_angle_deg": 45.0,
            "source_merge_distance_mm": 0.04,
            "corner_retreat_max_mm": 0.4,
            "corner_blend_segments": 8.0,
        }
        fixed_resin_width = 2.0
        try:
            from external_npz_preprocessor.param_config import load_print_params
            from external_npz_preprocessor.process_params import (
                RESIN_FIXED_BEAD_WIDTH_MM,
            )

            saved_params = load_print_params()
            fixed_resin_width = RESIN_FIXED_BEAD_WIDTH_MM
            external_defaults.update({
                "resin_layer_height_mm": saved_params.resin.layer_height_mm,
                "resin_extrusion_scale": saved_params.resin.extrusion_scale,
                "resin_feed_mm_s": saved_params.resin.feed_mm_s,
                "first_layer_resin_feed_mm_s": saved_params.resin.first_layer_feed_mm_s,
                "resin_temperature_c": saved_params.resin.temperature_c,
                "resin_prime_length_mm": saved_params.resin.prime_length_mm,
                "resin_prime_speed_mm_s": saved_params.resin.prime_speed_mm_s,
                "resin_retract_length_mm": saved_params.resin.retract_length_mm,
                "resin_retract_speed_mm_s": saved_params.resin.retract_speed_mm_s,
                "resin_fan_enabled": saved_params.resin.fan_enabled,
                "fiber_layer_height_mm": saved_params.fiber.layer_height_mm,
                "fiber_extrusion_scale": saved_params.fiber.extrusion_scale,
                "fiber_feed_mm_s": saved_params.fiber.feed_mm_s,
                "first_layer_fiber_feed_mm_s": saved_params.fiber.first_layer_feed_mm_s,
                "fiber_start_accel_s": saved_params.fiber.start_accel_s,
                "fiber_temperature_c": saved_params.fiber.temperature_c,
                "fiber_prime_length_mm": saved_params.fiber.prime_length_mm,
                "fiber_prime_speed_mm_s": saved_params.fiber.prime_speed_mm_s,
                "fiber_retract_length_mm": saved_params.fiber.retract_length_mm,
                "fiber_retract_speed_mm_s": saved_params.fiber.retract_speed_mm_s,
                "fiber_fan_enabled": saved_params.fiber.fan_enabled,
                "travel_feed_mm_s": saved_params.travel_feed_mm_s,
                "first_layer_travel_feed_mm_s": saved_params.first_layer_travel_feed_mm_s,
                "prime_settle_s": saved_params.prime_settle_s,
                "default_a": saved_params.default_a,
                "default_b": saved_params.default_b,
                "default_c": saved_params.default_c,
                "start_x_mm": saved_params.start_x_mm,
                "start_y_mm": saved_params.start_y_mm,
                "primeline_x_mm": saved_params.primeline_x_mm,
                "primeline_y_mm": saved_params.primeline_y_mm,
                "primeline_length_mm": saved_params.primeline_length_mm,
                "corner_angle_deg": saved_params.corner_angle_deg,
                "corner_retreat_ratio": saved_params.corner_retreat_ratio,
                "spline_max_error_mm": saved_params.spline_max_error_mm,
                "spline_max_angle_deg": saved_params.spline_max_angle_deg,
                "source_merge_distance_mm": saved_params.source_merge_distance_mm,
                "corner_retreat_max_mm": saved_params.corner_retreat_max_mm,
                "corner_blend_segments": saved_params.corner_blend_segments,
            })
        except Exception:
            pass

        external_group = QtWidgets.QGroupBox("外部 NPZ 工艺参数")
        external_group_layout = QtWidgets.QVBoxLayout(external_group)
        external_group_layout.setContentsMargins(8, 8, 8, 8)
        external_group_layout.setSpacing(8)

        def _external_spin(value, minimum=0.0, maximum=100000.0):
            spin = _NoWheelDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(minimum, maximum)
            spin.setValue(float(value))
            spin.setMinimumHeight(26)
            return spin

        def _external_param_group(title):
            group = QtWidgets.QGroupBox(title)
            grid = QtWidgets.QGridLayout(group)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)
            return group, grid

        def _add_external_rows(grid, rows):
            for row, fields in enumerate(rows):
                (
                    left_key,
                    left_label,
                    left_widget,
                    right_key,
                    right_label,
                    right_widget,
                ) = fields
                if left_key:
                    grid.addWidget(QtWidgets.QLabel(left_label), row, 0)
                    grid.addWidget(left_widget, row, 1)
                    self._external_npz_inputs[left_key] = left_widget
                if right_key:
                    grid.addWidget(QtWidgets.QLabel(right_label), row, 2)
                    grid.addWidget(right_widget, row, 3)
                    self._external_npz_inputs[right_key] = right_widget

        fixed_resin_width_label = QtWidgets.QLabel(
            f"固定树脂线宽: {fixed_resin_width:.1f} mm"
        )
        fixed_resin_width_label.setObjectName("fieldLabel")
        external_group_layout.addWidget(fixed_resin_width_label)

        self._external_npz_inputs = {}

        resin_fan = QtWidgets.QCheckBox()
        resin_fan.setChecked(bool(external_defaults["resin_fan_enabled"]))
        fiber_fan = QtWidgets.QCheckBox()
        fiber_fan.setChecked(bool(external_defaults["fiber_fan_enabled"]))

        first_layer_group, first_layer_grid = _external_param_group("首层速度")
        _add_external_rows(
            first_layer_grid,
            [
                (
                    "first_layer_resin_feed_mm_s",
                    "首层树脂打印速度 mm/s",
                    _external_spin(
                        external_defaults["first_layer_resin_feed_mm_s"],
                        minimum=0.001,
                    ),
                    "first_layer_fiber_feed_mm_s",
                    "首层纤维打印速度 mm/s",
                    _external_spin(
                        external_defaults["first_layer_fiber_feed_mm_s"],
                        minimum=0.001,
                    ),
                ),
                (
                    "first_layer_travel_feed_mm_s",
                    "首层空走速度 mm/s",
                    _external_spin(
                        external_defaults["first_layer_travel_feed_mm_s"],
                        minimum=0.001,
                    ),
                    None,
                    "",
                    QtWidgets.QLabel(""),
                ),
            ],
        )

        resin_group, resin_grid = _external_param_group("树脂材料")
        _add_external_rows(
            resin_grid,
            [
                (
                    "resin_layer_height_mm",
                    "层高 mm",
                    _external_spin(external_defaults["resin_layer_height_mm"]),
                    "resin_extrusion_scale",
                    "挤出倍率",
                    _external_spin(external_defaults["resin_extrusion_scale"]),
                ),
                (
                    "resin_feed_mm_s",
                    "非首层打印速度 mm/s",
                    _external_spin(external_defaults["resin_feed_mm_s"]),
                    "resin_temperature_c",
                    "温度 C",
                    _external_spin(external_defaults["resin_temperature_c"], maximum=500.0),
                ),
                (
                    "resin_prime_length_mm",
                    "预挤出长度 mm",
                    _external_spin(external_defaults["resin_prime_length_mm"]),
                    "resin_prime_speed_mm_s",
                    "预挤出速度 mm/s",
                    _external_spin(external_defaults["resin_prime_speed_mm_s"]),
                ),
                (
                    "resin_retract_length_mm",
                    "回抽长度 mm",
                    _external_spin(external_defaults["resin_retract_length_mm"]),
                    "resin_retract_speed_mm_s",
                    "回抽速度 mm/s",
                    _external_spin(external_defaults["resin_retract_speed_mm_s"]),
                ),
                (
                    "resin_fan_enabled",
                    "风扇",
                    resin_fan,
                    None,
                    "",
                    QtWidgets.QLabel(""),
                ),
            ],
        )

        fiber_group, fiber_grid = _external_param_group("纤维材料")
        _add_external_rows(
            fiber_grid,
            [
                (
                    "fiber_layer_height_mm",
                    "层高 mm",
                    _external_spin(external_defaults["fiber_layer_height_mm"]),
                    "fiber_extrusion_scale",
                    "挤出倍率",
                    _external_spin(external_defaults["fiber_extrusion_scale"]),
                ),
                (
                    "fiber_feed_mm_s",
                    "非首层打印速度 mm/s",
                    _external_spin(external_defaults["fiber_feed_mm_s"]),
                    "fiber_start_accel_s",
                    "起步加速时间 s",
                    _external_spin(
                        external_defaults["fiber_start_accel_s"],
                        minimum=0.001,
                    ),
                ),
                (
                    None,
                    "",
                    QtWidgets.QLabel(""),
                    "fiber_temperature_c",
                    "温度 C",
                    _external_spin(
                        external_defaults["fiber_temperature_c"],
                        maximum=500.0,
                    ),
                ),
                (
                    "fiber_prime_length_mm",
                    "预挤出长度 mm",
                    _external_spin(external_defaults["fiber_prime_length_mm"]),
                    "fiber_prime_speed_mm_s",
                    "预挤出速度 mm/s",
                    _external_spin(external_defaults["fiber_prime_speed_mm_s"]),
                ),
                (
                    "fiber_retract_length_mm",
                    "回抽长度 mm",
                    _external_spin(external_defaults["fiber_retract_length_mm"]),
                    "fiber_retract_speed_mm_s",
                    "回抽速度 mm/s",
                    _external_spin(external_defaults["fiber_retract_speed_mm_s"]),
                ),
                (
                    "fiber_fan_enabled",
                    "风扇",
                    fiber_fan,
                    None,
                    "",
                    QtWidgets.QLabel(""),
                ),
            ],
        )

        motion_group, motion_grid = _external_param_group("运动与坐标")
        _add_external_rows(
            motion_grid,
            [
                (
                    "prime_settle_s",
                    "预挤出稳定等待 s",
                    _external_spin(external_defaults["prime_settle_s"]),
                    None,
                    "",
                    QtWidgets.QLabel(""),
                ),
                (
                    "travel_feed_mm_s",
                    "非首层空走速度 mm/s",
                    _external_spin(external_defaults["travel_feed_mm_s"]),
                    "start_x_mm",
                    "左下角 X mm",
                    _external_spin(
                        external_defaults["start_x_mm"],
                        minimum=-100000.0,
                        maximum=100000.0,
                    ),
                ),
                (
                    "start_y_mm",
                    "左下角 Y mm",
                    _external_spin(
                        external_defaults["start_y_mm"],
                        minimum=-100000.0,
                        maximum=100000.0,
                    ),
                    "default_a",
                    "默认 A",
                    _external_spin(
                        external_defaults["default_a"], minimum=-360.0, maximum=360.0
                    ),
                ),
                (
                    "default_b",
                    "默认 B",
                    _external_spin(
                        external_defaults["default_b"], minimum=-360.0, maximum=360.0
                    ),
                    "default_c",
                    "默认 C",
                    _external_spin(
                        external_defaults["default_c"], minimum=-360.0, maximum=360.0
                    ),
                ),
            ],
        )

        primeline_group, primeline_grid = _external_param_group("起始擦料线")
        _add_external_rows(
            primeline_grid,
            [
                (
                    "primeline_x_mm",
                    "相对零件 X mm",
                    _external_spin(
                        external_defaults["primeline_x_mm"],
                        minimum=-100000.0,
                        maximum=100000.0,
                    ),
                    "primeline_y_mm",
                    "相对零件 Y mm",
                    _external_spin(
                        external_defaults["primeline_y_mm"],
                        minimum=-100000.0,
                        maximum=100000.0,
                    ),
                ),
                (
                    "primeline_length_mm",
                    "长度 mm",
                    _external_spin(
                        external_defaults["primeline_length_mm"],
                        maximum=100000.0,
                    ),
                    None,
                    "",
                    QtWidgets.QLabel(""),
                ),
            ],
        )

        smoothing_group, smoothing_grid = _external_param_group("路径平滑")
        _add_external_rows(
            smoothing_grid,
            [
                (
                    "corner_angle_deg",
                    "角点阈值 deg",
                    _external_spin(external_defaults["corner_angle_deg"], maximum=180.0),
                    "corner_retreat_ratio",
                    "角点回退比例",
                    _external_spin(external_defaults["corner_retreat_ratio"], maximum=1.0),
                ),
                (
                    "spline_max_error_mm",
                    "B样条误差上限 mm",
                    _external_spin(external_defaults["spline_max_error_mm"], maximum=10.0),
                    "spline_max_angle_deg",
                    "B样条转角上限 deg",
                    _external_spin(external_defaults["spline_max_angle_deg"], maximum=180.0),
                ),
                (
                    "source_merge_distance_mm",
                    "源短段合并 mm",
                    _external_spin(external_defaults["source_merge_distance_mm"], maximum=10.0),
                    "corner_retreat_max_mm",
                    "角点回退上限 mm",
                    _external_spin(external_defaults["corner_retreat_max_mm"], maximum=10.0),
                ),
                (
                    "corner_blend_segments",
                    "角点细分段数",
                    _external_spin(external_defaults["corner_blend_segments"], maximum=64.0),
                    None,
                    "",
                    QtWidgets.QLabel(""),
                ),
            ],
        )

        cut_group, cut_grid = _external_param_group("剪切参数")
        _add_external_rows(
            cut_grid,
            [
                (
                    "external_cut_lift_mm",
                    "剪切抬升距离 mm",
                    _external_spin(
                        external_defaults["external_cut_lift_mm"],
                        maximum=1000.0,
                    ),
                    "external_cut_wait_s",
                    "剪切等待时间 s",
                    _external_spin(
                        external_defaults["external_cut_wait_s"],
                        maximum=1000.0,
                    ),
                ),
            ],
        )

        external_group_layout.addWidget(first_layer_group)

        material_layout = QtWidgets.QVBoxLayout()
        material_layout.setSpacing(8)
        material_layout.addWidget(resin_group, 1)
        material_layout.addWidget(fiber_group, 1)
        external_group_layout.addLayout(material_layout)

        path_layout = QtWidgets.QVBoxLayout()
        path_layout.setSpacing(8)
        path_layout.addWidget(motion_group, 1)
        path_layout.addWidget(primeline_group, 1)
        path_layout.addWidget(smoothing_group, 1)
        external_group_layout.addLayout(path_layout)
        external_group_layout.addWidget(cut_group)

        external_layout.addWidget(external_group)
        self._btn_save_external_npz_params = QtWidgets.QPushButton("保存外部 NPZ 参数")
        self._btn_save_external_npz_params.setMinimumHeight(30)
        self._btn_save_external_npz_params.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_save_external_npz_params.clicked.connect(
            self._on_save_external_npz_params
        )
        external_layout.addWidget(self._btn_save_external_npz_params)
        external_layout.addStretch(1)
        settings_tabs.addTab(external_tab, "外部 NPZ")

        settings_scroll.setWidget(settings_tabs)
        planner_container.body_layout().addWidget(settings_scroll, 1)
        self._planner_container = planner_container

        def _show_planner_settings():
            planner_container.show()
            planner_container.raise_()
            planner_container.activateWindow()
        planner_toggle.clicked.connect(_show_planner_settings)

        # Export button + progress
        export_btn_row = QtWidgets.QHBoxLayout()
        export_btn_row.setSpacing(8)
        self._btn_export_npz = QtWidgets.QPushButton("导出 NPZ")
        self._btn_export_npz.setObjectName("btnExportNpz")
        self._btn_export_npz.setMinimumHeight(36)
        self._btn_export_npz.setCursor(QtCore.Qt.PointingHandCursor)
        export_btn_row.addWidget(self._btn_export_npz)
        export_layout.addLayout(export_btn_row)

        # View Layer Images button
        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(8)
        self._btn_view_layers = QtWidgets.QPushButton("查看层图像")
        self._btn_view_layers.setObjectName("btnViewLayers")
        self._btn_view_layers.setMinimumHeight(36)
        self._btn_view_layers.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_view_layers.setEnabled(False)
        view_row.addWidget(self._btn_view_layers)
        self._btn_view_vtk_paths = QtWidgets.QPushButton("三维路径预览")
        self._btn_view_vtk_paths.setObjectName("btnViewVtkPaths")
        self._btn_view_vtk_paths.setMinimumHeight(36)
        self._btn_view_vtk_paths.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_view_vtk_paths.setEnabled(False)
        view_row.addWidget(self._btn_view_vtk_paths)
        export_layout.addLayout(view_row)

        npz_dir_subtitle = QtWidgets.QLabel("选择已导出的NPZ文件")
        npz_dir_subtitle.setStyleSheet(
            "font-weight: bold; color: #1a73e8; font-size: 12px; margin-top: 4px;")
        export_layout.addWidget(npz_dir_subtitle)

        npz_dir_row = QtWidgets.QHBoxLayout()
        npz_dir_row.setSpacing(4)
        self._selected_npz_dir_input = QtWidgets.QLineEdit()
        self._selected_npz_dir_input.setReadOnly(True)
        self._selected_npz_dir_input.setPlaceholderText("可选择已导出的NPZ文件用于启动")
        self._btn_select_npz_dir = QtWidgets.QPushButton("选择")
        self._btn_select_npz_dir.setFixedWidth(48)
        self._btn_select_npz_dir.setFixedHeight(28)
        self._btn_select_npz_dir.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_clear_npz_dir = QtWidgets.QPushButton("清除")
        self._btn_clear_npz_dir.setFixedWidth(48)
        self._btn_clear_npz_dir.setFixedHeight(28)
        self._btn_clear_npz_dir.setCursor(QtCore.Qt.PointingHandCursor)
        npz_dir_row.addWidget(self._selected_npz_dir_input, 1)
        npz_dir_row.addWidget(self._btn_select_npz_dir)
        npz_dir_row.addWidget(self._btn_clear_npz_dir)
        export_layout.addLayout(npz_dir_row)

        self._export_progress = QtWidgets.QProgressBar()
        self._export_progress.setMinimumHeight(18)
        self._export_progress.setRange(0, 0)  # indeterminate
        self._export_progress.setVisible(False)
        export_layout.addWidget(self._export_progress)

        self._export_status = QtWidgets.QLabel("")
        self._export_status.setObjectName("fieldLabel")
        self._export_status.setWordWrap(True)
        export_layout.addWidget(self._export_status)

        # Connect export signals
        self._btn_browse_gcode.clicked.connect(self._on_browse_gcode)
        self._gcode_path_input.textChanged.connect(self._on_gcode_path_changed)
        self._btn_export_npz.clicked.connect(self._on_export_npz)
        self.export_finished.connect(self._on_export_finished)
        self.export_progress.connect(self._on_export_progress)
        self._btn_view_layers.clicked.connect(self._on_view_layers)
        self._btn_view_vtk_paths.clicked.connect(self._on_view_vtk_paths)
        self._btn_select_npz_dir.clicked.connect(self._on_select_npz_dir)
        self._btn_clear_npz_dir.clicked.connect(self._on_clear_npz_dir)

        # ======== Print Test 区域 ========
        print_test_box = QtWidgets.QGroupBox("打印测试")
        print_test_box.setObjectName("groupPrintTest")
        print_test_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum)
        print_test_layout = QtWidgets.QVBoxLayout(print_test_box)
        print_test_layout.setSpacing(6)

        test_form = QtWidgets.QGridLayout()
        test_form.setHorizontalSpacing(8)
        test_form.setVerticalSpacing(4)
        self._test_temp_input = QtWidgets.QLineEdit("250")
        self._test_layer_height_min_input = QtWidgets.QLineEdit("0.5")
        self._test_layer_height_max_input = QtWidgets.QLineEdit("0.5")
        self._test_speed_input = QtWidgets.QLineEdit("10.0")
        self._test_scale_min_input = QtWidgets.QLineEdit("1.0")
        self._test_scale_max_input = QtWidgets.QLineEdit("1.0")
        self._test_prime_length_input = QtWidgets.QLineEdit("18.0")
        self._test_prime_speed_input = QtWidgets.QLineEdit("15.0")
        self._test_retract_length_input = QtWidgets.QLineEdit("15.0")
        self._test_retract_speed_input = QtWidgets.QLineEdit("30.0")
        self._test_line_length_input = QtWidgets.QLineEdit("300.0")
        self._test_y_spacing_input = QtWidgets.QLineEdit("10.0")
        self._test_tool_change_safe_lift_input = QtWidgets.QLineEdit(
            f"{_TEST_TOOL_CHANGE_SAFE_LIFT_DEFAULT_MM:.1f}"
        )
        self._test_resin_z_comp_input = QtWidgets.QLineEdit(
            f"{self._head_calibration.resin_z_print_compensation_mm:.3f}"
        )
        self._test_fiber_temp_input = QtWidgets.QLineEdit("250")
        self._test_fiber_layer_height_min_input = QtWidgets.QLineEdit("0.05")
        self._test_fiber_layer_height_max_input = QtWidgets.QLineEdit("0.05")
        self._test_fiber_scale_min_input = QtWidgets.QLineEdit("1.0")
        self._test_fiber_scale_max_input = QtWidgets.QLineEdit("1.0")
        self._test_fiber_prime_length_input = QtWidgets.QLineEdit("12.0")
        self._test_fiber_prime_speed_input = QtWidgets.QLineEdit("5.0")
        self._test_fiber_retract_length_input = QtWidgets.QLineEdit("10.0")
        self._test_fiber_retract_speed_input = QtWidgets.QLineEdit("5.0")
        self._test_fiber_x_comp_input = QtWidgets.QLineEdit(
            f"{self._head_calibration.fiber_x_print_compensation_mm:.3f}"
        )
        self._test_fiber_y_comp_input = QtWidgets.QLineEdit(
            f"{self._head_calibration.fiber_y_print_compensation_mm:.3f}"
        )
        self._test_fiber_z_comp_input = QtWidgets.QLineEdit(
            f"{self._head_calibration.fiber_z_print_compensation_mm:.3f}"
        )

        range_inputs = (
            self._test_layer_height_min_input,
            self._test_layer_height_max_input,
            self._test_scale_min_input,
            self._test_scale_max_input,
            self._test_fiber_layer_height_min_input,
            self._test_fiber_layer_height_max_input,
            self._test_fiber_scale_min_input,
            self._test_fiber_scale_max_input,
        )
        positive_inputs = (
            self._test_temp_input,
            self._test_speed_input,
            self._test_line_length_input,
            self._test_y_spacing_input,
            self._test_fiber_temp_input,
            *range_inputs,
        )
        for inp in positive_inputs:
            inp.setMaximumWidth(72)
            validator = QtGui.QDoubleValidator(0.001, 1000.0, 3, inp)
            validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            inp.setValidator(validator)
        non_negative_inputs = (
            self._test_prime_length_input,
            self._test_prime_speed_input,
            self._test_retract_length_input,
            self._test_retract_speed_input,
            self._test_tool_change_safe_lift_input,
            self._test_fiber_prime_length_input,
            self._test_fiber_prime_speed_input,
            self._test_fiber_retract_length_input,
            self._test_fiber_retract_speed_input,
        )
        for inp in non_negative_inputs:
            inp.setMaximumWidth(72)
            validator = QtGui.QDoubleValidator(0.0, 1000.0, 3, inp)
            validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            inp.setValidator(validator)
        for inp in (
            self._test_resin_z_comp_input,
            self._test_fiber_x_comp_input,
            self._test_fiber_y_comp_input,
        ):
            inp.setMaximumWidth(72)
            validator = QtGui.QDoubleValidator(-1000.0, 1000.0, 3, inp)
            validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            inp.setValidator(validator)
        self._test_fiber_z_comp_input.setMaximumWidth(72)
        validator = QtGui.QDoubleValidator(
            0.001, 1000.0, 3, self._test_fiber_z_comp_input
        )
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self._test_fiber_z_comp_input.setValidator(validator)

        def _section_title(text):
            label = QtWidgets.QLabel(text)
            label.setStyleSheet(
                "font-weight: bold; color: #1a73e8; font-size: 12px; margin-top: 4px;"
            )
            return label

        def _range_widget(min_input, max_input, unit_text=None):
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            layout.addWidget(min_input)
            layout.addWidget(QtWidgets.QLabel("-"))
            layout.addWidget(max_input)
            if unit_text:
                unit_label = QtWidgets.QLabel(unit_text)
                unit_label.setObjectName("fieldLabel")
                layout.addWidget(unit_label)
            layout.addStretch(1)
            return widget

        def _value_widget(input_widget, unit_text=None):
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            layout.addWidget(input_widget)
            if unit_text:
                unit_label = QtWidgets.QLabel(unit_text)
                unit_label.setObjectName("fieldLabel")
                layout.addWidget(unit_label)
            layout.addStretch(1)
            return widget

        def _add_form_rows(form_layout, rows):
            for index, (label, widget) in enumerate(rows):
                lbl = QtWidgets.QLabel(label)
                lbl.setObjectName("fieldLabel")
                form_layout.addWidget(lbl, index, 0)
                form_layout.addWidget(widget, index, 1)
            form_layout.setColumnStretch(1, 1)

        global_form = QtWidgets.QGridLayout()
        global_form.setHorizontalSpacing(8)
        global_form.setVerticalSpacing(4)
        global_rows = (
            ("测试速度", _value_widget(self._test_speed_input, "mm/s")),
            ("测试线长度", _value_widget(self._test_line_length_input, "mm")),
            ("Y 间距", _value_widget(self._test_y_spacing_input, "mm")),
            ("换头安全抬升", _value_widget(self._test_tool_change_safe_lift_input, "mm")),
        )
        for index, (label, widget) in enumerate(global_rows):
            lbl = QtWidgets.QLabel(label)
            lbl.setObjectName("fieldLabel")
            col = index * 2
            global_form.addWidget(lbl, 0, col)
            global_form.addWidget(widget, 0, col + 1)
        global_form.setColumnStretch(len(global_rows) * 2, 1)
        print_test_layout.addWidget(_section_title("全局测试参数"))
        print_test_layout.addLayout(global_form)

        head_columns = QtWidgets.QHBoxLayout()
        head_columns.setSpacing(14)

        resin_column = QtWidgets.QWidget()
        resin_layout = QtWidgets.QVBoxLayout(resin_column)
        resin_layout.setContentsMargins(0, 0, 0, 0)
        resin_layout.setSpacing(4)
        resin_layout.addWidget(_section_title("树脂头参数"))
        resin_form = QtWidgets.QGridLayout()
        resin_form.setHorizontalSpacing(8)
        resin_form.setVerticalSpacing(4)
        _add_form_rows(resin_form, (
            ("目标温度", _value_widget(self._test_temp_input)),
            ("层高", _range_widget(
                self._test_layer_height_min_input,
                self._test_layer_height_max_input,
                "mm",
            )),
            ("挤出倍率", _range_widget(
                self._test_scale_min_input,
                self._test_scale_max_input,
            )),
            ("预挤出长度", _value_widget(self._test_prime_length_input, "mm E")),
            ("预挤出速度", _value_widget(self._test_prime_speed_input, "mm/s E")),
            ("回抽长度", _value_widget(self._test_retract_length_input, "mm E")),
            ("回抽速度", _value_widget(self._test_retract_speed_input, "mm/s E")),
        ))
        resin_layout.addLayout(resin_form)
        head_columns.addWidget(resin_column, 1, QtCore.Qt.AlignTop)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        head_columns.addWidget(separator, 0, QtCore.Qt.AlignTop)

        fiber_column = QtWidgets.QWidget()
        fiber_layout = QtWidgets.QVBoxLayout(fiber_column)
        fiber_layout.setContentsMargins(0, 0, 0, 0)
        fiber_layout.setSpacing(4)
        fiber_layout.addWidget(_section_title("纤维头参数"))
        fiber_form = QtWidgets.QGridLayout()
        fiber_form.setHorizontalSpacing(8)
        fiber_form.setVerticalSpacing(4)
        _add_form_rows(fiber_form, (
            ("目标温度", _value_widget(self._test_fiber_temp_input)),
            ("层高", _range_widget(
                self._test_fiber_layer_height_min_input,
                self._test_fiber_layer_height_max_input,
                "mm",
            )),
            ("挤出倍率", _range_widget(
                self._test_fiber_scale_min_input,
                self._test_fiber_scale_max_input,
            )),
            ("预挤出长度", _value_widget(self._test_fiber_prime_length_input, "mm E")),
            ("预挤出速度", _value_widget(self._test_fiber_prime_speed_input, "mm/s E")),
            ("回抽长度", _value_widget(self._test_fiber_retract_length_input, "mm E")),
            ("回抽速度", _value_widget(self._test_fiber_retract_speed_input, "mm/s E")),
        ))
        fiber_layout.addLayout(fiber_form)
        head_columns.addWidget(fiber_column, 1, QtCore.Qt.AlignTop)
        print_test_layout.addLayout(head_columns)

        self._test_correction_label = QtWidgets.QLabel("RSI 修正量: 未收到")
        self._test_correction_label.setObjectName("fieldLabel")
        self._test_correction_label.setWordWrap(True)
        print_test_layout.addWidget(self._test_correction_label)

        self._btn_test_prepare = QtWidgets.QPushButton("进入树脂测试准备")
        self._btn_test_prepare.setMinimumHeight(30)
        self._btn_test_prepare.setCursor(QtCore.Qt.PointingHandCursor)

        z_row_up = QtWidgets.QHBoxLayout()
        z_row_down = QtWidgets.QHBoxLayout()
        self._test_z_buttons = []
        for text_label, delta in (
            ("Z +0.1", 0.1),
            ("Z +1", 1.0),
            ("Z +5", 5.0),
        ):
            btn = QtWidgets.QPushButton(text_label)
            btn.setMinimumHeight(28)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked=False, d=delta: self._on_print_test_z(d))
            z_row_up.addWidget(btn)
            self._test_z_buttons.append(btn)
        for text_label, delta in (
            ("Z -0.1", -0.1),
            ("Z -1", -1.0),
            ("Z -5", -5.0),
        ):
            btn = QtWidgets.QPushButton(text_label)
            btn.setMinimumHeight(28)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked=False, d=delta: self._on_print_test_z(d))
            z_row_down.addWidget(btn)
            self._test_z_buttons.append(btn)

        self._btn_test_confirm_resin_height = QtWidgets.QPushButton("确认打印高度")
        self._btn_test_continue_fiber = QtWidgets.QPushButton("切换到纤维头")
        self._btn_test_print_resin = QtWidgets.QPushButton("开始树脂测试")
        self._btn_test_send_fiber_offset_nudge = QtWidgets.QPushButton("下发微调")
        self._btn_test_confirm_fiber_offset = QtWidgets.QPushButton("确认偏置并下发")
        self._btn_test_print_fiber = QtWidgets.QPushButton("直接打印纤维")
        self._btn_test_print_composite = QtWidgets.QPushButton("复合打印")
        self._btn_test_cut = QtWidgets.QPushButton("剪切")
        self._btn_test_confirm_height = QtWidgets.QPushButton("确认高度并打印测试线")
        self._btn_test_confirm_height.setVisible(False)

        resin_offset_grid = QtWidgets.QGridLayout()
        resin_offset_grid.setHorizontalSpacing(6)
        resin_offset_grid.setVerticalSpacing(4)
        resin_z_label = QtWidgets.QLabel("Z 补偿")
        resin_z_label.setObjectName("fieldLabel")
        resin_offset_grid.addWidget(resin_z_label, 0, 0)
        self._test_resin_z_comp_input.setEnabled(False)
        self._test_resin_z_comp_input.setMinimumHeight(28)
        resin_offset_grid.addWidget(
            self._test_resin_z_comp_input, 0, 1, QtCore.Qt.AlignLeft
        )
        resin_offset_grid.setColumnStretch(2, 1)

        fiber_offset_grid = QtWidgets.QGridLayout()
        fiber_offset_grid.setHorizontalSpacing(6)
        fiber_offset_grid.setVerticalSpacing(4)
        fiber_offset_rows = (
            ("X 偏置", self._test_fiber_x_comp_input),
            ("Y 偏置", self._test_fiber_y_comp_input),
            ("Z 偏置", self._test_fiber_z_comp_input),
        )
        self._fiber_offset_nudge_buttons = []
        for index, (label_text, input_widget) in enumerate(fiber_offset_rows):
            col = index * 2
            label = QtWidgets.QLabel(label_text)
            label.setObjectName("fieldLabel")
            fiber_offset_grid.addWidget(label, 0, col)
            input_widget.setEnabled(False)
            input_widget.setMinimumHeight(28)

            input_row = QtWidgets.QHBoxLayout()
            input_row.setSpacing(3)
            input_row.addWidget(input_widget, 1)
            axis = label_text[0]
            for text_label, delta in (("-", -0.1), ("+", 0.1)):
                btn = QtWidgets.QPushButton(text_label)
                btn.setFixedSize(18, 18)
                btn.setCursor(QtCore.Qt.PointingHandCursor)
                btn.setEnabled(False)
                btn.setToolTip(f"{axis} 偏置 {delta:+.1f} mm")
                btn.clicked.connect(
                    lambda _checked=False, a=axis, d=delta:
                    self._on_print_test_nudge_fiber_offset(a, d)
                )
                input_row.addWidget(btn)
                self._fiber_offset_nudge_buttons.append(btn)
            fiber_offset_grid.addLayout(input_row, 0, col + 1)
            fiber_offset_grid.setColumnStretch(col + 1, 1)

        action_columns = QtWidgets.QHBoxLayout()
        action_columns.setSpacing(14)

        def _action_column(title, buttons, extra_layouts=()):
            column = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(column)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            layout.addWidget(_section_title(title))
            for extra in extra_layouts:
                layout.addLayout(extra)
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(4)
            for index, btn in enumerate(buttons):
                btn.setMinimumHeight(28)
                btn.setCursor(QtCore.Qt.PointingHandCursor)
                btn.setEnabled(False)
                grid.addWidget(btn, index // 2, index % 2)
            layout.addLayout(grid)
            return column

        self._btn_test_prepare.setEnabled(True)
        print_test_layout.addWidget(self._btn_test_prepare)
        action_columns.addWidget(_action_column(
            "树脂头动作",
            (
                self._btn_test_confirm_resin_height,
                self._btn_test_print_resin,
            ),
            (resin_offset_grid, z_row_up, z_row_down),
        ), 2)
        action_columns.addWidget(_action_column(
            "纤维头动作",
            (
                self._btn_test_continue_fiber,
                self._btn_test_confirm_fiber_offset,
                self._btn_test_send_fiber_offset_nudge,
                self._btn_test_print_fiber,
                self._btn_test_print_composite,
                self._btn_test_cut,
            ),
            (fiber_offset_grid,),
        ), 2)
        print_test_layout.addLayout(action_columns)

        self._btn_export_uart_log = QtWidgets.QPushButton("导出诊断日志")
        self._btn_export_uart_log.setMinimumHeight(36)
        self._btn_export_uart_log.setCursor(QtCore.Qt.PointingHandCursor)

        self._test_status = QtWidgets.QLabel("未进入测试。")
        self._test_status.setObjectName("fieldLabel")
        self._test_status.setWordWrap(True)
        print_test_layout.addWidget(self._test_status)

        self._btn_test_prepare.clicked.connect(self._on_print_test_prepare)
        self._btn_test_confirm_height.clicked.connect(self._on_print_test_confirm_height)
        self._btn_test_confirm_resin_height.clicked.connect(
            self._on_print_test_confirm_resin_height
        )
        self._btn_test_continue_fiber.clicked.connect(self._on_print_test_continue_fiber)
        self._btn_test_print_resin.clicked.connect(self._on_print_test_print_resin)
        self._btn_test_send_fiber_offset_nudge.clicked.connect(
            self._on_print_test_send_fiber_offset_nudge
        )
        self._btn_test_confirm_fiber_offset.clicked.connect(
            self._on_print_test_confirm_fiber_offset
        )
        self._btn_test_print_fiber.clicked.connect(self._on_print_test_print_fiber)
        self._btn_test_print_composite.clicked.connect(self._on_print_test_print_composite)
        self._btn_test_cut.clicked.connect(self._on_print_test_cut)
        self._btn_export_uart_log.clicked.connect(self._on_export_diagnostic_log)

        # ======== Launch Control 区域 ========
        launch_box = QtWidgets.QGroupBox("启动")
        launch_box.setObjectName("groupLaunch")
        launch_box.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        launch_inner = QtWidgets.QVBoxLayout(launch_box)
        launch_inner.setSpacing(8)

        launch_btn_row = QtWidgets.QHBoxLayout()
        launch_btn_row.setSpacing(12)

        self._btn_launch_settings = QtWidgets.QPushButton("设置")
        self._btn_launch_settings.setObjectName("btnLaunchSettings")
        self._btn_launch_settings.setMinimumHeight(36)
        self._btn_launch_settings.setCursor(QtCore.Qt.PointingHandCursor)

        self._btn_launch = QtWidgets.QPushButton("启动")
        self._btn_launch.setObjectName("btnLaunch")
        self._btn_launch.setMinimumHeight(36)
        self._btn_launch.setCursor(QtCore.Qt.PointingHandCursor)

        self._btn_stop_launch = QtWidgets.QPushButton("停止节点")
        self._btn_stop_launch.setObjectName("btnStopLaunch")
        self._btn_stop_launch.setMinimumHeight(36)
        self._btn_stop_launch.setCursor(QtCore.Qt.PointingHandCursor)
        self._btn_stop_launch.setEnabled(False)

        launch_btn_row.addWidget(self._btn_launch_settings)
        launch_btn_row.addWidget(self._btn_launch)
        launch_btn_row.addWidget(self._btn_stop_launch)
        launch_inner.addLayout(launch_btn_row)

        launch_status_row = QtWidgets.QHBoxLayout()
        launch_status_row.setSpacing(8)
        launch_label = QtWidgets.QLabel("启动状态")
        launch_label.setObjectName("fieldLabel")
        self._launch_status = QtWidgets.QLabel("未启动")
        self._launch_status.setObjectName("launchStatus")
        launch_status_row.addWidget(launch_label)
        launch_status_row.addWidget(self._launch_status, 1)
        launch_inner.addLayout(launch_status_row)

        self._export_box = export_box
        self._launch_box = launch_box
        self._print_test_box = print_test_box
        self._control_box = control_box
        self._latency_box = latency_box

        # Add all boxes to col2_layout in the desired order
        col2_layout.addWidget(export_box)
        col2_layout.addWidget(launch_box)
        col2_layout.addWidget(print_test_box)
        col2_layout.addWidget(control_box)
        col2_layout.addWidget(latency_box)
        col2_layout.addStretch(1)
        col2_layout.addWidget(self._btn_export_uart_log)

        layout.addLayout(col0_layout, 1, 0)
        layout.addLayout(col1_layout, 1, 1)
        layout.addLayout(col2_layout, 1, 2)

        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 1)

        self.setStyleSheet(
            "QWidget { background: #f7f7f7; "
            "font-family: 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', "
            "'Microsoft YaHei', sans-serif; }"
            "QGroupBox {"
            "  font-weight: 600;"
            "  margin-top: 4px;"
            "  padding: 10px 8px 8px 8px;"
            "  border: 1px solid #d0d0d0;"
            "  border-radius: 6px;"
            "  background: #ffffff;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  padding: 0 6px;"
            "}"
            "QLabel#titleLabel {"
            "  font-size: 16px;"
            "  font-weight: 700;"
            "  color: #2b2b2b;"
            "}"
            "QPushButton#btnModeTest, QPushButton#btnModePrint {"
            "  font-weight: 700;"
            "  font-size: 22px;"
            "  border: 1px solid #1a73e8;"
            "  border-radius: 8px;"
            "  background: #ffffff;"
            "  color: #1a73e8;"
            "  padding: 18px 28px;"
            "}"
            "QPushButton#btnModeTest:hover, QPushButton#btnModePrint:hover {"
            "  background: #e8f0fe;"
            "  border-color: #1558b0;"
            "}"
            "QPushButton#btnModeBack {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #c0c0c0;"
            "  border-radius: 5px;"
            "  background: #ffffff;"
            "  color: #333333;"
            "  padding: 4px 14px;"
            "}"
            "QPushButton#btnModeBack:hover {"
            "  background: #f0f0f0;"
            "  border-color: #999999;"
            "}"
            "QPushButton#btnRsiLogDetail, QPushButton#btnUartLogDetail {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #c0c0c0;"
            "  border-radius: 5px;"
            "  background: #ffffff;"
            "  color: #333333;"
            "  padding: 3px 10px;"
            "}"
            "QPushButton#btnRsiLogDetail:hover, QPushButton#btnUartLogDetail:hover {"
            "  background: #f0f0f0;"
            "  border-color: #999999;"
            "}"
            "QPushButton#btnRsiLogDetail:disabled, QPushButton#btnUartLogDetail:disabled {"
            "  background: #eeeeee;"
            "  color: #aaaaaa;"
            "  border-color: #dddddd;"
            "}"
            "QLabel#fieldLabel { color: #666666; }"
            "QLabel#valueLabel {"
            "  font-weight: 600;"
            "  color: #222222;"
            "}"
            "QGroupBox#groupHeartbeat::title { color: #b42318; }"
            "QGroupBox#groupKuka::title { color: #b15e00; }"
            "QGroupBox#groupPrintheadCF::title { color: #000000; }"
            "QGroupBox#groupExtrudeScale::title { color: #000000; }"
            "QGroupBox#groupPrintheadResin::title { color: #444444; }"
            "QLabel#axisLabel {"
            "  font-size: 10px;"
            "  color: #666666;"
            "}"
            "QLabel#axisValue {"
            "  font-weight: 600;"
            "  border: 1px solid #d0d0d0;"
            "  border-radius: 4px;"
            "  background: #fafafa;"
            "  padding: 2px 4px;"
            "  color: #666666;"
            "}"
            "QPushButton#btnPause {"
            "  font-weight: 600;"
            "  font-size: 13px;"
            "  border: 1px solid #c0c0c0;"
            "  border-radius: 6px;"
            "  background: #ffffff;"
            "  color: #333333;"
            "  padding: 6px 16px;"
            "}"
            "QPushButton#btnPause:hover {"
            "  background: #f0f0f0;"
            "  border-color: #999999;"
            "}"
            "QPushButton#btnPause:disabled {"
            "  background: #eeeeee;"
            "  color: #aaaaaa;"
            "  border-color: #dddddd;"
            "}"
            "QPushButton#btnResume {"
            "  font-weight: 600;"
            "  font-size: 13px;"
            "  border: 1px solid #34a853;"
            "  border-radius: 6px;"
            "  background: #ffffff;"
            "  color: #1b6e3c;"
            "  padding: 6px 16px;"
            "}"
            "QPushButton#btnResume:hover {"
            "  background: #e6f4ea;"
            "  border-color: #1b6e3c;"
            "}"
            "QPushButton#btnResume:disabled {"
            "  background: #eeeeee;"
            "  color: #aaaaaa;"
            "  border-color: #dddddd;"
            "}"
            "QPushButton#btnStop {"
            "  font-weight: 600;"
            "  font-size: 13px;"
            "  border: 1px solid #d93025;"
            "  border-radius: 6px;"
            "  background: #ffffff;"
            "  color: #b42318;"
            "  padding: 6px 16px;"
            "}"
            "QPushButton#btnStop:hover {"
            "  background: #fce8e6;"
            "  border-color: #b42318;"
            "}"
            "QPushButton#btnStop:disabled {"
            "  background: #eeeeee;"
            "  color: #aaaaaa;"
            "  border-color: #dddddd;"
            "}"
            "QLabel#controlStatus {"
            "  font-weight: 700;"
            "  font-size: 13px;"
            "  color: #2b2b2b;"
            "}"
            "QGroupBox#groupControl::title { color: #1a73e8; }"
            "QGroupBox#groupLatency::title { color: #0f766e; }"
            "QGroupBox#groupPrintheadControl::title { color: #1a73e8; }"
            "QGroupBox#groupCtrlcf::title { color: #000000; }"
            "QGroupBox#groupCtrlresin::title { color: #444444; }"
            "QPushButton[objectName^='btnFanOn'] {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #34a853;"
            "  border-radius: 5px;"
            "  background: #ffffff;"
            "  color: #1b6e3c;"
            "  padding: 4px 14px;"
            "}"
            "QPushButton[objectName^='btnFanOn']:hover {"
            "  background: #e6f4ea;"
            "  border-color: #1b6e3c;"
            "}"
            "QPushButton[objectName^='btnFanOff'] {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #c0c0c0;"
            "  border-radius: 5px;"
            "  background: #ffffff;"
            "  color: #666666;"
            "  padding: 4px 14px;"
            "}"
            "QPushButton[objectName^='btnFanOff']:hover {"
            "  background: #f0f0f0;"
            "  border-color: #999999;"
            "}"
            "QPushButton[objectName^='btnTempApply'] {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #1a73e8;"
            "  border-radius: 5px;"
            "  background: #ffffff;"
            "  color: #1a73e8;"
            "  padding: 4px 14px;"
            "}"
            "QPushButton[objectName^='btnTempApply']:hover {"
            "  background: #e8f0fe;"
            "  border-color: #1558b0;"
            "}"
            "QPushButton#btnToolCF, QPushButton#btnToolResin {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #b15e00;"
            "  border-radius: 5px;"
            "  background: #ffffff;"
            "  color: #b15e00;"
            "  padding: 4px 16px;"
            "}"
            "QPushButton#btnToolCF:hover, QPushButton#btnToolResin:hover {"
            "  background: #fff3e0;"
            "  border-color: #8a4500;"
            "}"
            "QGroupBox#groupPrintheadControl QLineEdit {"
            "  border: 1px solid #d0d0d0;"
            "  border-radius: 4px;"
            "  padding: 4px 6px;"
            "  background: #ffffff;"
            "}"
            "QGroupBox#groupPrintheadControl QLineEdit:focus {"
            "  border-color: #1a73e8;"
            "}"
            "QGroupBox#groupLaunch::title { color: #1a73e8; }"
            "QPushButton#btnLaunchSettings {"
            "  font-weight: 600;"
            "  font-size: 13px;"
            "  border: 1px solid #c0c0c0;"
            "  border-radius: 6px;"
            "  background: #ffffff;"
            "  color: #333333;"
            "  padding: 6px 16px;"
            "}"
            "QPushButton#btnLaunchSettings:hover {"
            "  background: #f0f0f0;"
            "  border-color: #999999;"
            "}"
            "QPushButton#btnLaunch {"
            "  font-weight: 600;"
            "  font-size: 13px;"
            "  border: 1px solid #1a73e8;"
            "  border-radius: 6px;"
            "  background: #1a73e8;"
            "  color: #ffffff;"
            "  padding: 6px 16px;"
            "}"
            "QPushButton#btnLaunch:hover {"
            "  background: #1558b0;"
            "  border-color: #1558b0;"
            "}"
            "QPushButton#btnLaunch:disabled {"
            "  background: #eeeeee;"
            "  color: #aaaaaa;"
            "  border-color: #dddddd;"
            "}"
            "QPushButton#btnStopLaunch {"
            "  font-weight: 600;"
            "  font-size: 13px;"
            "  border: 1px solid #d93025;"
            "  border-radius: 6px;"
            "  background: #ffffff;"
            "  color: #b42318;"
            "  padding: 6px 16px;"
            "}"
            "QPushButton#btnStopLaunch:hover {"
            "  background: #fce8e6;"
            "  border-color: #b42318;"
            "}"
            "QPushButton#btnStopLaunch:disabled {"
            "  background: #eeeeee;"
            "  color: #aaaaaa;"
            "  border-color: #dddddd;"
            "}"
            "QLabel#launchStatus {"
            "  font-weight: 700;"
            "  font-size: 13px;"
            "  color: #666666;"
            "}"
            "QGroupBox#groupExport QDoubleSpinBox {"
            "  border: 1px solid #d0d0d0;"
            "  border-radius: 4px;"
            "  padding: 4px 6px;"
            "  background: #ffffff;"
            "}"
            "QGroupBox#groupExport QDoubleSpinBox:focus {"
            "  border-color: #1a73e8;"
            "}"
            "QGroupBox#groupExport::title { color: #1a73e8; }"
            "QGroupBox#groupExport QLineEdit {"
            "  border: 1px solid #d0d0d0;"
            "  border-radius: 4px;"
            "  padding: 4px 6px;"
            "  background: #ffffff;"
            "}"
            "QGroupBox#groupExport QLineEdit:focus {"
            "  border-color: #1a73e8;"
            "}"
            "QPushButton#btnBrowseGcode {"
            "  border: 1px solid #1a73e8;"
            "  border-radius: 4px;"
            "  background: #ffffff;"
            "  color: #1a73e8;"
            "  font-weight: 600;"
            "}"
            "QPushButton#btnBrowseGcode:hover {"
            "  background: #e8f0fe;"
            "}"
            "QPushButton#btnPlannerToggle {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #d0d0d0;"
            "  border-radius: 5px;"
            "  background: #fafafa;"
            "  color: #555555;"
            "  padding: 4px 12px;"
            "  text-align: left;"
            "}"
            "QPushButton#btnPlannerToggle:hover {"
            "  background: #f0f0f0;"
            "  border-color: #999999;"
            "}"
            "QPushButton[objectName^='btnPidToggle'] {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #d0d0d0;"
            "  border-radius: 5px;"
            "  background: #fafafa;"
            "  color: #555555;"
            "  padding: 4px 12px;"
            "  text-align: left;"
            "}"
            "QPushButton[objectName^='btnPidToggle']:hover {"
            "  background: #f0f0f0;"
            "  border-color: #999999;"
            "}"
            "QPushButton[objectName^='btnPidRestore'] {"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid #c0c0c0;"
            "  border-radius: 5px;"
            "  background: #ffffff;"
            "  color: #666666;"
            "  padding: 4px 14px;"
            "}"
            "QPushButton[objectName^='btnPidRestore']:hover {"
            "  background: #f0f0f0;"
            "  border-color: #999999;"
            "}"
            "QPushButton#btnExportNpz {"
            "  font-weight: 600;"
            "  font-size: 13px;"
            "  border: 1px solid #1a73e8;"
            "  border-radius: 6px;"
            "  background: #1a73e8;"
            "  color: #ffffff;"
            "  padding: 6px 16px;"
            "}"
            "QPushButton#btnExportNpz:hover {"
            "  background: #1558b0;"
            "  border-color: #1558b0;"
            "}"
            "QPushButton#btnExportNpz:disabled {"
            "  background: #eeeeee;"
            "  color: #aaaaaa;"
            "  border-color: #dddddd;"
            "}")

        self._btn_mode_test.clicked.connect(lambda: self._set_active_mode(_MODE_PAGE_TEST))
        self._btn_mode_print.clicked.connect(lambda: self._set_active_mode(_MODE_PAGE_PRINT))
        self._btn_mode_back.clicked.connect(self._show_mode_select)
        self._btn_rsi_log_detail.clicked.connect(self._show_rsi_log_detail)
        self._btn_uart_log_detail.clicked.connect(self._show_uart_log_detail)
        self._show_mode_select()

        self._extrude_scale_apply.clicked.connect(self._on_extrude_scale_apply)
        self._extrude_scale_input.returnPressed.connect(self._on_extrude_scale_apply)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_resume.clicked.connect(self._on_resume)
        self._btn_stop.clicked.connect(self._on_stop)

        # Printhead control connections
        self._btn_fan_on_cf.clicked.connect(lambda: self._on_fan("cf", True))
        self._btn_fan_off_cf.clicked.connect(lambda: self._on_fan("cf", False))
        self._btn_fan_on_resin.clicked.connect(lambda: self._on_fan("resin", True))
        self._btn_fan_off_resin.clicked.connect(lambda: self._on_fan("resin", False))
        self._btn_temp_apply_cf.clicked.connect(lambda: self._on_temp_apply("cf"))
        self._temp_input_cf.returnPressed.connect(lambda: self._on_temp_apply("cf"))
        self._btn_temp_apply_resin.clicked.connect(lambda: self._on_temp_apply("resin"))
        self._temp_input_resin.returnPressed.connect(lambda: self._on_temp_apply("resin"))
        self._btn_tool_cf.clicked.connect(lambda: self._on_tool_switch("cf"))
        self._btn_tool_resin.clicked.connect(lambda: self._on_tool_switch("resin"))

        # PID control connections
        self._btn_pid_apply_cf.clicked.connect(lambda: self._on_pid_apply("cf"))
        self._btn_pid_apply_resin.clicked.connect(lambda: self._on_pid_apply("resin"))
        self._btn_pid_restore_cf.clicked.connect(lambda: self._on_pid_restore("cf"))
        self._btn_pid_restore_resin.clicked.connect(lambda: self._on_pid_restore("resin"))

    def _set_value(self, key, text, color=None):
        label = self._labels.get(key)
        if not label:
            return
        label.setText(text)
        if isinstance(label, _AutoScaleLabel):
            if color:
                label.set_color(color)
        else:
            if color:
                label.setStyleSheet(f"color: {color};")
            else:
                label.setStyleSheet("")

    def _format_tool(self, tool_id):
        if tool_id == 1:
            return "碳纤维"
        if tool_id == 2:
            return "树脂"
        return str(tool_id)

    def _on_extrude_scale_apply(self):
        text = self._extrude_scale_input.text().strip()
        if not text:
            self._set_extrude_status("请输入倍率值。", "#b42318")
            return
        try:
            val = float(text)
        except ValueError:
            self._set_extrude_status("无效倍率值。", "#b42318")
            return
        if not (val > 0.0):
            self._set_extrude_status("倍率必须大于 0。", "#b42318")
            return
        self._set_extrude_status("提交中...", "#b15e00")
        self.scale_submit.emit(val)

    def _set_extrude_status(self, text, color=None):
        self._extrude_scale_status.setText(text)
        if color:
            self._extrude_scale_status.setStyleSheet(f"color: {color};")
        else:
            self._extrude_scale_status.setStyleSheet("")

    def set_extrude_scale(self, value, status_text=None, status_color=None):
        self._extrude_scale_current = value
        self._extrude_scale_value.setText(f"{value:.3f}")
        if status_text is not None:
            self._set_extrude_status(status_text, status_color)

    def set_extrude_status(self, text, color=None):
        self._set_extrude_status(text, color)

    def current_extrude_scale(self):
        return self._extrude_scale_current

    def current_tool_id(self):
        return self._current_tool_id

    def _format_diagnostic_time(self, epoch):
        local = time.localtime(epoch)
        ms = int((epoch - int(epoch)) * 1000)
        return time.strftime("%Y-%m-%d %H:%M:%S", local) + f".{ms:03d}"

    def _append_diagnostic(self, source, kind, detail):
        epoch = time.time()
        record = {
            "time_epoch": epoch,
            "time": self._format_diagnostic_time(epoch),
            "source": str(source),
            "kind": str(kind),
            "detail": detail,
        }
        self._diagnostic_log_history.append(record)
        overflow = len(self._diagnostic_log_history) - self._DIAGNOSTIC_LOG_LIMIT
        if overflow > 0:
            del self._diagnostic_log_history[:overflow]

    def _trajectory_diagnostic(self, traj):
        return {
            "seq": int(getattr(traj, "seq", 0)),
            "tool_id": int(getattr(traj, "tool_id", 0)),
            "x": float(getattr(traj, "x", 0.0)),
            "y": float(getattr(traj, "y", 0.0)),
            "z": float(getattr(traj, "z", 0.0)),
            "a": float(getattr(traj, "a", 0.0)),
            "b": float(getattr(traj, "b", 0.0)),
            "c": float(getattr(traj, "c", 0.0)),
            "e": float(getattr(traj, "e", 0.0)),
        }

    def _event_diagnostic(self, event):
        return {
            "event_type": str(getattr(event, "event_type", "") or ""),
            "payload": str(getattr(event, "payload", "") or ""),
            "event_src_line": int(getattr(event, "event_src_line", 0)),
            "trigger_seq": int(getattr(event, "trigger_seq", 0)),
        }

    def _printhead_diagnostic(self, ps):
        return {
            "ready_for_motion": bool(getattr(ps, "ready_for_motion", False)),
            "ready_event_seq": int(getattr(ps, "ready_event_seq", 0)),
            "ready_event_type": str(getattr(ps, "ready_event_type", "") or ""),
            "fan_ok_cf": bool(getattr(ps, "fan_ok_cf", False)),
            "fan_ok_resin": bool(getattr(ps, "fan_ok_resin", False)),
            "current_temp_cf": float(getattr(ps, "current_temp_cf", 0.0)),
            "target_temp_cf": float(getattr(ps, "target_temp_cf", 0.0)),
            "current_temp_resin": float(getattr(ps, "current_temp_resin", 0.0)),
            "target_temp_resin": float(getattr(ps, "target_temp_resin", 0.0)),
            "current_tool": int(getattr(ps, "current_tool", 0)),
            "error_code": int(getattr(ps, "error_code", 0)),
            "raw": str(getattr(ps, "raw", "") or ""),
        }

    def _show_latency_detail(self):
        self._latency_diag_widget.adjustSize()
        self._latency_diag_widget.show()
        self._latency_diag_widget.raise_()
        self._latency_diag_widget.activateWindow()

    def _update_latency(self, msg: ExtruderLatencyStatus):
        self._append_diagnostic("latency", "snapshot", {
            "arm_seq": int(msg.arm_seq),
            "last_eack_seq": int(msg.last_eack_seq),
            "stat_last_e_seq": int(msg.stat_last_e_seq),
            "has_eack": bool(msg.has_eack),
            "has_stat": bool(msg.has_stat),
            "seq_lag": int(msg.seq_lag),
            "cycle_lag_ms": float(msg.cycle_lag_ms),
            "ack_delay_ms": float(msg.ack_delay_ms),
            "linux_mcu_delay_ms": float(msg.linux_mcu_delay_ms),
            "has_robot_match": bool(msg.has_robot_match),
            "actual_robot_seq": int(msg.actual_robot_seq),
            "robot_match_error_mm": float(msg.robot_match_error_mm),
            "mcu_robot_delay_ms": float(msg.mcu_robot_delay_ms),
            "eack_count": int(msg.eack_count),
            "old_seq_warn_count": int(msg.old_seq_warn_count),
            "gap_warn_count": int(msg.gap_warn_count),
            "last_tool_id": int(msg.last_tool_id),
            "last_extrude_abs": float(msg.last_extrude_abs),
            "last_mcu_us": int(msg.last_mcu_us),
            "last_warn": str(msg.last_warn or ""),
            "last_raw": str(msg.last_raw or ""),
        })
        link_labels = getattr(self, "_latency_link_labels", {})
        diag_labels = getattr(self, "_latency_diag_labels", {})
        if not link_labels:
            return

        def finite(value):
            return value == value

        def fmt_ms(value):
            if not finite(value):
                return "-"
            return f"{value:.0f} ms"

        def fmt_signed_ms(value):
            if not finite(value):
                return "-"
            return f"{value:.0f} ms"

        def fmt_unc(value, uncertainty):
            if not finite(value):
                return "-"
            if finite(uncertainty) and uncertainty > 0.0:
                return f"{value:.0f} ms ±{uncertainty:.0f} ms"
            return f"{value:.0f} ms"

        def set_link(key, current_text, p95_text, color="#2b2b2b"):
            labels = link_labels.get(key)
            if not labels:
                return
            value_label, p95_label = labels
            value_label.setText(current_text)
            value_label.setStyleSheet(f"color: {color};")
            p95_label.setText(p95_text)

        has_eack = bool(msg.has_eack)
        has_robot = bool(msg.has_robot_match)
        robot_unc = msg.robot_match_uncertainty_ms if has_robot else float("nan")

        set_link(
            "linux_mcu",
            fmt_ms(msg.linux_mcu_delay_ms) if has_eack else "-",
            f"P95 {fmt_ms(msg.linux_mcu_p95_ms)}" if has_eack else "P95 -",
        )
        set_link(
            "linux_robot",
            fmt_unc(msg.linux_robot_delay_ms, robot_unc) if has_robot else "-",
            f"P95 {fmt_ms(msg.linux_robot_p95_ms)}" if has_robot else "P95 -",
        )
        mcu_robot_color = "#2b2b2b"
        if has_robot and has_eack:
            if msg.mcu_robot_delay_ms < 0:
                mcu_robot_color = "#1b6e3c"
            elif msg.mcu_robot_delay_ms > 0:
                mcu_robot_color = "#b15e00"
        set_link(
            "mcu_robot",
            fmt_unc(
                msg.mcu_robot_delay_ms,
                robot_unc) if has_robot and has_eack else "-",
            (
                f"P95 |偏移| {fmt_ms(msg.mcu_robot_abs_p95_ms)}"
                if has_robot and has_eack
                else "P95 |偏移| -"
            ),
            mcu_robot_color,
        )

        if not has_robot:
            if finite(msg.robot_match_error_mm):
                summary = (
                    "挤出相对机械臂：无法估计，"
                    f"匹配误差 {msg.robot_match_error_mm:.2f} mm 超过 "
                    f"{msg.robot_match_max_error_mm:.2f} mm"
                )
            else:
                summary = "挤出相对机械臂：等待机械臂位置匹配"
        elif not has_eack:
            summary = "挤出相对机械臂：等待 EACK"
        else:
            value = msg.mcu_robot_delay_ms
            uncertainty = max(0.0, robot_unc if finite(robot_unc) else 0.0)
            low = value - uncertainty
            high = value + uncertainty
            if high < 0:
                summary = f"挤出相对机械臂：提前，约 {abs(high):.0f} ~ {abs(low):.0f} ms"
            elif low > 0:
                summary = f"挤出相对机械臂：滞后，约 {low:.0f} ~ {high:.0f} ms"
            else:
                summary = f"挤出与机械臂接近同步，方向不确定，范围 {low:.0f} ~ {high:.0f} ms"
        self._latency_summary.setText(summary)

        def set_diag(key, text, color="#666666"):
            label = diag_labels.get(key)
            if label is not None:
                label.setText(text)
                label.setStyleSheet(f"color: {color};")

        set_diag("linux_mcu_p99", fmt_ms(msg.linux_mcu_p99_ms) if has_eack else "-")
        set_diag("linux_robot_p99", fmt_ms(msg.linux_robot_p99_ms) if has_robot else "-")
        set_diag("mcu_robot_avg", fmt_signed_ms(msg.mcu_robot_avg_ms)
                 if has_robot and has_eack else "-")
        set_diag("mcu_robot_p99", fmt_ms(msg.mcu_robot_abs_p99_ms)
                 if has_robot and has_eack else "-")
        set_diag("robot_seq", str(msg.actual_robot_seq) if has_robot else "-")
        set_diag("seqs", f"发送 {msg.arm_seq} / EACK {msg.last_eack_seq if has_eack else '-'}")
        if finite(msg.robot_match_error_mm):
            set_diag(
                "match_error",
                f"{msg.robot_match_error_mm:.2f} / {msg.robot_match_max_error_mm:.2f} mm")
        else:
            set_diag("match_error", "-")
        set_diag("uncertainty", f"±{msg.robot_match_uncertainty_ms:.0f} ms" if has_robot else "-")

        warn_text = msg.last_warn or "无 EWARN"
        warn_color = "#b42318" if msg.last_warn else "#666666"
        set_diag(
            "warn",
            (
                f"EACK {msg.eack_count} / old {msg.old_seq_warn_count} / "
                f"gap {msg.gap_warn_count} / {warn_text}"
            ),
            warn_color,
        )

    def _update_print_time(self, msg: UiStatus):
        if not msg.print_time_valid:
            self._print_time_label.setText("时间估计 --")
            return
        self._print_time_label.setText(
            "总 " + _format_print_duration(msg.planned_total_time_s) +
            " | 已用 " + _format_print_duration(msg.planned_elapsed_time_s) +
            " | 剩余约 " + _format_print_duration(msg.planned_remaining_time_s)
        )

    def _update_print_progress(self, msg: UiStatus):
        self._update_print_time(msg)
        if not msg.current_traj_valid:
            self._print_progress_bar.setValue(0)
            self._print_progress_label.setText("层进度 -- / --")
            return
        layer_index = int(msg.current_traj.layer_index)
        total_layers = int(msg.current_traj.total_layers)
        if total_layers <= 0:
            self._print_progress_bar.setValue(0)
            self._print_progress_label.setText("层进度 -- / --")
            return
        current_layer = max(1, min(total_layers, layer_index + 1))
        progress = int(round(current_layer * 100.0 / total_layers))
        self._print_progress_bar.setValue(max(0, min(100, progress)))
        self._print_progress_label.setText(f"第 {current_layer} / {total_layers} 层")

    def _update_ui(self, msg: UiStatus):
        self._append_diagnostic("ui_status", "snapshot", {
            "state": str(msg.state or ""),
            "last_warn": str(msg.last_warn or ""),
            "last_error": str(msg.last_error or ""),
            "ready_for_motion": bool(msg.ready_for_motion),
            "kuka_status_valid": bool(msg.kuka_status_valid),
            "kuka_status_age_s": float(msg.kuka_status_age_s),
            "rsi_heartbeat_valid": bool(msg.rsi_heartbeat_valid),
            "rsi_heartbeat_age_s": float(msg.rsi_heartbeat_age_s),
            "printhead_status_valid": bool(msg.printhead_status_valid),
            "printhead_status_age_s": float(msg.printhead_status_age_s),
            "printhead_status": self._printhead_diagnostic(msg.printhead_status),
            "traj_next_seq": int(msg.traj_next_seq),
            "traj_backlog": int(msg.traj_backlog),
            "event_next_seq": int(msg.event_next_seq),
            "event_pending": int(msg.event_pending),
            "current_traj_valid": bool(msg.current_traj_valid),
            "next_traj_valid": bool(msg.next_traj_valid),
            "current_event_valid": bool(msg.current_event_valid),
            "next_event_valid": bool(msg.next_event_valid),
            "current_traj": self._trajectory_diagnostic(msg.current_traj),
            "next_traj": self._trajectory_diagnostic(msg.next_traj),
            "current_event": self._event_diagnostic(msg.current_event),
            "next_event": self._event_diagnostic(msg.next_event),
            "print_test_busy": bool(self._print_test_busy),
            "print_test_seen_correction": bool(self._print_test_seen_correction),
            "print_test_target": (
                list(self._print_test_target)
                if self._print_test_target is not None
                else None
            ),
            "print_test_resin_temp": (
                list(self._print_test_resin_temp)
                if self._print_test_resin_temp is not None
                else None
            ),
        })
        self._update_print_progress(msg)
        state_str = msg.state or "-"
        state_color = self._STATE_COLORS.get(state_str, "#a0a0a0")
        self._set_value("System State", state_str, state_color)
        self._update_control_buttons(msg.state or "")

        previous_tool_id = self._current_tool_id
        if msg.printhead_status_valid:
            ps = msg.printhead_status
            self._current_tool_id = int(ps.current_tool)
            self._print_test_printhead_status = ps
            self._current_tool_value.setText(self._format_tool(ps.current_tool))

            using_color = "#1b6e3c"
            cf_state = "使用中" if ps.current_tool == 1 else "空闲"
            resin_state = "使用中" if ps.current_tool == 2 else "空闲"
            self._set_value(
                "Carbon Fiber State",
                cf_state,
                using_color if cf_state == "使用中" else "#2b2b2b")
            self._set_value(
                "Resin State",
                resin_state,
                using_color if resin_state == "使用中" else "#2b2b2b")

            cf_fan_color = "#1b6e3c" if ps.fan_ok_cf else "#b42318"
            self._set_value("Carbon Fiber Fan OK", "开" if ps.fan_ok_cf else "关", cf_fan_color)
            self._set_value("Carbon Fiber Current Temp", f"{ps.current_temp_cf:.1f}", "#2b2b2b")
            self._set_value("Carbon Fiber Target Temp", f"{ps.target_temp_cf:.1f}", "#2b2b2b")

            resin_fan_color = "#1b6e3c" if ps.fan_ok_resin else "#b42318"
            self._set_value("Resin Fan OK", "开" if ps.fan_ok_resin else "关", resin_fan_color)
            self._set_value("Resin Current Temp", f"{ps.current_temp_resin:.1f}", "#2b2b2b")
            self._set_value("Resin Target Temp", f"{ps.target_temp_resin:.1f}", "#2b2b2b")
            self._print_test_resin_temp = (ps.current_temp_resin, ps.target_temp_resin)
            if (
                self._print_test_waiting_for_tool is not None
                and int(ps.current_tool) == int(self._print_test_waiting_for_tool)
            ):
                pending_after_tool_change = self._print_test_pending_after_tool_change
                head_key = (
                    "fiber"
                    if int(self._print_test_waiting_for_tool) == _PRINT_TEST_FIBER_TOOL_ID
                    else "resin"
                )
                ready, reason = self._print_test_head_ready(head_key, ps)
                if not ready:
                    self._set_print_test_status(reason, "#b15e00")
                    return
                # Tool changes and head preparation are asynchronous; keep the
                # requested follow-up action separate so the original print path resumes.
                self._print_test_waiting_for_tool = None
                self._print_test_requested_target_tool = None
                self._print_test_pending_after_tool_change = None
                if pending_after_tool_change == "print_resin_matrix":
                    self._set_print_test_status(
                        "树脂头、风扇已就绪，开始树脂测试...", "#1b6e3c"
                    )
                    self._start_print_test_resin_matrix()
                elif pending_after_tool_change == "adjust_fiber_offset":
                    self._set_print_test_controls_enabled(self._print_test_seen_correction)
                    self._set_print_test_status(
                        "纤维头已切换完成，可调整纤维偏置。", "#1b6e3c"
                    )
                elif pending_after_tool_change == "print_fiber_matrix":
                    self._set_print_test_status(
                        "纤维头、风扇已就绪，开始纤维测试...", "#1b6e3c"
                    )
                    self._start_print_test_fiber_matrix()
                elif pending_after_tool_change == "print_composite_matrix":
                    self._set_print_test_status(
                        "纤维头、风扇已就绪，开始复合打印...", "#1b6e3c"
                    )
                    self._start_print_test_composite_matrix()
                else:
                    self._set_print_test_controls_enabled(self._print_test_seen_correction)
                    self._set_print_test_status("打印头已切换完成。", "#1b6e3c")
        else:
            self._current_tool_id = 0
            self._print_test_printhead_status = None
            missing_keys = [
                "Carbon Fiber State",
                "Carbon Fiber Fan OK",
                "Carbon Fiber Current Temp",
                "Carbon Fiber Target Temp",
                "Resin State",
                "Resin Fan OK",
                "Resin Current Temp",
                "Resin Target Temp",
            ]
            for key in missing_keys:
                self._set_value(key, "-", "#b42318")
            self._current_tool_value.setText("-")
        if (
            previous_tool_id != self._current_tool_id
            and self._print_test_params is not None
            and not self._print_test_busy
        ):
            self._set_print_test_controls_enabled(self._print_test_seen_correction)

        if msg.kuka_status_valid:
            ks = msg.kuka_status
            self._set_value("KUKA X", f"{ks.x:.2f}")
            self._set_value("KUKA Y", f"{ks.y:.2f}")
            self._set_value("KUKA Z", f"{ks.z:.2f}")
            self._set_value("KUKA A", f"{ks.a:.2f}")
            self._set_value("KUKA B", f"{ks.b:.2f}")
            self._set_value("KUKA C", f"{ks.c:.2f}")
        else:
            for axis in ("X", "Y", "Z", "A", "B", "C"):
                self._set_value(f"KUKA {axis}", "-", "#b42318")

        self._set_value("Traj Backlog", str(msg.traj_backlog), "#2b2b2b")
        self._set_value("Next Traj Seq", str(msg.traj_next_seq), "#2b2b2b")
        self._set_value("Next Event Seq", str(msg.event_next_seq), "#2b2b2b")
        self._set_value("Events Pending", str(msg.event_pending), "#2b2b2b")
        if msg.current_traj_valid:
            ct = msg.current_traj
            self._set_value("Traj Seq", str(ct.seq), "#2b2b2b")
            self._set_value("Traj Tool", self._format_tool(ct.tool_id), "#2b2b2b")
            self._set_value("Traj X", f"{ct.x:.2f}", "#2b2b2b")
            self._set_value("Traj Y", f"{ct.y:.2f}", "#2b2b2b")
            self._set_value("Traj Z", f"{ct.z:.2f}", "#2b2b2b")
            self._set_value("Traj A", f"{ct.a:.2f}", "#2b2b2b")
            self._set_value("Traj B", f"{ct.b:.2f}", "#2b2b2b")
            self._set_value("Traj C", f"{ct.c:.2f}", "#2b2b2b")
            self._set_value("Traj E", f"{ct.e:.3f}", "#2b2b2b")
        else:
            for key in (
                "Traj Seq",
                "Traj Tool",
                "Traj X",
                "Traj Y",
                "Traj Z",
                "Traj A",
                "Traj B",
                "Traj C",
                "Traj E",
            ):
                self._set_value(key, "-", "#b42318")

        if msg.next_traj_valid:
            nt = msg.next_traj
            self._set_value("Traj Seq (Next)", str(nt.seq), "#2b2b2b")
            self._set_value("Traj Tool (Next)", self._format_tool(nt.tool_id), "#2b2b2b")
            self._set_value("Traj X (Next)", f"{nt.x:.2f}", "#2b2b2b")
            self._set_value("Traj Y (Next)", f"{nt.y:.2f}", "#2b2b2b")
            self._set_value("Traj Z (Next)", f"{nt.z:.2f}", "#2b2b2b")
            self._set_value("Traj A (Next)", f"{nt.a:.2f}", "#2b2b2b")
            self._set_value("Traj B (Next)", f"{nt.b:.2f}", "#2b2b2b")
            self._set_value("Traj C (Next)", f"{nt.c:.2f}", "#2b2b2b")
            self._set_value("Traj E (Next)", f"{nt.e:.3f}", "#2b2b2b")
        else:
            for key in (
                "Traj Seq (Next)",
                "Traj Tool (Next)",
                "Traj X (Next)",
                "Traj Y (Next)",
                "Traj Z (Next)",
                "Traj A (Next)",
                "Traj B (Next)",
                "Traj C (Next)",
                "Traj E (Next)",
            ):
                self._set_value(key, "-", "#b42318")

        if msg.current_event_valid:
            ce = msg.current_event
            self._set_value("Event Type", ce.event_type or "-", "#2b2b2b")
            self._set_value("Event Payload", ce.payload or "-", "#2b2b2b")
            self._set_value("Event Src Line", str(ce.event_src_line), "#2b2b2b")
            self._set_value("Event Trigger Seq", str(ce.trigger_seq), "#2b2b2b")
        else:
            for key in (
                "Event Type",
                "Event Payload",
                "Event Src Line",
                "Event Trigger Seq",
            ):
                self._set_value(key, "-", "#b42318")

        if msg.next_event_valid:
            ne = msg.next_event
            self._set_value("Event Type (Next)", ne.event_type or "-", "#2b2b2b")
            self._set_value("Event Payload (Next)", ne.payload or "-", "#2b2b2b")
            self._set_value("Event Src Line (Next)", str(ne.event_src_line), "#2b2b2b")
            self._set_value("Event Trigger Seq (Next)", str(ne.trigger_seq), "#2b2b2b")
        else:
            for key in (
                "Event Type (Next)",
                "Event Payload (Next)",
                "Event Src Line (Next)",
                "Event Trigger Seq (Next)",
            ):
                self._set_value(key, "-", "#b42318")

    def _on_pause(self):
        self.command_submit.emit("REQUEST_PAUSE")

    def _on_resume(self):
        self.command_submit.emit("RESUME")

    def _on_stop(self):
        reply = _ask_yes_no(
            self,
            "确认停止",
            "确定要停止打印吗？\n\n"
            "这将抬升 Z 轴，然后切断所有通信。\n"
            "KUKA 将触发安全停机。",
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.command_submit.emit("ABORT")

    def _on_fan(self, head_id, on):
        val = "1" if on else "0"
        cmd = f"EV 0 fan_{head_id} {val}\n"
        self.uart_command_submit.emit(cmd)

    def _on_temp_apply(self, head_id):
        temp_input = getattr(self, f"_temp_input_{head_id}")
        text = temp_input.text().strip()
        if not text:
            return
        try:
            temp = float(text)
        except ValueError:
            return
        if temp < 0:
            return
        cmd = f"EV 0 heat_{head_id} {temp}\n"
        self.uart_command_submit.emit(cmd)

    def _on_tool_switch(self, head_id):
        tool_id = "1" if head_id == "cf" else "2"
        event_type = f"tool_change_{head_id}"
        cmd = f"EV 0 {event_type} {tool_id}\n"
        self.uart_command_submit.emit(cmd)

    def _on_pid_apply(self, head_id):
        pid_inputs = getattr(self, f"_pid_inputs_{head_id}")
        pid_status = getattr(self, f"_pid_status_{head_id}")
        values = {}
        for key in _PID_PARAM_KEYS:
            text = pid_inputs[key].text().strip()
            if not text:
                pid_status.setText("参数不能为空。")
                pid_status.setStyleSheet("color: #b42318;")
                return
            try:
                values[key] = float(text)
            except ValueError:
                pid_status.setText(f"无效参数: {_PID_PARAM_LABELS[key]}")
                pid_status.setStyleSheet("color: #b42318;")
                return
        cmd = (f"EV 0 pid_set_{head_id} {values['kp']} {values['ki']} {values['kd']} "
               f"{values['max_output']} {values['min_output']} "
               f"{values['max_integral']} {values['min_integral']}\n")
        self.uart_command_submit.emit(cmd)
        pid_status.setText("已发送。")
        pid_status.setStyleSheet("color: #1b6e3c;")

    def _on_pid_restore(self, head_id):
        pid_inputs = getattr(self, f"_pid_inputs_{head_id}")
        pid_status = getattr(self, f"_pid_status_{head_id}")
        defaults = _PID_DEFAULTS[head_id]
        for key in _PID_PARAM_KEYS:
            pid_inputs[key].setText(f"{defaults[key]:.2f}")
        pid_status.setText("已恢复默认值。")
        pid_status.setStyleSheet("color: #1a73e8;")

    # ---- Print test ----

    def _set_print_test_status(self, text, color=None):
        self._append_diagnostic(
            "ui", "print_test_status", {
                "text": str(text), "color": str(
                    color or "")})
        self._test_status.setText(text)
        if color:
            self._test_status.setStyleSheet(f"color: {color};")
        else:
            self._test_status.setStyleSheet("")

    def _float_input(self, widget, label, *, minimum=None, allow_zero=True):
        raw = widget.text().strip()
        if not raw:
            raise ValueError(f"{label}不能为空。")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{label}必须是数字。") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label}必须是有限数字。")
        if minimum is not None and value < float(minimum):
            raise ValueError(f"{label}不能小于 {float(minimum):g}。")
        if not allow_zero and value == 0.0:
            raise ValueError(f"{label}必须大于 0。")
        return value

    def _current_head_calibration_from_inputs(self):
        return HeadCalibration(
            resin_z_print_compensation_mm=self._float_input(
                self._test_resin_z_comp_input, "树脂 Z 补偿"
            ),
            fiber_x_print_compensation_mm=self._float_input(
                self._test_fiber_x_comp_input, "纤维 X 偏置"
            ),
            fiber_y_print_compensation_mm=self._float_input(
                self._test_fiber_y_comp_input, "纤维 Y 偏置"
            ),
            fiber_z_print_compensation_mm=self._float_input(
                self._test_fiber_z_comp_input, "纤维 Z 偏置", minimum=0.001
            ),
        )

    def _set_head_calibration_inputs(self, calibration):
        self._test_resin_z_comp_input.setText(
            f"{calibration.resin_z_print_compensation_mm:.3f}"
        )
        self._test_fiber_x_comp_input.setText(
            f"{calibration.fiber_x_print_compensation_mm:.3f}"
        )
        self._test_fiber_y_comp_input.setText(
            f"{calibration.fiber_y_print_compensation_mm:.3f}"
        )
        self._test_fiber_z_comp_input.setText(
            f"{calibration.fiber_z_print_compensation_mm:.3f}"
        )

    def _save_current_head_calibration(self):
        self._head_calibration = self._current_head_calibration_from_inputs()
        save_head_calibration(
            self._head_calibration, path=DEFAULT_HEAD_CALIBRATION_PATH
        )
        return self._head_calibration

    def _parse_print_test_params(self):
        from gcode_planner.print_test_generator import (
            TEST_MATRIX_MAX_LINES,
            expand_test_values,
        )

        speed = self._float_input(
            self._test_speed_input, "速度", minimum=0.0, allow_zero=False
        )
        line_length = self._float_input(
            self._test_line_length_input, "测试线长度", minimum=0.0, allow_zero=False
        )
        y_spacing = self._float_input(
            self._test_y_spacing_input, "Y 间距", minimum=0.0, allow_zero=False
        )
        tool_change_safe_lift = self._float_input(
            self._test_tool_change_safe_lift_input,
            "换头安全抬升",
            minimum=0.0,
            allow_zero=True,
        )

        resin_temp = self._float_input(
            self._test_temp_input, "树脂目标温度", minimum=0.0
        )
        resin_layer_heights = expand_test_values(
            f"{self._test_layer_height_min_input.text()}-"
            f"{self._test_layer_height_max_input.text()}",
            label="树脂层高",
        )
        resin_scales = expand_test_values(
            f"{self._test_scale_min_input.text()}-"
            f"{self._test_scale_max_input.text()}",
            label="树脂挤出倍率",
        )
        resin_prime_length = self._float_input(
            self._test_prime_length_input, "树脂预挤出长度", minimum=0.0
        )
        resin_prime_speed = self._float_input(
            self._test_prime_speed_input,
            "树脂预挤出速度",
            minimum=0.0,
            allow_zero=False,
        )
        resin_retract_length = self._float_input(
            self._test_retract_length_input, "树脂回抽长度", minimum=0.0
        )
        resin_retract_speed = self._float_input(
            self._test_retract_speed_input,
            "树脂回抽速度",
            minimum=0.0,
            allow_zero=False,
        )

        fiber_temp = self._float_input(
            self._test_fiber_temp_input, "纤维目标温度", minimum=0.0
        )
        fiber_layer_heights = expand_test_values(
            f"{self._test_fiber_layer_height_min_input.text()}-"
            f"{self._test_fiber_layer_height_max_input.text()}",
            label="纤维层高",
        )
        fiber_scales = expand_test_values(
            f"{self._test_fiber_scale_min_input.text()}-"
            f"{self._test_fiber_scale_max_input.text()}",
            label="纤维挤出倍率",
        )
        fiber_prime_length = self._float_input(
            self._test_fiber_prime_length_input, "纤维预挤出长度", minimum=0.0
        )
        fiber_prime_speed = self._float_input(
            self._test_fiber_prime_speed_input,
            "纤维预挤出速度",
            minimum=0.0,
            allow_zero=False,
        )
        fiber_retract_length = self._float_input(
            self._test_fiber_retract_length_input, "纤维回抽长度", minimum=0.0
        )
        fiber_retract_speed = self._float_input(
            self._test_fiber_retract_speed_input,
            "纤维回抽速度",
            minimum=0.0,
            allow_zero=False,
        )

        resin_line_count = len(resin_layer_heights) * len(resin_scales)
        fiber_line_count = len(fiber_layer_heights) * len(fiber_scales)
        if resin_line_count > TEST_MATRIX_MAX_LINES:
            raise ValueError(
                f"树脂测试线数量为 {resin_line_count}，超过最大 "
                f"{TEST_MATRIX_MAX_LINES} 条，请缩小范围。"
            )
        if fiber_line_count > TEST_MATRIX_MAX_LINES:
            raise ValueError(
                f"纤维测试线数量为 {fiber_line_count}，超过最大 "
                f"{TEST_MATRIX_MAX_LINES} 条，请缩小范围。"
            )
        return {
            "global": {
                "speed": speed,
                "line_length": line_length,
                "y_spacing": y_spacing,
                "tool_change_safe_lift": tool_change_safe_lift,
            },
            "resin": {
                "temp": resin_temp,
                "layer_heights": resin_layer_heights,
                "scales": resin_scales,
                "prime_length": resin_prime_length,
                "prime_speed": resin_prime_speed,
                "retract_length": resin_retract_length,
                "retract_speed": resin_retract_speed,
                "line_count": resin_line_count,
            },
            "fiber": {
                "temp": fiber_temp,
                "layer_heights": fiber_layer_heights,
                "scales": fiber_scales,
                "prime_length": fiber_prime_length,
                "prime_speed": fiber_prime_speed,
                "retract_length": fiber_retract_length,
                "retract_speed": fiber_retract_speed,
                "line_count": fiber_line_count,
            },
        }

    def _on_print_test_prepare(self):
        self._head_calibration = load_head_calibration()
        self._set_head_calibration_inputs(self._head_calibration)
        try:
            params = self._parse_print_test_params()
        except Exception as exc:
            self._set_print_test_status(f"参数无效: {exc}", "#b42318")
            return
        self._print_test_params = params
        self._print_test_seen_correction = False
        self._print_test_current_correction = _PRINT_TEST_ZERO_CORRECTION
        self._print_test_resin_temp = None
        self._print_test_printhead_status = None
        self._test_correction_label.setText("RSI 修正量: 未收到")
        self._print_test_resin_height_confirmed = False
        self._print_test_fiber_confirmed = False
        self._print_test_fiber_offset_initial_sent = False
        self._print_test_pending_initial_fiber_offset = None
        self._print_test_last_sent_fiber_offset = None
        self._print_test_waiting_for_tool = None
        self._print_test_pending_after_zero = None
        self._print_test_pending_after_tool_change = None
        self._print_test_requested_target_tool = None
        self._print_test_resin_z_floor = None
        self._uart_log_history.clear()
        self._diagnostic_log_history.clear()
        self._append_diagnostic("ui", "diagnostic_reset", {"reason": "print_test_prepare"})
        self._uart_log_text.clear()
        self._uart_log_latest_display = ""
        self._uart_log_summary.setText("测试日志已清空")
        self.scale_submit.emit(1.0)
        self.uart_command_submit.emit("EV 0 tool_change_resin 2\n")
        self.uart_command_submit.emit("EV 0 fan_resin 1\n")
        self.uart_command_submit.emit("EV 0 fan_cf 1\n")
        self.uart_command_submit.emit(f"EV 0 heat_resin {params['resin']['temp']}\n")
        self.uart_command_submit.emit(f"EV 0 heat_cf {params['fiber']['temp']}\n")
        self.print_test_rsi_command_submit.emit("RESET")
        self._set_print_test_controls_enabled(self._print_test_seen_correction)
        self._set_print_test_status(
            f"已进入测试准备：固定树脂喷头，矩阵 {params['resin']['line_count']} 条线，"
            "UART 挤出倍率已设为 1.0。请启动/准备 KUKA RSI，"
            "收到修正量后可做 Z 微调。",
            "#1b6e3c",
        )

    def _set_print_test_controls_enabled(self, enabled):
        base_ready = bool(enabled and not self._print_test_busy)
        for btn in getattr(self, "_test_z_buttons", []):
            btn.setEnabled(base_ready)
        if hasattr(self, "_test_resin_z_comp_input"):
            self._test_resin_z_comp_input.setEnabled(base_ready)
        if hasattr(self, "_btn_test_confirm_height"):
            self._btn_test_confirm_height.setEnabled(base_ready)
        if not hasattr(self, "_btn_test_confirm_resin_height"):
            return
        self._btn_test_confirm_resin_height.setEnabled(base_ready)
        self._btn_test_print_resin.setEnabled(
            base_ready and self._print_test_resin_height_confirmed
        )
        self._btn_test_continue_fiber.setEnabled(
            base_ready and self._print_test_resin_height_confirmed
        )
        fiber_ready = base_ready and self.current_tool_id() == 1
        for inp in (
            self._test_fiber_x_comp_input,
            self._test_fiber_y_comp_input,
            self._test_fiber_z_comp_input,
        ):
            inp.setEnabled(fiber_ready)
        fiber_offset_confirm_ready = fiber_ready and not self._print_test_fiber_offset_initial_sent
        self._btn_test_confirm_fiber_offset.setEnabled(fiber_offset_confirm_ready)
        fiber_nudge_ready = fiber_ready
        self._btn_test_send_fiber_offset_nudge.setEnabled(fiber_nudge_ready)
        for btn in getattr(self, "_fiber_offset_nudge_buttons", []):
            btn.setEnabled(fiber_nudge_ready)
        fiber_action_ready = base_ready and self._print_test_fiber_confirmed
        self._btn_test_print_fiber.setEnabled(fiber_action_ready)
        self._btn_test_print_composite.setEnabled(fiber_action_ready)
        self._btn_test_cut.setEnabled(fiber_ready)

    def _on_current_correction(self, msg):
        self._append_diagnostic("rsi", "current_correction", {
            "seq": int(getattr(msg, "seq", 0)),
            "tool_id": int(getattr(msg, "tool_id", 0)),
            "x": float(msg.x),
            "y": float(msg.y),
            "z": float(msg.z),
            "a": float(msg.a),
            "b": float(msg.b),
            "c": float(msg.c),
            "e": float(getattr(msg, "e", 0.0)),
        })
        self._print_test_current_correction = (msg.x, msg.y, msg.z, msg.a, msg.b, msg.c)
        self._print_test_seen_correction = True
        self._test_correction_label.setText(
            "RSI 修正量: "
            f"X={msg.x:.3f} Y={msg.y:.3f} Z={msg.z:.3f} "
            f"A={msg.a:.3f} B={msg.b:.3f} C={msg.c:.3f}"
        )
        if self._print_test_params is not None and not self._print_test_busy:
            self._set_print_test_controls_enabled(True)
        if self._print_test_busy and self._print_test_target is not None:
            target = self._print_test_target
            err = max(abs(v - t)
                      for v, t in zip(self._print_test_current_correction[:6], target[:6]))
            if err <= 0.03:
                self._print_test_busy = False
                self._print_test_target = None
                if self._print_test_pending_after_zero == "tool_change_cf":
                    self._print_test_pending_after_zero = None
                    if self._print_test_pending_after_tool_change is None:
                        self._print_test_pending_after_tool_change = "adjust_fiber_offset"
                    self._print_test_waiting_for_tool = _PRINT_TEST_FIBER_TOOL_ID
                    self._print_test_requested_target_tool = _PRINT_TEST_FIBER_TOOL_ID
                    self._set_print_test_controls_enabled(False)
                    self._send_print_test_head_prepare("fiber")
                    self._set_print_test_status(
                        "已到达纤维换头安全位置，正在切换纤维头并确认风扇...", "#1b6e3c"
                    )
                    return
                self._set_print_test_controls_enabled(True)
                self._set_print_test_status("测试动作已到达目标，RSI 正在保持最后一帧。", "#1b6e3c")

    def _on_print_test_z(self, delta_z):
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        start = self._print_test_current_correction
        target_z = start[2] + float(delta_z)
        target = (start[0], start[1], target_z, start[3], start[4], start[5])
        self._run_print_test_job("z", start, target_z=target_z, target_pose=target)

    def _on_print_test_confirm_height(self):
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        start = self._print_test_current_correction
        target = self._print_test_matrix_target(start, self._print_test_params, "resin")
        self._run_print_test_job("resin_matrix", start, target_pose=target)

    def _on_print_test_confirm_resin_height(self):
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        try:
            self._save_current_head_calibration()
        except Exception as exc:
            self._set_print_test_status(f"标定保存失败: {exc}", "#b42318")
            return
        self._print_test_resin_height_confirmed = True
        self._print_test_resin_z_floor = float(self._print_test_current_correction[2])
        self._set_print_test_controls_enabled(self._print_test_seen_correction)
        self._set_print_test_status(
            f"树脂打印高度已确认，Z 警戒线 {self._print_test_resin_z_floor:.3f} mm。",
            "#1b6e3c",
        )

    def _on_print_test_continue_fiber(self):
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        if not self._print_test_resin_height_confirmed:
            self._set_print_test_status("请先确认树脂打印高度。", "#b42318")
            return
        if self._print_test_busy:
            self._set_print_test_status("上一段测试动作尚未完成。", "#b42318")
            return
        reply = _ask_yes_no(
            self,
            "切换纤维喷头",
            "将先规划回 RSI 全 0 correction，到位后再切换到纤维喷头。\n\n"
            "确认继续？",
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self._request_print_test_fiber_tool_from_safe_position("adjust_fiber_offset")

    def _on_print_test_print_resin(self):
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        if not self._print_test_resin_height_confirmed:
            self._set_print_test_status("请先确认树脂打印高度。", "#b42318")
            return
        self._ensure_resin_tool_then_start_print_test_resin()

    def _start_print_test_resin_matrix(self):
        start = self._print_test_current_correction
        target = self._print_test_matrix_target(start, self._print_test_params, "resin")
        self._run_print_test_job("resin_matrix", start, target_pose=target)

    def _ensure_resin_tool_then_start_print_test_resin(self):
        # Resin printing always re-sends head preparation so stale tool/fan/heat
        # state cannot start a matrix without a fresh UART confirmation.
        if self._print_test_waiting_for_tool is not None:
            self._set_print_test_status("正在等待打印头切换完成。", "#b15e00")
            return
        self._request_print_test_resin_tool()

    def _request_print_test_resin_tool(self):
        self._print_test_waiting_for_tool = _PRINT_TEST_RESIN_TOOL_ID
        self._print_test_requested_target_tool = _PRINT_TEST_RESIN_TOOL_ID
        self._print_test_pending_after_tool_change = "print_resin_matrix"
        self._set_print_test_controls_enabled(False)
        self._send_print_test_head_prepare("resin")
        self._set_print_test_status(
            "已发送树脂头切换和风扇命令，等待 UART 工具和风扇状态确认...",
            "#b15e00",
        )

    def _on_print_test_nudge_fiber_offset(self, axis, delta):
        if self.current_tool_id() != 1:
            self._set_print_test_status("当前未使用纤维头，不能微调纤维偏置。", "#b42318")
            return
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        axis = str(axis).upper()
        if axis not in ("X", "Y", "Z"):
            self._set_print_test_status("未知纤维偏置轴。", "#b42318")
            return
        try:
            calibration = self._current_head_calibration_from_inputs()
            values = {
                "X": calibration.fiber_x_print_compensation_mm,
                "Y": calibration.fiber_y_print_compensation_mm,
                "Z": calibration.fiber_z_print_compensation_mm,
            }
            values[axis] += float(delta)
            self._test_fiber_x_comp_input.setText(f"{values['X']:.3f}")
            self._test_fiber_y_comp_input.setText(f"{values['Y']:.3f}")
            self._test_fiber_z_comp_input.setText(f"{values['Z']:.3f}")
        except Exception as exc:
            self._set_print_test_status(f"纤维偏置微调失败: {exc}", "#b42318")
            return
        self._print_test_fiber_confirmed = False
        self._set_print_test_controls_enabled(self._print_test_seen_correction)
        self._set_print_test_status("纤维偏置已微调，点击“下发微调”只会发送相对上次下发值的增量。", "#1b6e3c")

    def _on_print_test_send_fiber_offset_nudge(self):
        if self.current_tool_id() != 1:
            self._set_print_test_status("当前未使用纤维头，不能下发纤维偏置微调。", "#b42318")
            return
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        if self._print_test_busy:
            self._set_print_test_status("上一段测试动作尚未完成。", "#b42318")
            return
        if self._print_test_last_sent_fiber_offset is None:
            self._set_print_test_status("请先确认偏置并下发。", "#b42318")
            return
        try:
            calibration = self._current_head_calibration_from_inputs()
        except Exception as exc:
            self._set_print_test_status(f"纤维偏置无效: {exc}", "#b42318")
            return
        last_x, last_y, last_z = self._print_test_last_sent_fiber_offset
        delta_x = calibration.fiber_x_print_compensation_mm - last_x
        delta_y = calibration.fiber_y_print_compensation_mm - last_y
        delta_z = calibration.fiber_z_print_compensation_mm - last_z

        if max(abs(delta_x), abs(delta_y), abs(delta_z)) <= 1e-9:
            self._set_print_test_status("当前没有待下发的纤维偏置增量。", "#b15e00")
            return
        start = self._print_test_current_correction
        target = (
            start[0] + delta_x,
            start[1] + delta_y,
            start[2] + delta_z,
            start[3],
            start[4],
            start[5],
        )
        self._print_test_fiber_confirmed = False
        self._print_test_last_sent_fiber_offset = (
            calibration.fiber_x_print_compensation_mm,
            calibration.fiber_y_print_compensation_mm,
            calibration.fiber_z_print_compensation_mm,
        )
        self._set_print_test_controls_enabled(self._print_test_seen_correction)
        self._run_print_test_job("travel", start, target_pose=target)
        self._set_print_test_status(
            f"已下发纤维偏置微调增量: X {delta_x:+.3f} mm, Y {delta_y:+.3f} mm, Z {delta_z:+.3f} mm。",
            "#1b6e3c",
        )

    def _on_print_test_confirm_fiber_offset(self):
        if self.current_tool_id() != 1:
            self._set_print_test_status("当前未使用纤维头，不能确认纤维偏置。", "#b42318")
            return
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        if self._print_test_fiber_offset_initial_sent:
            self._set_print_test_status("纤维初始偏置已下发，请使用“下发微调”。", "#b15e00")
            return
        if self._print_test_busy:
            self._set_print_test_status("上一段测试动作尚未完成。", "#b42318")
            return
        try:
            calibration = self._current_head_calibration_from_inputs()
            self._head_calibration = calibration
            save_head_calibration(calibration, path=DEFAULT_HEAD_CALIBRATION_PATH)
        except Exception as exc:
            self._set_print_test_status(f"标定保存失败: {exc}", "#b42318")
            return
        start = self._print_test_current_correction
        target = (
            start[0] + calibration.fiber_x_print_compensation_mm,
            start[1] + calibration.fiber_y_print_compensation_mm,
            start[2]
            + calibration.resin_z_print_compensation_mm
            + calibration.fiber_z_print_compensation_mm,
            start[3],
            start[4],
            start[5],
        )
        self._print_test_pending_initial_fiber_offset = (
            calibration.fiber_x_print_compensation_mm,
            calibration.fiber_y_print_compensation_mm,
            calibration.fiber_z_print_compensation_mm,
        )
        self._set_print_test_controls_enabled(self._print_test_seen_correction)
        self._run_print_test_job(
            "travel",
            start,
            target_pose=target
        )
        self._set_print_test_status("纤维偏置已确认、保存并下发；后续微调将只发送增量。", "#1b6e3c")

    def _mark_print_test_fiber_offset_initial_sent(self):
        pending = self._print_test_pending_initial_fiber_offset
        if pending is None:
            return
        self._print_test_fiber_offset_initial_sent = True
        self._print_test_fiber_confirmed = True
        self._print_test_last_sent_fiber_offset = pending
        self._print_test_pending_initial_fiber_offset = None

    def _on_print_test_print_fiber(self):
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        if not self._print_test_fiber_confirmed:
            self._set_print_test_status("请先确认纤维头偏置。", "#b42318")
            return
        self._ensure_fiber_tool_then_continue(
            "print_fiber_matrix", self._start_print_test_fiber_matrix
        )

    def _start_print_test_fiber_matrix(self):
        start = self._print_test_current_correction
        target = self._print_test_matrix_target(start, self._print_test_params, "fiber")
        self._run_print_test_job("fiber_matrix", start, target_pose=target)

    def _on_print_test_print_composite(self):
        if self._print_test_params is None:
            self._set_print_test_status("请先进入测试准备。", "#b42318")
            return
        if not self._print_test_seen_correction:
            self._set_print_test_status("尚未收到 KUKA/RSI 首帧修正量。", "#b42318")
            return
        if not self._print_test_fiber_confirmed:
            self._set_print_test_status("请先确认纤维头偏置。", "#b42318")
            return
        self._ensure_fiber_tool_then_continue(
            "print_composite_matrix", self._start_print_test_composite_matrix
        )

    def _start_print_test_composite_matrix(self):
        start = self._print_test_current_correction
        target = self._print_test_matrix_target(
            start, self._print_test_params, "fiber", include_resin_to_fiber_delta=True
        )
        self._run_print_test_job("composite_matrix", start, target_pose=target)

    def _ensure_fiber_tool_then_continue(self, pending_after_tool_change, continuation):
        # Fiber actions first travel to the test safe position when changing from
        # another tool; if already on fiber, still re-send head preparation.
        if self._print_test_waiting_for_tool is not None:
            self._set_print_test_status("正在等待打印头切换完成。", "#b15e00")
            return
        current_tool = self.current_tool_id()
        if current_tool == _PRINT_TEST_FIBER_TOOL_ID:
            self._request_print_test_fiber_tool(pending_after_tool_change)
            return
        self._request_print_test_fiber_tool_from_safe_position(pending_after_tool_change)

    def _request_print_test_fiber_tool(self, pending_after_tool_change):
        self._print_test_waiting_for_tool = _PRINT_TEST_FIBER_TOOL_ID
        self._print_test_requested_target_tool = _PRINT_TEST_FIBER_TOOL_ID
        self._print_test_pending_after_tool_change = pending_after_tool_change
        self._set_print_test_controls_enabled(False)
        self._send_print_test_head_prepare("fiber")
        self._set_print_test_status(
            "已发送纤维头切换和风扇命令，等待 UART 工具和风扇状态确认...",
            "#b15e00",
        )

    def _request_print_test_fiber_tool_from_safe_position(self, pending_after_tool_change):
        self._print_test_pending_after_tool_change = pending_after_tool_change
        self._print_test_pending_after_zero = "tool_change_cf"
        self._run_print_test_job(
            "travel",
            self._print_test_current_correction,
            target_pose=_PRINT_TEST_ZERO_CORRECTION,
        )

    def _send_print_test_head_prepare(self, head_key):
        if head_key == "resin":
            self.uart_command_submit.emit("EV 0 tool_change_resin 2\n")
            self.uart_command_submit.emit("EV 0 fan_resin 1\n")
        elif head_key == "fiber":
            self.uart_command_submit.emit("EV 0 tool_change_cf 1\n")
            self.uart_command_submit.emit("EV 0 fan_cf 1\n")

    def _print_test_head_ready(self, head_key, ps=None):
        status = ps if ps is not None else self._print_test_printhead_status
        if status is None or self._print_test_params is None:
            return False, "尚未收到打印头状态。"
        if head_key == "resin":
            if int(status.current_tool) != _PRINT_TEST_RESIN_TOOL_ID:
                return False, "等待树脂头切换完成..."
            if not bool(status.fan_ok_resin):
                return False, "等待树脂头风扇开启..."
        elif head_key == "fiber":
            if int(status.current_tool) != _PRINT_TEST_FIBER_TOOL_ID:
                return False, "等待纤维头切换完成..."
            if not bool(status.fan_ok_cf):
                return False, "等待纤维头风扇开启..."
        else:
            return False, "未知打印头。"
        return True, ""

    def _print_test_matrix_target(
        self, start, params, head_key, *, include_resin_to_fiber_delta=False
    ):
        head_params = params[head_key]
        global_params = params["global"]
        line_count = int(head_params["line_count"])
        layer_heights = head_params["layer_heights"]
        last_layer_height = float(layer_heights[-1]) if layer_heights else 0.0
        line_length = float(global_params["line_length"])
        y_spacing = float(global_params["y_spacing"])
        safe_lift = float(global_params["tool_change_safe_lift"])
        base = tuple(float(v) for v in start[:6])
        resin_surface_height = 0.0
        if include_resin_to_fiber_delta:
            resin_layers = params.get("resin", {}).get("layer_heights", [])
            resin_surface_height = float(resin_layers[0]) if resin_layers else 0.0
        final_x = base[0] + line_length if line_count % 2 == 1 else base[0]
        final_y_steps = line_count if head_key == "resin" else max(0, line_count - 1)
        return (
            final_x,
            base[1] + final_y_steps * y_spacing,
            base[2] + resin_surface_height + last_layer_height + safe_lift,
            base[3],
            base[4],
            base[5],
        )

    def _on_print_test_cut(self):
        if self.current_tool_id() != 1:
            self._set_print_test_status("当前未使用纤维头，不能剪切。", "#b42318")
            return
        self.uart_command_submit.emit("EV 0 cut_cf\n")
        self._set_print_test_status("已发送剪切命令。", "#1b6e3c")

    def _print_test_npz_parts(self, npz_path):
        part_path = Path(npz_path)
        if part_path.is_file():
            return [part_path]
        if part_path.suffix == ".npz":
            base = part_path.with_suffix("")
            return sorted(part_path.parent.glob(base.name + "_part*.npz"))
        return []

    def _validate_print_test_npz_z_floor(self, npz_path):
        floor = self._print_test_resin_z_floor
        if floor is None:
            return True
        parts = self._print_test_npz_parts(npz_path)
        if not parts:
            self.print_test_status.emit(
                f"测试动作生成失败: 未找到 NPZ 文件 {npz_path}", "#b42318"
            )
            return False
        try:
            import numpy as np

            min_z = None
            for part_path in parts:
                with np.load(str(part_path)) as data:
                    part_min_z = float(np.min(data["z"]))
                min_z = part_min_z if min_z is None else min(min_z, part_min_z)
        except Exception as exc:
            self.print_test_status.emit(
                f"测试动作生成失败: NPZ Z 安全检查失败: {exc}", "#b42318"
            )
            return False
        if min_z < floor - 1e-6:
            self.print_test_status.emit(
                f"拒绝下发测试 NPZ: 最低 Z={min_z:.3f} mm，"
                f"低于已确认树脂高度警戒线 {floor:.3f} mm。",
                "#b42318",
            )
            return False
        return True

    def _run_print_test_job(self, job_type, start_pose, target_pose, target_z=None):
        if self._print_test_busy:
            self._set_print_test_status("上一段测试动作尚未完成。", "#b42318")
            return
        params = self._print_test_params or {}
        global_params = params.get("global", {})
        speed = float(global_params.get("speed", 10.0))
        self._print_test_busy = True
        self._print_test_target = target_pose
        self._set_print_test_controls_enabled(False)
        self._set_print_test_status("正在生成临时测试 NPZ...", "#b15e00")
        if job_type in ("resin_matrix", "fiber_matrix", "composite_matrix"):
            try:
                self._head_calibration = self._current_head_calibration_from_inputs()
                save_head_calibration(
                    self._head_calibration, path=DEFAULT_HEAD_CALIBRATION_PATH
                )
            except Exception as exc:
                self._print_test_busy = False
                self._print_test_target = None
                self._print_test_pending_initial_fiber_offset = None
                self._set_print_test_controls_enabled(True)
                self._set_print_test_status(f"标定保存失败: {exc}", "#b42318")
                return

        def _worker():
            try:
                from gcode_planner.gcode_parser import parse_gcode_lines
                from path_processing_core.npz_exporter import export_npz
                from gcode_planner.print_test_generator import (
                    format_gcode,
                    generate_composite_test_matrix_gcode,
                    generate_head_test_matrix_gcode,
                    generate_pose_adjust_gcode,
                    generate_test_matrix_gcode,
                    generate_z_adjust_gcode,
                )
                # Keep all generated test-mode artifacts under one temporary subtree.
                base_dir = str(_DEFAULT_DATA_ROOT / "print_test" / "tmp")
                stamp = time.strftime("%Y%m%d_%H%M%S")
                job_dir = os.path.join(base_dir, stamp)
                os.makedirs(job_dir, exist_ok=True)
                if job_type == "z":
                    lines = generate_z_adjust_gcode(
                        start_pose=start_pose,
                        target_z=target_z,
                        speed_mm_s=speed,
                    )
                    stem = "z_adjust"
                elif job_type == "travel":
                    lines = generate_pose_adjust_gcode(
                        start_pose=start_pose,
                        target_pose=target_pose,
                        speed_mm_s=min(speed, 10.0),
                    )
                    stem = "travel"
                elif job_type == "line":
                    resin = params["resin"]
                    lines = generate_test_matrix_gcode(
                        start_pose=start_pose,
                        layer_heights_mm=resin["layer_heights"],
                        extrusion_scales=resin["scales"],
                        speed_mm_s=speed,
                        line_length_mm=float(global_params["line_length"]),
                        y_spacing_mm=float(global_params["y_spacing"]),
                        finish_lift_mm=float(global_params["tool_change_safe_lift"]),
                        prime_length_mm=float(resin["prime_length"]),
                        retract_length_mm=float(resin["retract_length"]),
                        prime_speed_mm_s=float(resin["prime_speed"]),
                        retract_speed_mm_s=float(resin["retract_speed"]),
                        lift_between_lines=False,
                    )
                    stem = "test_matrix"
                elif job_type == "resin_matrix":
                    resin = params["resin"]
                    lines = generate_head_test_matrix_gcode(
                        start_pose=start_pose,
                        tool="resin",
                        layer_heights_mm=resin["layer_heights"],
                        extrusion_scales=resin["scales"],
                        speed_mm_s=speed,
                        line_length_mm=float(global_params["line_length"]),
                        y_spacing_mm=float(global_params["y_spacing"]),
                        finish_lift_mm=float(global_params["tool_change_safe_lift"]),
                        prime_length_mm=float(resin["prime_length"]),
                        retract_length_mm=float(resin["retract_length"]),
                        prime_speed_mm_s=float(resin["prime_speed"]),
                        retract_speed_mm_s=float(resin["retract_speed"]),
                    )
                    stem = "resin_matrix"
                elif job_type == "fiber_matrix":
                    fiber = params["fiber"]
                    lines = generate_head_test_matrix_gcode(
                        start_pose=start_pose,
                        tool="fiber",
                        layer_heights_mm=fiber["layer_heights"],
                        extrusion_scales=fiber["scales"],
                        speed_mm_s=speed,
                        line_length_mm=float(global_params["line_length"]),
                        y_spacing_mm=float(global_params["y_spacing"]),
                        finish_lift_mm=float(global_params["tool_change_safe_lift"]),
                        prime_length_mm=float(fiber["prime_length"]),
                        retract_length_mm=float(fiber["retract_length"]),
                        prime_speed_mm_s=float(fiber["prime_speed"]),
                        retract_speed_mm_s=float(fiber["retract_speed"]),
                    )
                    stem = "fiber_matrix"
                elif job_type == "composite_matrix":
                    resin = params["resin"]
                    fiber = params["fiber"]
                    lines = generate_composite_test_matrix_gcode(
                        start_pose=start_pose,
                        resin_layer_heights_mm=resin["layer_heights"],
                        resin_extrusion_scales=resin["scales"],
                        fiber_layer_heights_mm=fiber["layer_heights"],
                        fiber_extrusion_scales=fiber["scales"],
                        speed_mm_s=speed,
                        line_length_mm=float(global_params["line_length"]),
                        y_spacing_mm=float(global_params["y_spacing"]),
                        finish_lift_mm=float(global_params["tool_change_safe_lift"]),
                        prime_length_mm=float(resin["prime_length"]),
                        retract_length_mm=float(resin["retract_length"]),
                        prime_speed_mm_s=float(resin["prime_speed"]),
                        retract_speed_mm_s=float(resin["retract_speed"]),
                        fiber_prime_length_mm=float(fiber["prime_length"]),
                        fiber_retract_length_mm=float(fiber["retract_length"]),
                        fiber_prime_speed_mm_s=float(fiber["prime_speed"]),
                        fiber_retract_speed_mm_s=float(fiber["retract_speed"]),
                        calibration=self._head_calibration,
                        tool_change_safe_lift_mm=float(
                            global_params["tool_change_safe_lift"]
                        ),
                    )
                    stem = "composite_matrix"
                else:
                    raise ValueError(f"未知测试动作类型: {job_type}")
                gcode_path = os.path.join(job_dir, f"{stem}.gcode")
                npz_path = os.path.join(job_dir, f"{stem}.npz")
                with open(gcode_path, "w", encoding="utf-8") as f:
                    f.write(format_gcode(lines))
                parsed = parse_gcode_lines(lines)
                export_npz(
                    parsed,
                    npz_path,
                    dt=0.004,
                    chunk_size=5000000,
                    default_feed_mm_s=speed,
                    tool_offset=(0.0, 0.0, 0.0),
                    enable_extrude_wait=True,
                    initial_tool_id=(
                        _PRINT_TEST_FIBER_TOOL_ID
                        if job_type in ("fiber_matrix", "composite_matrix")
                        else _PRINT_TEST_RESIN_TOOL_ID
                    ),
                )
                if not self._validate_print_test_npz_z_floor(npz_path):
                    self._print_test_busy = False
                    self._print_test_target = None
                    self._print_test_pending_after_zero = None
                    self._print_test_pending_initial_fiber_offset = None
                    self.print_test_controls_enabled.emit(True)
                    return
                self.print_test_rsi_command_submit.emit("RESET")
                time.sleep(0.05)
                self.print_test_load_npz_submit.emit(npz_path)
                self.print_test_status.emit(f"已下发测试动作: {npz_path}", "#1b6e3c")
            except Exception as exc:
                self._print_test_busy = False
                self._print_test_target = None
                self._print_test_pending_after_zero = None
                self._print_test_pending_initial_fiber_offset = None
                self.print_test_controls_enabled.emit(True)
                self.print_test_status.emit(f"测试动作生成失败: {exc}", "#b42318")

        threading.Thread(target=_worker, daemon=True).start()

    # ---- Offset persistence ----

    def _on_offset_changed(self, _value=None):
        x = self._offset_spins["X"].value()
        y = self._offset_spins["Y"].value()
        fiber_z = self._fiber_z_print_comp_spin.value()
        resin_z = self.current_resin_z_print_compensation()
        try:
            _save_offset_config(x, y, fiber_z, resin_z)
            save_head_calibration(
                HeadCalibration(
                    resin_z_print_compensation_mm=resin_z,
                    fiber_x_print_compensation_mm=x,
                    fiber_y_print_compensation_mm=y,
                    fiber_z_print_compensation_mm=fiber_z,
                ),
                path=DEFAULT_HEAD_CALIBRATION_PATH,
            )
            self._offset_status.setText(
                f"已保存: X={x:.2f}  Y={y:.2f}  纤维Z偏置={fiber_z:.2f}  树脂Z补偿={resin_z:.2f}")
            self._offset_status.setStyleSheet("color: #1b6e3c;")
        except Exception as exc:
            self._offset_status.setText(f"保存失败: {exc}")
            self._offset_status.setStyleSheet("color: #b42318;")

    def get_tool_offset(self):
        return (
            self._offset_spins["X"].value(),
            self._offset_spins["Y"].value(),
            self._fiber_z_print_comp_spin.value(),
        )

    def current_resin_z_print_compensation(self):
        return self._resin_z_print_comp_spin.value()

    def current_fiber_z_print_compensation(self):
        return self._fiber_z_print_comp_spin.value()

    # ---- GCode Export ----

    def _on_browse_gcode(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择源文件",
            _dialog_start_dir(self._gcode_path_input.text(), str(_DEFAULT_DATA_ROOT)),
            "GCode / NPZ Files (*.gcode *.gc *.g *.npz);;"
            "GCode Files (*.gcode *.gc *.g);;NPZ Files (*.npz);;All Files (*)",
        )
        if path:
            self._gcode_path_input.setText(path)

    def _on_gcode_path_changed(self, text):
        if not text:
            return
        base = os.path.splitext(os.path.basename(text))[0]
        data_root = _DEFAULT_NPZ_OUTPUT_DIR
        self._npz_out_input.setText(os.path.join(data_root, base, base + ".npz"))

    def _external_npz_process_params(self, planner_params=None):
        from dataclasses import replace

        from external_npz_preprocessor.process_params import (
            FiberProcessParams,
            ProcessParams,
            ResinProcessParams,
        )

        values = self._external_npz_inputs
        process_params = ProcessParams(
            resin=ResinProcessParams(
                layer_height_mm=values["resin_layer_height_mm"].value(),
                extrusion_scale=values["resin_extrusion_scale"].value(),
                feed_mm_s=values["resin_feed_mm_s"].value(),
                first_layer_feed_mm_s=values["first_layer_resin_feed_mm_s"].value(),
                temperature_c=values["resin_temperature_c"].value(),
                fan_enabled=values["resin_fan_enabled"].isChecked(),
                prime_length_mm=values["resin_prime_length_mm"].value(),
                prime_speed_mm_s=values["resin_prime_speed_mm_s"].value(),
                retract_length_mm=values["resin_retract_length_mm"].value(),
                retract_speed_mm_s=values["resin_retract_speed_mm_s"].value(),
            ),
            fiber=FiberProcessParams(
                layer_height_mm=values["fiber_layer_height_mm"].value(),
                extrusion_scale=values["fiber_extrusion_scale"].value(),
                feed_mm_s=values["fiber_feed_mm_s"].value(),
                first_layer_feed_mm_s=values["first_layer_fiber_feed_mm_s"].value(),
                start_accel_s=values["fiber_start_accel_s"].value(),
                temperature_c=values["fiber_temperature_c"].value(),
                fan_enabled=values["fiber_fan_enabled"].isChecked(),
                prime_length_mm=values["fiber_prime_length_mm"].value(),
                prime_speed_mm_s=values["fiber_prime_speed_mm_s"].value(),
                retract_length_mm=values["fiber_retract_length_mm"].value(),
                retract_speed_mm_s=values["fiber_retract_speed_mm_s"].value(),
            ),
            travel_feed_mm_s=values["travel_feed_mm_s"].value(),
            first_layer_travel_feed_mm_s=values["first_layer_travel_feed_mm_s"].value(),
            prime_settle_s=values["prime_settle_s"].value(),
            default_a=values["default_a"].value(),
            default_b=values["default_b"].value(),
            default_c=values["default_c"].value(),
            start_x_mm=values["start_x_mm"].value(),
            start_y_mm=values["start_y_mm"].value(),
            primeline_x_mm=values["primeline_x_mm"].value(),
            primeline_y_mm=values["primeline_y_mm"].value(),
            primeline_length_mm=values["primeline_length_mm"].value(),
            corner_angle_deg=values["corner_angle_deg"].value(),
            corner_retreat_ratio=values["corner_retreat_ratio"].value(),
            spline_max_error_mm=values["spline_max_error_mm"].value(),
            spline_max_angle_deg=values["spline_max_angle_deg"].value(),
            source_merge_distance_mm=values["source_merge_distance_mm"].value(),
            corner_retreat_max_mm=values["corner_retreat_max_mm"].value(),
            corner_blend_segments=int(values["corner_blend_segments"].value()),
        )
        if planner_params is None:
            return process_params
        return replace(process_params, dt=planner_params["dt"])

    def _on_save_external_npz_params(self):
        try:
            from external_npz_preprocessor.param_config import save_print_params

            path = save_print_params(self._external_npz_process_params())
            self._export_status.setText(f"已保存外部 NPZ 参数: {path}")
            self._export_status.setStyleSheet("color: #1b6e3c;")
        except Exception as exc:
            self._export_status.setText(f"保存外部 NPZ 参数失败: {exc}")
            self._export_status.setStyleSheet("color: #b42318;")

    def _on_export_npz(self):
        source_path = self._gcode_path_input.text().strip()
        if not source_path or not os.path.isfile(source_path):
            self._export_status.setText("请选择有效的 GCode 或约定格式 NPZ 文件。")
            self._export_status.setStyleSheet("color: #b42318;")
            return

        npz_out = self._npz_out_input.text().strip()
        if not npz_out:
            self._export_status.setText("请指定 NPZ 输出路径。")
            self._export_status.setStyleSheet("color: #b42318;")
            return

        source_ext = os.path.splitext(source_path)[1].lower()

        # Gather export params
        try:
            params = {
                "dt": float(self._planner_inputs["dt"].text()),
                "default_feed_mm_s": float(self._planner_inputs["default_feed_mm_s"].text()),
                "corner_angle_deg": float(self._planner_inputs["corner_angle_deg"].text()),
                "corner_retreat_ratio": float(self._planner_inputs["corner_retreat_ratio"].text()),
                "density": int(self._planner_inputs["density"].text()),
                "degree": int(self._planner_inputs["degree"].text()),
                "max_fit_points_per_segment": int(
                    self._planner_inputs["max_fit_points_per_segment"].text()
                ),
                "export_sleep_ms": int(self._planner_inputs["export_sleep_ms"].text()),
                "export_yield_every": int(self._planner_inputs["export_yield_every"].text()),
                "split_by_layer_type": self._planner_inputs["split_by_layer_type"].isChecked(),
                "plot_layer_xy": self._planner_inputs["plot_layer_xy"].isChecked(),
                "plot_stride": int(self._planner_inputs["plot_stride"].text()),
                "cut_lift_mm": float(self._planner_inputs["cut_lift_mm"].text()),
                "cut_wait_s": float(self._planner_inputs["cut_wait_s"].text()),
            }
            external_cut_lift_mm = params["cut_lift_mm"]
            external_cut_wait_s = params["cut_wait_s"]
            if source_ext == ".npz":
                external_process_params = self._external_npz_process_params(params)
                external_cut_lift_mm = (
                    self._external_npz_inputs["external_cut_lift_mm"].value()
                )
                external_cut_wait_s = (
                    self._external_npz_inputs["external_cut_wait_s"].value()
                )
            else:
                external_process_params = None
            self._last_export_split_by_layer_type = (
                params["split_by_layer_type"]
                and source_ext in (".gcode", ".gc", ".g")
            )
        except (ValueError, KeyError, ImportError) as exc:
            self._export_status.setText(f"无效的导出参数: {exc}")
            self._export_status.setStyleSheet("color: #b42318;")
            return

        offset = self.get_tool_offset()

        # Disable button, show progress
        self._btn_export_npz.setEnabled(False)
        self._export_progress.setVisible(True)
        self._export_progress.setRange(0, 100)
        self._export_progress.setValue(0)
        self._export_status.setText("导出中...")
        self._export_status.setStyleSheet("color: #b15e00;")

        def _worker():
            try:
                def progress_cb(ratio):
                    self.export_progress_val.emit(int(ratio * 100))

                out_dir = os.path.dirname(npz_out)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)

                if source_ext == ".npz":
                    self.export_progress.emit("读取约定格式 NPZ...")
                    from external_npz_preprocessor.export_runner import convert_external_npz

                    process_params = external_process_params
                    self.export_progress.emit("转换约定格式 NPZ 并导出系统 NPZ...")
                    stats = convert_external_npz(
                        source_path,
                        npz_out,
                        process_params,
                        progress_callback=progress_cb,
                        cut_lift_mm=external_cut_lift_mm,
                        cut_wait_s=external_cut_wait_s,
                    )
                elif source_ext in (".gcode", ".gc", ".g"):
                    self.export_progress.emit("读取 GCode...")
                    from gcode_planner.gcode_parser import load_gcode_lines, parse_gcode_lines
                    from gcode_planner.primeline import insert_resin_primeline
                    from path_processing_core.npz_exporter import export_npz

                    lines = load_gcode_lines(source_path)
                    self.export_progress.emit("解析 GCode...")
                    parsed = insert_resin_primeline(parse_gcode_lines(lines))
                    self.export_progress.emit(f"导出 NPZ（{len(parsed)} 条指令）...")
                    stats = export_npz(
                        parsed,
                        npz_out,
                        dt=params["dt"],
                        chunk_size=5000000,
                        default_feed_mm_s=params["default_feed_mm_s"],
                        corner_angle_deg=params["corner_angle_deg"],
                        corner_retreat_ratio=params["corner_retreat_ratio"],
                        density=params["density"],
                        degree=params["degree"],
                        max_fit_points_per_segment=params["max_fit_points_per_segment"],
                        export_sleep_ms=params["export_sleep_ms"],
                        export_yield_every=params["export_yield_every"],
                        tool_offset=offset,
                        progress_callback=progress_cb,
                        split_by_layer_type=params["split_by_layer_type"],
                        plot_layer_xy=params["plot_layer_xy"],
                        plot_stride=params["plot_stride"],
                        enable_extrude_wait=True,
                        resin_z_print_compensation_mm=self.current_resin_z_print_compensation(),
                        cut_lift_mm=params["cut_lift_mm"],
                        cut_wait_s=params["cut_wait_s"],
                    )
                else:
                    raise ValueError(f"不支持的源文件格式: {source_ext or '(无扩展名)'}")
                rows = stats.get("rows", 0)
                parts = stats.get("parts", 0)
                total_s = stats.get("total_s", 0.0)
                msg = (
                    f"完成: {rows} 行, {parts} 分块, {total_s:.1f}秒\n"
                    f"偏移: ({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f})"
                )
                self.export_finished.emit(True, msg)
            except Exception as exc:
                self.export_finished.emit(False, str(exc))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _on_export_progress(self, text):
        self._export_status.setText(text)
        self._export_status.setStyleSheet("color: #b15e00;")

    def _on_export_progress_val(self, val):
        self._export_progress.setValue(val)

    def _on_export_finished(self, success, message):
        self._btn_export_npz.setEnabled(True)
        self._export_progress.setVisible(False)
        if success:
            self._export_status.setText(message)
            self._export_status.setStyleSheet("color: #1b6e3c;")
            npz_path = self._npz_out_input.text().strip()
            if npz_path:
                preview_root = _npz_preview_root_from_path(npz_path)
                self._last_npz_dir = preview_root
                self._selected_npz_dir = preview_root
                self._selected_npz_launch_path = (
                    _normalize_npz_launch_path(npz_path)
                    if not getattr(self, "_last_export_split_by_layer_type", False)
                    else None
                )
                self._selected_npz_dir_input.setText(preview_root or npz_path)
                self._btn_view_layers.setEnabled(bool(preview_root))
                self._btn_view_vtk_paths.setEnabled(bool(preview_root))
                self._export_status.setText(
                    f"{message}\n三维预览入口: {preview_root or npz_path}"
                )
        else:
            self._export_status.setText(f"导出失败: {message}")
            self._export_status.setStyleSheet("color: #b42318;")

    def _on_view_layers(self):
        if not self._last_npz_dir or not os.path.isdir(self._last_npz_dir):
            self._export_status.setText("未找到 NPZ 导出目录。")
            self._export_status.setStyleSheet("color: #b42318;")
            return
        dlg = _LayerViewerDialog(self._last_npz_dir, self)
        dlg.exec_()

    def _on_view_vtk_paths(self):
        preview_root = self._last_npz_dir
        if not preview_root or not os.path.exists(preview_root):
            self._export_status.setText("未找到三维预览入口。")
            self._export_status.setStyleSheet("color: #b42318;")
            return
        try:
            from my_project_ui.vtk_path_preview import VtkPathPreviewDialog

            dlg = VtkPathPreviewDialog(preview_root, self)
            dlg.exec_()
        except Exception as exc:
            self._export_status.setText(f"三维路径预览启动失败: {exc}")
            self._export_status.setStyleSheet("color: #b42318;")

    def _npz_launch_relation_text(self):
        return (
            "启动设置中与 NPZ 直接相关的是 npz_path；UI 会把所选 NPZ 文件或 _part 分片入口传给它。\n"
            "npz_preload_chunks、queue_low/high、traj_prefill、traj_low/high 只影响加载、预取和队列阈值，"
            "不会改变 NPZ 文件内容。"
        )

    def _changed_npz_related_launch_params(self):
        changed = []
        launch_params = getattr(self, "_launch_params", {})
        for name in _NPZ_RELATED_LAUNCH_PARAMS:
            if name == "npz_path":
                continue
            default = _LAUNCH_DEFAULTS.get(name, "")
            current = launch_params.get(name, default)
            if str(current) != str(default):
                changed.append((name, current, default))
        return changed

    def _npz_dir_selection_warnings(self, saved_offset, saved_resin_z_print_compensation):
        current_offset = self.get_tool_offset()
        current_resin_z = self.current_resin_z_print_compensation()
        warnings = []

        if saved_offset is None:
            warnings.append("未找到或无法读取纤维头偏置 sidecar（*.offset.json），无法确认该 NPZ 是否使用了当前界面偏置值。")
        else:
            mismatch = any(
                abs(a - b) > _NPZ_OFFSET_TOLERANCE_MM
                for a, b in zip(saved_offset, current_offset)
            )
            if mismatch:
                warnings.append("NPZ 中的纤维头偏置与当前界面设置不一致。")

        if saved_resin_z_print_compensation is None:
            warnings.append("未找到或无法读取树脂 Z 打印补偿，无法确认该 NPZ 是否使用了当前界面补偿值。")
        elif abs(saved_resin_z_print_compensation - current_resin_z) > _NPZ_OFFSET_TOLERANCE_MM:
            warnings.append("NPZ 中的树脂 Z 打印补偿与当前界面设置不一致。")

        changed_launch_params = self._changed_npz_related_launch_params()
        if changed_launch_params:
            details = ", ".join(
                f"{name}={current}（默认 {default}）"
                for name, current, default in changed_launch_params
            )
            warnings.append(f"NPZ 相关启动参数已修改: {details}")

        return warnings

    def _confirm_npz_dir_selection(
            self,
            npz_file,
            launch_path,
            saved_offset,
            saved_resin_z_print_compensation,
            offset_file,
            warnings):
        current_offset = self.get_tool_offset()
        current_resin_z = self.current_resin_z_print_compensation()
        detail = [
            f"NPZ 文件: {npz_file}",
            f"启动将使用: {launch_path}",
            f"当前界面纤维头偏置: {_format_tool_offset(current_offset)}",
            f"当前界面树脂 Z 打印补偿: {current_resin_z:.2f} mm",
        ]

        if saved_offset is None:
            detail.append("NPZ 保存的纤维头偏置: 未知")
        else:
            detail.append(f"NPZ 保存的纤维头偏置: {_format_tool_offset(saved_offset)}")
            if offset_file:
                detail.append(f"偏移文件: {offset_file}")

        if saved_resin_z_print_compensation is None:
            detail.append("NPZ 保存的树脂 Z 打印补偿: 未知")
        else:
            detail.append(f"NPZ 保存的树脂 Z 打印补偿: {saved_resin_z_print_compensation:.2f} mm")

        detail.append("")
        detail.append("检测到以下相关参数变化/风险:")
        detail.extend(f"- {item}" for item in warnings)
        detail.append("")
        detail.append(self._npz_launch_relation_text())
        detail.append("")
        detail.append("是否将该文件作为本次启动使用的 NPZ 数据？")

        return _ask_yes_no(
            self,
            "NPZ 相关参数警告",
            "\n".join(detail),
            QtWidgets.QMessageBox.No,
        ) == QtWidgets.QMessageBox.Yes

    def _on_select_npz_dir(self):
        start_dir = _dialog_start_dir(
            self._selected_npz_launch_path or self._selected_npz_dir_input.text(),
            _DEFAULT_NPZ_OUTPUT_DIR,
        )
        npz_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择已导出的 NPZ 文件",
            start_dir,
            "NPZ Files (*.npz);;All Files (*)",
        )
        if not npz_file:
            return

        launch_path = _normalize_npz_launch_path(npz_file)
        if not launch_path:
            _show_warning(
                self,
                "NPZ 文件无效",
                "请选择可用于正式打印的 flat NPZ 或 _part 分片文件。",
            )
            return

        try:
            (
                saved_offset,
                saved_resin_z_print_compensation,
                offset_file,
            ) = _read_npz_export_metadata(launch_path)
        except Exception as exc:
            saved_offset, saved_resin_z_print_compensation, offset_file = None, None, None
            self._export_status.setText(f"读取 NPZ 偏移信息失败: {exc}")
            self._export_status.setStyleSheet("color: #b42318;")

        warnings = self._npz_dir_selection_warnings(saved_offset, saved_resin_z_print_compensation)
        if warnings and not self._confirm_npz_dir_selection(
                npz_file,
                launch_path,
                saved_offset,
                saved_resin_z_print_compensation,
                offset_file,
                warnings):
            return

        preview_root = _npz_preview_root_from_path(launch_path)
        layer_dir = preview_root or _npz_layer_dir_from_launch_path(launch_path)
        self._selected_npz_dir = layer_dir
        self._selected_npz_launch_path = launch_path
        self._last_npz_dir = layer_dir
        self._selected_npz_dir_input.setText(preview_root or npz_file)
        self._btn_view_layers.setEnabled(bool(layer_dir and os.path.exists(layer_dir)))
        self._btn_view_vtk_paths.setEnabled(bool(layer_dir and os.path.exists(layer_dir)))
        self._export_status.setText(
            f"已选择 NPZ 文件: {npz_file}\n三维预览入口: {preview_root or npz_file}"
        )
        self._export_status.setStyleSheet("color: #1b6e3c;")

    def _on_clear_npz_dir(self):
        self._selected_npz_dir = None
        self._selected_npz_launch_path = None
        self._selected_npz_dir_input.clear()
        self._btn_view_layers.setEnabled(False)
        self._btn_view_vtk_paths.setEnabled(False)
        self._export_status.setText("已清除手动选择的 NPZ 文件；启动时将按 GCode 自动匹配。")
        self._export_status.setStyleSheet("color: #1a73e8;")

    _STATE_COLORS = {
        "RUNNING": "#1b6e3c",
        "WAIT_HEARTBEAT": "#b15e00",
        "HEARTBEAT_LOST": "#b42318",
        "PAUSE_REQUESTED": "#b15e00",
        "PAUSE_READY": "#1a73e8",
        "PAUSED": "#1a73e8",
        "ABORTING": "#b42318",
        "ABORTED": "#b42318",
    }

    def _update_control_buttons(self, state):
        if state in ("PAUSE_REQUESTED", "PAUSE_READY", "PAUSED"):
            self._btn_pause.setEnabled(False)
            self._btn_resume.setEnabled(True)
            self._btn_stop.setEnabled(True)
        elif state == "WAIT_HEARTBEAT":
            self._btn_pause.setEnabled(False)
            self._btn_resume.setEnabled(True)
            self._btn_stop.setEnabled(True)
        elif state in ("ABORTING", "ABORTED"):
            self._btn_pause.setEnabled(False)
            self._btn_resume.setEnabled(False)
            self._btn_stop.setEnabled(False)
        else:
            self._btn_pause.setEnabled(True)
            self._btn_resume.setEnabled(False)
            self._btn_stop.setEnabled(True)

    def _on_rsi_xml(self, xml_text):
        import time
        import re
        ipoc_match = re.search(r"<IPOC>([^<]*)</IPOC>", xml_text or "")
        self._append_diagnostic("rsi", "sent_xml", {
            "ipoc": ipoc_match.group(1) if ipoc_match else "",
            "xml": str(xml_text or ""),
        })
        xml_stripped = re.sub(r'<IPOC>[^<]*</IPOC>', '<IPOC>...</IPOC>', xml_text)
        if xml_stripped == self._rsi_log_last_xml:
            return
        self._rsi_log_last_xml = xml_stripped
        display_text = _format_rsi_xml_for_display(xml_text)
        self._rsi_log_latest_display = display_text
        self._btn_rsi_log_detail.setEnabled(True)
        t = time.localtime()
        ts = f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
        self._rsi_log_summary.setText(f"最近 XML: {ts}")
        self._rsi_log_text.appendPlainText(f"[{ts}] RSI sent XML")
        self._rsi_log_text.appendPlainText(display_text)

    def _show_rsi_log_detail(self):
        dialog = _LogDetailDialog("RSI XML 放大查看", self._rsi_log_latest_display, self)
        dialog.exec_()

    def _on_uart_log(self, line_text):
        import time
        t = time.localtime()
        ts = f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
        display_text = str(line_text or "")
        direction = display_text[:2] if display_text.startswith(("RX ", "TX ")) else ""
        payload = display_text[3:].lstrip() if display_text.startswith(
            ("RX ", "TX ")) else display_text
        self._append_diagnostic("uart", "raw", {
            "direction": direction,
            "line": display_text,
            "payload": payload,
        })
        if payload.startswith("EWARN"):
            return
        entry = f"[{ts}] {display_text}"
        self._uart_log_history.append(entry)
        self._uart_log_latest_display = display_text
        self._btn_uart_log_detail.setEnabled(True)
        self._uart_log_summary.setText(f"最近日志: {ts}")
        self._uart_log_text.appendPlainText(entry)

    def _on_export_diagnostic_log(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        # Keep diagnostic exports separate from temporary NPZ/GCode runs.
        out_dir = str(_DEFAULT_DATA_ROOT / "diagnostic_logs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"diagnostic_log_{stamp}.jsonl")
        records = self._diagnostic_log_history or [
            {
                "time_epoch": time.time(),
                "time": self._format_diagnostic_time(time.time()),
                "source": "ui",
                "kind": "empty",
                "detail": {"message": "当前测试没有诊断日志"},
            }
        ]
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                f.write("\n")
        self._set_print_test_status(f"诊断日志已导出: {path}", "#1b6e3c")

    def _show_uart_log_detail(self):
        detail_text = "\n".join(self._uart_log_history) or self._uart_log_latest_display
        dialog = _LogDetailDialog("UART 日志放大查看", detail_text, self)
        dialog.exec_()


class MyProjectUiPlugin(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self.setObjectName("MyProjectUiPlugin")

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = context.node
        self._shutting_down = False
        self._param_client = self._node.create_client(
            SetParameters, "/uart_node/set_parameters"
        )
        self._cmd_pub = self._node.create_publisher(
            StringMsg, "/system/command", 10
        )
        self._uart_cmd_pub = self._node.create_publisher(
            StringMsg, "/uart/manual_command", 10
        )
        self._print_test_rsi_cmd_pub = self._node.create_publisher(
            StringMsg, "/print_test/rsi_command", 10
        )
        self._print_test_load_pub = self._node.create_publisher(
            StringMsg, "/print_test/load_npz", 10
        )
        self._widget = _UiStatusWidget()
        self._widget.scale_submit.connect(self._on_scale_submit)
        self._widget.command_submit.connect(self._on_command_submit)
        self._widget.uart_command_submit.connect(self._on_uart_command_submit)
        self._widget.print_test_rsi_command_submit.connect(self._on_print_test_rsi_command)
        self._widget.print_test_load_npz_submit.connect(self._on_print_test_load_npz)
        self._widget.print_test_initial_fiber_offset_dispatched.connect(
            self._widget._mark_print_test_fiber_offset_initial_sent
        )
        context.add_widget(self._widget)

        # Launch state
        self._launch_params = dict(_LAUNCH_DEFAULTS)
        self._launch_params["gcode_path"] = ""
        self._widget._launch_params = self._launch_params
        self._launch_process = None
        self._pending_launch = False

        # GCode path starts empty; user can choose one when exporting or auto-matching NPZ.

        # Sync changes from main UI to launch params
        self._widget._gcode_path_input.textChanged.connect(self._on_gcode_path_changed_sync)
        # Connect export signals to launch flow
        self._widget.export_finished.connect(self._on_export_finished_launcher)

        self._widget._btn_launch_settings.clicked.connect(
            self._on_launch_settings
        )
        self._widget._btn_launch.clicked.connect(self._on_launch)
        self._widget._btn_stop_launch.clicked.connect(self._on_stop_launch)

        self._node.create_subscription(
            UiStatus, "/ui/status", self._on_status, 10
        )
        self._node.create_subscription(
            ExtruderLatencyStatus, "/extruder/latency_status", self._on_latency_status, 10
        )
        self._node.create_subscription(
            StringMsg, "/rsi/sent_xml", self._on_rsi_xml, 10
        )
        self._node.create_subscription(
            StringMsg, "/uart/raw", self._on_uart_log_msg, 10
        )
        self._node.create_subscription(
            TrajectoryPoint, "/rsi/current_correction", self._on_current_correction, 10
        )

        self._spin_timer = QtCore.QTimer(self._widget)
        self._spin_timer.timeout.connect(self._spin_once)
        self._spin_timer.start(50)

        self._launch_check_timer = QtCore.QTimer(self._widget)
        self._launch_check_timer.timeout.connect(self._check_launch_process)
        self._launch_check_timer.start(1000)

    def _on_status(self, msg: UiStatus):
        self._widget.status_received.emit(msg)

    def _on_latency_status(self, msg: ExtruderLatencyStatus):
        self._widget.latency_received.emit(msg)

    def _on_rsi_xml(self, msg: StringMsg):
        self._widget.rsi_xml_received.emit(msg.data)

    def _on_uart_log_msg(self, msg: StringMsg):
        self._widget.uart_log_received.emit(msg.data)

    def _on_current_correction(self, msg: TrajectoryPoint):
        self._widget.current_correction_received.emit(msg)

    def _publish_uart_manual_command(self, cmd: str):
        msg = StringMsg()
        msg.data = cmd
        self._uart_cmd_pub.publish(msg)

    def _send_current_tool_heat_off(self):
        tool_id = self._widget.current_tool_id()
        if tool_id == 1:
            commands = ["EV 0 heat_cf 0\n", "EV 0 heat_resin 0\n"]
        elif tool_id == 2:
            commands = ["EV 0 heat_resin 0\n", "EV 0 heat_cf 0\n"]
        else:
            commands = ["EV 0 heat_cf 0\n", "EV 0 heat_resin 0\n"]
        for command in commands:
            self._publish_uart_manual_command(command)

    def _on_command_submit(self, cmd: str):
        if cmd == "ABORT":
            self._send_current_tool_heat_off()
        msg = StringMsg()
        msg.data = cmd
        self._cmd_pub.publish(msg)

    def _on_uart_command_submit(self, cmd: str):
        self._publish_uart_manual_command(cmd)

    def _on_print_test_rsi_command(self, cmd: str):
        msg = StringMsg()
        msg.data = cmd
        self._print_test_rsi_cmd_pub.publish(msg)

    def _on_print_test_load_npz(self, path: str):
        if not self._widget._validate_print_test_npz_z_floor(path):
            self._widget._print_test_pending_initial_fiber_offset = None
            return
        msg = StringMsg()
        msg.data = path
        self._print_test_load_pub.publish(msg)
        if self._widget._print_test_pending_initial_fiber_offset is not None:
            self._widget.print_test_initial_fiber_offset_dispatched.emit()

    def _on_scale_submit(self, value: float):
        if not self._param_client.service_is_ready():
            self._widget.set_extrude_status("UART 参数服务未就绪。", "#b42318")
            return

        req = SetParameters.Request()
        req.parameters = [
            Parameter("extrude_scale", Parameter.Type.DOUBLE, value).to_parameter_msg()
        ]
        future = self._param_client.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
            except Exception:
                self._widget.set_extrude_scale(value, "提交失败。", "#b42318")
                return
            results = resp.results if resp is not None else []
            if results and all(r.successful for r in results):
                self._widget.set_extrude_scale(value, "已应用。", "#1b6e3c")
            else:
                reason = results[0].reason if results else "提交失败"
                self._widget.set_extrude_scale(value, reason or "提交失败。", "#b42318")

        future.add_done_callback(_done)

    def _spin_once(self):
        if self._shutting_down or not rclpy.ok():
            if self._spin_timer.isActive():
                self._spin_timer.stop()
            return
        try:
            rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            if self._spin_timer.isActive():
                self._spin_timer.stop()

    # ---- Launch control ----

    def _on_launch_settings(self):
        dialog = _LaunchSettingsDialog(self._launch_params, self._widget)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self._launch_params.update(dialog.get_params())

    def _on_gcode_path_changed_sync(self, text):
        self._launch_params["gcode_path"] = text

    def _on_export_finished_launcher(self, success, message):
        if not self._pending_launch:
            return
        self._pending_launch = False
        if success:
            self._on_launch()
        else:
            self._widget._btn_launch.setEnabled(True)
            self._widget._launch_status.setText("启动中止: GCode 导出失败。")
            self._widget._launch_status.setStyleSheet(
                "color: #b42318; font-weight: 700; font-size: 13px;"
            )

    def _auto_npz_launch_path(self, gcode_path):
        if not gcode_path:
            return None
        base = os.path.splitext(os.path.basename(gcode_path))[0]
        data_root = _DEFAULT_NPZ_OUTPUT_DIR

        npz_dir = os.path.join(data_root, base)
        launch_path = _resolve_npz_launch_path_from_dir(npz_dir)
        if launch_path:
            return launch_path

        npz_file = os.path.join(data_root, base + ".npz")
        if os.path.isfile(npz_file):
            return npz_file

        npz_part = os.path.join(data_root, base + "_part0000.npz")
        if os.path.isfile(npz_part):
            return npz_file

        return None

    def _current_npz_launch_path(self):
        selected = getattr(self._widget, "_selected_npz_launch_path", None)
        if selected:
            return selected, "selected"

        gcode_path = self._launch_params.get("gcode_path", "").strip()
        return self._auto_npz_launch_path(gcode_path), "auto"

    def _check_npz_and_offset_match(self, npz_launch_path):
        if not npz_launch_path:
            return False, "missing", None, None, None

        p = Path(npz_launch_path)
        is_manifest = p.name.endswith("_manifest.json")
        if is_manifest and not p.is_file():
            return False, "missing", None, None, None
        if p.suffix == ".npz" and not p.is_file():
            part_probe = p.with_name(p.stem + "_part0000.npz")
            if not part_probe.is_file():
                return False, "missing", None, None, None

        try:
            (
                saved_offset,
                saved_resin_z_print_compensation,
                offset_file,
            ) = _read_npz_export_metadata(npz_launch_path)
        except Exception:
            return False, "mismatch", None, None, None

        if saved_offset is None:
            return False, "no_offset", None, saved_resin_z_print_compensation, offset_file

        cur_offset = self._widget.get_tool_offset()
        for val_saved, val_cur in zip(saved_offset, cur_offset):
            if abs(val_saved - val_cur) > _NPZ_OFFSET_TOLERANCE_MM:
                return (
                    False,
                    "mismatch",
                    saved_offset,
                    saved_resin_z_print_compensation,
                    offset_file,
                )

        cur_resin_z = self._widget.current_resin_z_print_compensation()
        if saved_resin_z_print_compensation is None:
            return False, "no_offset", saved_offset, saved_resin_z_print_compensation, offset_file
        if abs(saved_resin_z_print_compensation - cur_resin_z) > _NPZ_OFFSET_TOLERANCE_MM:
            return (
                    False,
                    "mismatch",
                    saved_offset,
                    saved_resin_z_print_compensation,
                    offset_file,
                )

        return True, "ok", saved_offset, saved_resin_z_print_compensation, offset_file

    def _launch_npz_notice(
            self,
            npz_launch_path,
            source,
            saved_offset,
            saved_resin_z_print_compensation,
            offset_file,
            status):
        cur_offset = self._widget.get_tool_offset()
        cur_resin_z = self._widget.current_resin_z_print_compensation()
        source_text = "手动选择的 NPZ 文件夹" if source == "selected" else "按 GCode 自动匹配的 NPZ"
        lines = [
            f"NPZ 来源: {source_text}",
            f"启动 npz_path: {npz_launch_path}",
            f"当前界面纤维头偏置: {_format_tool_offset(cur_offset)}",
            f"当前界面树脂 Z 打印补偿: {cur_resin_z:.2f} mm",
        ]
        if saved_offset is not None:
            lines.append(f"NPZ 保存的纤维头偏置: {_format_tool_offset(saved_offset)}")
        if saved_resin_z_print_compensation is not None:
            lines.append(f"NPZ 保存的树脂 Z 打印补偿: {saved_resin_z_print_compensation:.2f} mm")
        if offset_file:
            lines.append(f"偏移文件: {offset_file}")

        if status == "ok":
            lines.append("纤维头偏置和树脂 Z 打印补偿校验通过。")
        elif status == "no_offset":
            lines.append("警告: 未找到或无法读取纤维头偏置或树脂 Z 打印补偿 sidecar，无法确认参数是否一致。")
        elif status == "mismatch":
            lines.append("警告: NPZ 中的纤维头偏置或树脂 Z 打印补偿与当前界面设置不一致。")

        related_values = []
        for name in _NPZ_RELATED_LAUNCH_PARAMS:
            if name == "npz_path":
                related_values.append(f"npz_path={npz_launch_path}")
            else:
                value = self._launch_params.get(name, _LAUNCH_DEFAULTS.get(name, ""))
                related_values.append(f"{name}={value}")

        lines.extend([
            "",
            "启动设置中与 NPZ 直接相关的是 npz_path。",
            (
                "npz_preload_chunks、queue_low/high、traj_prefill、traj_low/high 会影响"
                " NPZ 加载、预取和队列阈值；xyzabc_decimals/e_decimals"
                " 会影响轨迹数值发布精度。"
            ),
            "这些参数不会改写 NPZ 文件内容。",
            "当前相关启动参数: " + ", ".join(related_values),
            "",
            "是否继续启动？",
        ])
        return "\n".join(lines)

    def _test_mode_bootstrap_npz_path(self):
        return self._launch_params.get("npz_path") or _DEFAULT_NPZ_PATH

    def _on_launch(self):
        if (
            self._launch_process is not None
            and self._launch_process.poll() is None
        ):
            return

        npz_launch_path, source = self._current_npz_launch_path()
        if not npz_launch_path:
            if self._widget.active_mode() == _MODE_PAGE_TEST:
                if self._do_launch(self._test_mode_bootstrap_npz_path()):
                    self._widget._launch_status.setText("测试模式节点已启动（等待 KUKA 首包/临时 NPZ）")
                return
            gcode_path = self._launch_params.get("gcode_path", "").strip()
            if not gcode_path or not os.path.isfile(gcode_path):
                self._widget._launch_status.setText("启动失败: 未选择有效 GCode 或 NPZ 文件夹。")
            else:
                self._widget._launch_status.setText("启动失败: 未找到与此 GCode 匹配的 NPZ。")
            self._widget._launch_status.setStyleSheet(
                "color: #b42318; font-weight: 700; font-size: 13px;"
            )
            _show_warning(
                self._widget,
                "NPZ 文件缺失",
                "未找到可用于启动的 NPZ 数据。请先导出 NPZ，或选择已导出的 NPZ 文件。",
            )
            return

        (
            ok,
            status,
            saved_offset,
            saved_resin_z_print_compensation,
            offset_file,
        ) = self._check_npz_and_offset_match(npz_launch_path)
        if status == "missing":
            _show_warning(
                self._widget,
                "NPZ 文件缺失",
                f"启动路径无效或文件不存在:\n{npz_launch_path}",
            )
            self._widget._launch_status.setText("启动已取消: NPZ missing。")
            self._widget._launch_status.setStyleSheet(
                "color: #b42318; font-weight: 700; font-size: 13px;"
            )
            return

        title = "确认启动" if ok else "NPZ 校验警告"
        default_button = QtWidgets.QMessageBox.Yes if ok else QtWidgets.QMessageBox.No
        reply = _ask_yes_no(
            self._widget,
            title,
            self._launch_npz_notice(
                npz_launch_path,
                source,
                saved_offset,
                saved_resin_z_print_compensation,
                offset_file,
                status),
            default_button,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            self._widget._launch_status.setText(f"启动已取消: NPZ {status}。")
            self._widget._launch_status.setStyleSheet(
                "color: #b42318; font-weight: 700; font-size: 13px;"
            )
            return

        self._do_launch(npz_launch_path)

    def _do_launch(self, npz_launch_path=None):
        if npz_launch_path is None:
            npz_launch_path, _source = self._current_npz_launch_path()
        cmd = [
            "ros2",
            "launch",
            "my_project_startup",
            "startup.launch.py",
            f"npz_path:={npz_launch_path}",
        ]
        for name, value in self._launch_params.items():
            if name == "gcode_path":
                continue
            default = _LAUNCH_DEFAULTS.get(name, "")
            if value != default:
                cmd.append(f"{name}:={value}")
        try:
            self._launch_process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
            )
            self._widget._btn_launch.setEnabled(False)
            self._widget._btn_stop_launch.setEnabled(True)
            self._widget._btn_launch_settings.setEnabled(False)
            self._widget._launch_status.setText("运行中（等待 KUKA 首包...")
            self._widget._launch_status.setStyleSheet(
                "color: #1b6e3c; font-weight: 700; font-size: 13px;"
            )
            return True
        except Exception as exc:
            self._widget._btn_launch.setEnabled(True)
            self._widget._launch_status.setText(f"启动失败: {exc}")
            self._widget._launch_status.setStyleSheet(
                "color: #b42318; font-weight: 700; font-size: 13px;"
            )

            return False

    def _on_stop_launch(self):
        self._send_current_tool_heat_off()
        time.sleep(0.1)
        if (
            self._launch_process is not None
            and self._launch_process.poll() is None
        ):
            try:
                os.killpg(
                    os.getpgid(self._launch_process.pid), signal.SIGTERM
                )
            except OSError:
                self._launch_process.terminate()
        self._launch_process = None
        self._widget._btn_launch.setEnabled(True)
        self._widget._btn_stop_launch.setEnabled(False)
        self._widget._btn_launch_settings.setEnabled(True)
        self._widget._launch_status.setText("已停止")
        self._widget._launch_status.setStyleSheet(
            "color: #b42318; font-weight: 700; font-size: 13px;"
        )

    def _check_launch_process(self):
        if self._launch_process is None:
            return
        rc = self._launch_process.poll()
        if rc is not None:
            self._launch_process = None
            self._widget._btn_launch.setEnabled(True)
            self._widget._btn_stop_launch.setEnabled(False)
            self._widget._btn_launch_settings.setEnabled(True)
            if rc == 0:
                self._widget._launch_status.setText("已退出")
                self._widget._launch_status.setStyleSheet(
                    "color: #666666; font-weight: 700; font-size: 13px;"
                )
            else:
                self._widget._launch_status.setText(
                    f"异常退出（代码 {rc}）"
                )
                self._widget._launch_status.setStyleSheet(
                    "color: #b42318; font-weight: 700; font-size: 13px;"
                )

    def shutdown_plugin(self):
        self._shutting_down = True
        self._send_current_tool_heat_off()
        time.sleep(0.1)
        if self._launch_check_timer.isActive():
            self._launch_check_timer.stop()
        if self._spin_timer.isActive():
            self._spin_timer.stop()
        # Terminate launch process if running
        if (
            self._launch_process is not None
            and self._launch_process.poll() is None
        ):
            try:
                os.killpg(
                    os.getpgid(self._launch_process.pid), signal.SIGTERM
                )
            except OSError:
                self._launch_process.terminate()
            self._launch_process = None
