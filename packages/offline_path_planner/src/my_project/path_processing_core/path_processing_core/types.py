from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Sequence, Union, List, Tuple


@dataclass
class Position:
    x: float
    y: float
    z: float
    a: float
    b: float
    c: float


@dataclass
class MoveCommand:
    type: str  # "TRAVEL" 或者 "PRINT"
    cmd: str  # "G0" 或者 "G1"
    start_pos: Position  # 段起点
    pos: Position
    e_val: float
    delta_e: float
    feedrate: float  # mm/min
    line: int
    layer: int = 0
    subtype: str = "UNKNOWN"
    raw: Optional[str] = None
    target_v_in: Optional[float] = None  # planned entry speed (mm/s) 七阶多项式插值边界条件
    target_v_out: Optional[float] = None  # planned exit speed (mm/s)
    is_pure_state_change: bool = False  # 标记无位移且无挤出的纯状态改变指令


@dataclass
class ExtrudeWait:
    type: str  # "EXTRUDE_WAIT"
    wait_sec: float
    delta_e: float
    feedrate: float
    line: int
    layer: int = 0
    subtype: str = "UNKNOWN"
    raw: Optional[str] = None


@dataclass
class ResetECommand:
    type: str  # "RESET_E"
    val: float
    line: int
    layer: int = 0
    subtype: str = "UNKNOWN"
    raw: Optional[str] = None
    pose: Optional[Position] = None


@dataclass
class ToolChangeCommand:
    type: str  # "TOOL_CHANGE"
    tool: int
    line: int
    layer: int = 0
    subtype: str = "UNKNOWN"
    raw: Optional[str] = None


@dataclass
class MCommand:
    type: str  # "M_COMMAND"
    code: str
    params: Dict[str, float] = field(default_factory=dict)
    line: Optional[int] = None
    layer: int = 0
    subtype: str = "UNKNOWN"
    raw: Optional[str] = None
    tool: Optional[int] = None


@dataclass
class CurveCommand:
    """专用于角点处或全局拟合曲线."""

    type: str  # "PRINT" / "TRAVEL"
    cmd: str   # "CURVE"
    start_pos: Position
    control_points: List[Position]  # 包含终点
    e_val: float
    delta_e: float
    feedrate: float
    line: int
    raw: Optional[str] = None
    # Optional start acceleration time override for this curve only.
    time_acc_s: Optional[float] = None
    target_v_in: Optional[float] = None
    target_v_out: Optional[float] = None


@dataclass
class GlobalCurveCommand(CurveCommand):
    """
    全局 B 样条拟合指令.

    constraints: List[Tuple[normalized_s, speed_limit]]
    normalized_s: 0.0 ~ 1.0, 对应曲线弧长位置
    speed_limit: mm/s, 该位置的最大通过速度
    """

    constraints: List[Tuple[float, float]] = field(default_factory=list)
    original_moves: List[MoveCommand] = field(default_factory=list)
    # Optional absolute E values at the polyline control points.  When
    # present, the sampler follows this profile segment-by-segment instead of
    # distributing only the total delta_e by geometric arc length.
    e_profile: Optional[List[float]] = None
    # Optional absolute source-E samples on the position B-spline parameter
    # axis. Unlike e_profile these samples correspond to fitted input points,
    # not to B-spline control points, so they are safe for SPLINE commands.
    source_e_parameters: Optional[List[float]] = None
    source_e_values: Optional[List[float]] = None
    # KUKA A(Z)-B(Y)-C(X) quaternion samples on the position B-spline's
    # normalized parameter axis.  Sampling uses local SLERP to prevent a
    # global least-squares attitude fit from overshooting surface normals.
    orientation_parameters: Optional[List[float]] = None
    orientation_quaternions: Optional[List[Tuple[float, float, float, float]]] = None


def validate_source_e_profile(
    parameters: Sequence[float] | None,
    values: Sequence[float] | None,
    *,
    start_e: float,
    end_e: float,
    tolerance: float = 1e-9,
) -> tuple[List[float], List[float]] | None:
    """Validate an optional absolute E profile shared with a B-spline u axis.

    ``None`` keeps the legacy total-E-by-arc-length behavior. A partially
    supplied profile is invalid: source E is an explicit opt-in contract.
    """
    if parameters is None and values is None:
        return None
    if parameters is None or values is None:
        raise ValueError("source E parameters and values must be supplied together")

    normalized_parameters = [float(value) for value in parameters]
    normalized_values = [float(value) for value in values]
    if len(normalized_parameters) != len(normalized_values) or len(normalized_parameters) < 2:
        raise ValueError("source E profile requires equal parameter/value lengths of at least 2")
    if not all(math.isfinite(value) for value in (*normalized_parameters, *normalized_values)):
        raise ValueError("source E profile values must be finite")
    if abs(normalized_parameters[0]) > tolerance or abs(normalized_parameters[-1] - 1.0) > tolerance:
        raise ValueError("source E parameters must start at 0 and end at 1")
    if any(
        right - left <= tolerance
        for left, right in zip(normalized_parameters, normalized_parameters[1:])
    ):
        raise ValueError("source E parameters must be strictly increasing")
    if any(
        right < left - tolerance
        for left, right in zip(normalized_values, normalized_values[1:])
    ):
        raise ValueError("source E values must be monotonic non-decreasing")
    if abs(normalized_values[0] - float(start_e)) > tolerance:
        raise ValueError("source E profile start does not match curve start E")
    if abs(normalized_values[-1] - float(end_e)) > tolerance:
        raise ValueError("source E profile end does not match curve end E")
    return normalized_parameters, normalized_values


ParsedCommand = Union[MoveCommand, CurveCommand, GlobalCurveCommand,
                      ExtrudeWait, ResetECommand, ToolChangeCommand, MCommand]
ParsedCommandList = List[ParsedCommand]
