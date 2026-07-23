from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple


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


ParsedCommand = Union[MoveCommand, CurveCommand, GlobalCurveCommand,
                      ExtrudeWait, ResetECommand, ToolChangeCommand, MCommand]
ParsedCommandList = List[ParsedCommand]
