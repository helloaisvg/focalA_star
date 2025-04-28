import math
from dataclasses import dataclass, field
from typing import Optional, Callable

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class Location:
    """
    frozen 才是 hashable 的，可以放入 set。
    """
    x: int
    y: int


@dataclass_json
@dataclass(frozen=True)
class Cell(Location):
    pass


@dataclass_json
@dataclass(frozen=True)
class TimeSteps:
    timeStart: int  # 开始去这个位置的第一个时间步
    timeEnd: int  # 到达这个位置的时刻


def is_time_overlay(start1: int, end1: int, start2: int, end2: int):
    return start1 <= end2 and start2 <= end1


@dataclass_json
@dataclass(frozen=True)
class State(Location, TimeSteps):
    """
    允许跨多个时间步
    车头方向，
    0，x 正，向右
    90，y 正，向下
    180，x 负，向左
    270，y 负，向上
    """
    head: int
    timeNum: int

    def is_same_location(self, o: Location):
        return self.x == o.x and self.y == o.y

    def is_time_overlay(self, o: TimeSteps):
        return is_time_overlay(self.timeStart, self.timeEnd, o.timeStart, o.timeEnd)

    def desc_loc_head(self) -> str:
        return f"({self.x},{self.y},{self.head})"

    def desc(self, map_dim_x: int):
        index = self.x + self.y * map_dim_x
        return f"{self.timeStart}:{self.timeEnd}|{index}|{self.x},{self.y}"


def cell_to_state(c: Cell) -> State:
    return State(x=c.x, y=c.y, timeStart=0, timeEnd=0, timeNum=0, head=0)


def state_to_cell(s: State) -> Cell:
    return Cell(s.x, s.y)


@dataclass_json
@dataclass
class TargetOnePlanResult:
    """
    单目标
    """
    robotName: str
    ok: bool = True
    reason: str = None
    cost: float = 0.0
    minF: float = 0.0
    expandedCount: int = 0
    planCost: float = 0  # 秒
    timeNum: int = 0
    timeStart: int = -1
    timeEnd: int = -1
    fromState: State = None
    toState: State = None
    path: list[State] = None
    extra: any = None


@dataclass_json
@dataclass
class TargetManyPlanResult:
    """
    多目标
    """
    robotName: str
    ok: bool = True
    reason: str = None
    cost: float = 0.0
    minF: float = 0.0
    expandedCount: int = 0
    planCost: float = 0  # 秒
    timeNum: int = 0
    timeStart: int = -1
    timeEnd: int = -1
    steps: list[TargetOnePlanResult] = None
    path: list[State] = None  # 总路径
    extra: any = None


@dataclass_json
@dataclass
class RobotTask:
    name: str
    fromState: Cell
    toStates: list[Cell]
    stopTimes: int = 1  # 停多久


@dataclass_json
@dataclass
class MapfReq:
    w: float
    mapDimX: int
    mapDimY: int  # y 向下为正
    obstacles: set[int]
    tasks: dict[str, RobotTask]  # robot name ->
    goalStops: int = 0


@dataclass_json
@dataclass
class MapfResult:
    ok: bool = True
    msg: str = ""
    plans: dict[str, TargetManyPlanResult] = None
    timeCost: float = 0.0


@dataclass_json
@dataclass(frozen=True)
class LowNode:
    state: State
    parent: Optional['LowNode'] = None
    f: float = 0.0
    f2: float = 0.0
    g: float = 0.0  # 到这个节点的实际成本

    def desc(self, map_dim_x: int):
        return f"{self.state.desc(map_dim_x)}|g={self.g}|f={self.f}|f2={self.f2}"


@dataclass_json
@dataclass(frozen=True)
class VertexConstraint(TimeSteps, Location):
    pass


@dataclass_json
@dataclass(frozen=True)
class EdgeConstraint(TimeSteps):
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass_json
@dataclass(frozen=True)
class Constraints:
    robotName: str
    vertexConstraints: field(default_factory=list)
    edgeConstraints: field(default_factory=list)


@dataclass_json
@dataclass(frozen=True)
class HighNode:
    id: int  # 节点 ID
    parentId: int
    solution: dict[str, TargetManyPlanResult]  # robot name ->
    constraints: dict[str, Constraints]  # robot name ->
    cost: float
    lb: float
    focalHeuristic: float


class OpenHighNode:

    def __init__(self, n: HighNode):
        self.n = n

    def __lt__(self, other: 'OpenHighNode'):
        # if (cost != n.cost)
        return self.n.cost < other.n.cost


class FocalHighNode:

    def __init__(self, n: HighNode):
        self.n = n

    def __lt__(self, other: 'FocalHighNode'):
        if self.n.focalHeuristic != other.n.focalHeuristic:
            return self.n.focalHeuristic < other.n.focalHeuristic
        return self.n.cost < other.n.cost


class OpenLowNode:
    """
    排序：lowest fScore，highest gScore
    """

    def __init__(self, n: LowNode):
        self.n = n

    def __lt__(self, other: 'OpenLowNode'):
        if self.n.f != other.n.f:
            return self.n.f < other.n.f
        return other.n.g < self.n.g


class FocalLowNode:
    """
    # Sort order (see "Improved Solvers for Bounded-Suboptimal Multi-Agent Path Finding" by Cohen et. al.)
    # 1. lowest focalHeuristic
    # 2. lowest fScore
    # 3. highest gScore
    """

    def __init__(self, n: LowNode):
        self.n = n

    def __lt__(self, other: 'FocalLowNode'):
        if self.n.f2 != other.n.f2:
            return self.n.f2 < other.n.f2
        elif self.n.f != other.n.f:
            return self.n.f < other.n.f
        return other.n.g < self.n.g


@dataclass_json
@dataclass
class MapConfig:
    robotNum: int = 10
    mapDimX: int = 30
    mapDimY: int = 20
    obstacleRatio: float = .3
    w: float = 1.5
    targetNum: int = 1
    goalStopTimes: int = 5
    obstacles: set[int] = field(default_factory=set)


@dataclass_json
@dataclass
class RobotTaskReq:
    fromIndex: int
    toIndex: int


@dataclass_json
@dataclass
class MapReq:
    config: MapConfig
    tasks: dict[str, RobotTaskReq]


def x_y_to_index(x: int, y: int, map_dim_x: int) -> int:
    return y * map_dim_x + x


def distance_of_two_points(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


@dataclass
class FaContext:
    robotName: str
    highId: int
    w: float
    mapDimX: int
    mapDimY: int
    obstacles: set[int]
    moveUnitCost: float
    rotateUnitCost: float
    goalStopTimeNum: int
    neighborValidator: Optional[Callable[[State, State], bool]] = None
    focalStateHeuristic: Optional[Callable[[State, float], float]] = None
    focalTransitionHeuristic: Optional[Callable[[State, State, float, float], float]] = None
    logLow: bool = False  # 是否打印底层搜索日志


@dataclass_json
@dataclass
class FaOp:
    robotName: str
    highId: int
    lowId: int
    w: float
    mapDimX: int
    mapDimY: int
    # obstacles: set[int]
    moveUnitCost: float
    rotateUnitCost: float
    goalStopTimeNum: int
    startCell: str
    goalCell: str
    startIndex: int
    goalIndex: int
    expandedList: list[str]
    openSize: list[int]  # 每次展开后 open 集合的大小
    focalSize: list[int]  # 每次展开后 focal 集合的大小
    warnings: list[str]
    ok: bool = False
    errMsg: str = ""
    path: Optional[list[State]] = None
    startedOn: float = 0.0
    endedOn: float = 0.0
    timeCost: float = 0.0
    expandedCount: int = 0
