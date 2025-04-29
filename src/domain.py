from dataclasses import dataclass, field
from typing import Optional

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
    wait: bool = False

    def is_same_location(self, o: Location):
        return self.x == o.x and self.y == o.y

    def desc_loc_head(self) -> str:
        return f"({self.x},{self.y},{self.head})"

    def desc(self, map_dim_x: int):
        index = self.x + self.y * map_dim_x
        return f"{self.timeStart}:{self.timeEnd}|{index}|{self.x},{self.y}|{self.wait}"

    def __str__(self):
        return f"{self.timeStart}:{self.timeEnd}|{self.x},{self.y}@{self.head}|{self.wait}"

    def time_state_key(self):
        return f"{self.timeStart}:{self.timeEnd}|{self.x},{self.y}"


@dataclass_json
@dataclass
class TargetOnePlanResult:
    """
    单目标
    """
    robotName: str
    ok: bool = True
    reason: str = None
    cost: float = 0.0  # 机器人执行成本
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
    cost: float = 0.0  # 机器人执行成本
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
    id: int
    state: State
    parent: Optional['LowNode'] = None
    g: float = 0.0  # 到这个节点的成本
    f: float = 0.0
    focalValue: float = 0.0  # 注意，这个不是一定是 f 值

    def desc(self, map_dim_x: int):
        return f"{self.state.desc(map_dim_x)}|g={self.g}|f={self.f}|f2={self.focalValue}"


class Constraint(TimeSteps):
    pass


@dataclass_json
@dataclass(frozen=True)
class VertexConstraint(Constraint, Location):
    def __str__(self):
        return f"V:{self.timeStart}:{self.timeEnd}|({self.x},{self.y})"


@dataclass_json
@dataclass(frozen=True)
class EdgeConstraint(Constraint):
    x1: int
    y1: int
    x2: int
    y2: int

    def __str__(self):
        return f"E:{self.timeStart}:{self.timeEnd}|({self.x1},{self.y1})->({self.x2},{self.y2})"


@dataclass_json
@dataclass(frozen=True)
class RobotConstraints:
    """
    一个机器人的所有约束
    """
    robotName: str
    vertexConstraints: set[VertexConstraint] = field(default_factory=set)
    edgeConstraints: set[EdgeConstraint] = field(default_factory=set)


@dataclass
class ConflictConstraints:
    """
    一个冲突产生的约束
    """
    timeStart: int  # 最早时间
    constraints: dict[str, Constraint]  # robot name ->


@dataclass_json
@dataclass(frozen=True)
class HighNode:
    id: int  # 节点 ID
    parentId: int
    solution: dict[str, TargetManyPlanResult]  # robot name ->
    constraints: dict[str, RobotConstraints]  # robot name ->
    cost: float  # 最终的所有机器人的 f 值相加
    conflictsCount: int  # 这个高层节点中的冲突总数
    firstConstraints: Optional[ConflictConstraints] = None  # 这个高层节点中的第一个冲突


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
        if self.n.conflictsCount != other.n.conflictsCount:
            return self.n.conflictsCount < other.n.conflictsCount
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
        elif self.n.g != other.n.g:
            return self.n.g > other.n.g
        else:
            return self.n.id < other.n.id


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
        if self.n.focalValue != other.n.focalValue:
            return self.n.focalValue < other.n.focalValue
        elif self.n.f != other.n.f:
            return self.n.f < other.n.f
        elif self.n.g != other.n.g:
            return self.n.g > other.n.g
        else:
            return self.n.id < other.n.id


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


@dataclass
class LowContext:
    robotName: str
    highId: int
    w: float
    mapDimX: int
    mapDimY: int
    obstacles: set[int]
    moveUnitCost: float
    rotateUnitCost: float
    goalStopTimeNum: int
    constraints: Optional[RobotConstraints] = None  # 约束可能为空，初始求解时
    oldAllPaths: Optional[dict[str, list[State]]] = None  # 在进行此次底层求解所有机器人的已知路径


@dataclass
class FaContext:
    """
    Focal A* Context for low level search
    """
    robotName: str
    highId: int
    w: float
    mapDimX: int
    mapDimY: int
    obstacles: set[int]
    moveUnitCost: float
    rotateUnitCost: float
    goalStopTimeNum: int
    neighborValidator: Optional[callable] = None
    focalStateHeuristic: Optional[callable] = None
    focalTransitionHeuristic: Optional[callable] = None


@dataclass
class FaOp:
    """
    记录一次底层求解过程
    """
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

@dataclass_json
@dataclass
class LowOp:
    """
    记录一次底层求解过程
    """
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
    constraints: Optional[RobotConstraints]
    expandedNum: int
    expandedList: list[str]
    openSize: list[int]  # 每次展开后 open 集合的大小
    focalSize: list[int]  # 每次展开后 focal 集合的大小
    logs: list[str]
    warnings: list[str]
    ok: bool = False
    errMsg: str = ""
    path: Optional[list[State]] = None
    startedOn: float = 0.0
    endedOn: float = 0.0
    timeCost: float = 0.0
    expandedCount: int = 0
