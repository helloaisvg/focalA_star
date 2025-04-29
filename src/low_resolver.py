import heapq
import logging
import time
from dataclasses import replace
from math import ceil
from typing import Optional
from functools import total_ordering

from pydash import find_index

from src.common import is_time_overlay
from src.conflicts import count_robot_transition_conflicts
from src.domain import State, TargetOnePlanResult, OpenLowNode, FocalLowNode, LowContext, LowOp


@total_ordering
class LowNode:
    """
    低层搜索节点
    """

    def __init__(self, node_id: int, state: State, parent: Optional['LowNode'],
                 g: float, f: float, focalValue: float):
        self.id = node_id
        self.state = state
        self.parent = parent
        self.g = g  # 实际成本
        self.f = f  # 估计总成本 (g + h)
        self.focalValue = focalValue  # 用于Focal搜索的值

    def __lt__(self, other):
        # 优先比较 f 值，若 f 值相等，则比较 g 值（更接近目标的优先）
        if self.f == other.f:
            return self.g > other.g  # g 值大的优先
        return self.f < other.f

    def __eq__(self, other):
        # 比较基于 id
        return self.id == other.id

    def desc(self, map_dim_x: int) -> str:
        """节点描述信息"""
        return (f"{self.state.x + self.state.y * map_dim_x}|"
                f"{self.state.x},{self.state.y}|"
                f"{self.state.timeStart}:{self.state.timeEnd}|"
                f"{self.g:.1f}|{self.f:.1f}|{self.focalValue:.1f}")


class LowResolver:
    """
    单目标
    有界最优，Focal Search A*
    f, g, h 单位是时间，表示时间成本
    """

    def __init__(self, ctx: LowContext, low_id: int, time_offset: int,
                 start_state: State, goal_state: State,
                 last_goal_constraint: int = -1):
        """
        :param time_offset: 起始时刻
        :param start_state:
        :param goal_state:
        :param last_goal_constraint: 如果目标位置被约束了，last_goal_constraint 是被约束的最后一个时刻
        """
        print(f"Search one: robot={ctx.robotName}, offset={time_offset}, start={start_state}, goal={goal_state}, "
              f"constraints={ctx.constraints}")
        self.ctx = ctx
        self.time_offset = time_offset
        self.start_state = start_state
        self.goal_state = goal_state
        self.last_goal_constraint = last_goal_constraint

        self.node_id = 0

        # 计算最晚约束的时间
        self.constraint_time_end = -1
        if ctx.constraints:
            for vc in ctx.constraints.vertexConstraints:
                if vc.timeEnd > self.constraint_time_end:
                    self.constraint_time_end = vc.timeEnd
            for ec in ctx.constraints.edgeConstraints:
                if ec.timeEnd > self.constraint_time_end:
                    self.constraint_time_end = ec.timeEnd

        self.op = LowOp(
            robotName=ctx.robotName,
            highId=ctx.highId,
            lowId=low_id,
            w=ctx.w,
            mapDimX=ctx.mapDimX,
            mapDimY=ctx.mapDimY,
            moveUnitCost=ctx.moveUnitCost,
            rotateUnitCost=ctx.rotateUnitCost,
            goalStopTimeNum=ctx.goalStopTimeNum,
            startCell=f"{start_state.x},{start_state.y}",
            goalCell=f"{goal_state.x},{goal_state.y}",
            startIndex=start_state.x + ctx.mapDimX * start_state.y,
            goalIndex=goal_state.x + ctx.mapDimX * goal_state.y,
            constraints=ctx.constraints,
            expandedNum=0,
            expandedList=[],
            openSize=[],
            focalSize=[],
            logs=[],
            warnings=[],
            startedOn=time.time()
        )

        self.open_set: list[OpenLowNode] = []
        self.focal_set: list[FocalLowNode] = []
        self.closed_set: dict[str, LowNode] = {}  # by key

    def resolve_one(self):
        r = self.do_resolve_one()

        # 求解过程写入日志
        op = self.op
        op.expandedNum = len(op.expandedList)
        op.endedOn = time.time()
        op.timeCost = op.endedOn - op.startedOn
        op.ok = r.ok
        op.reason = r.reason
        op.path = r.path

        txt = op.to_json(indent=2)
        file = f"op/high-{op.highId}-low-{op.lowId}-r-{op.robotName}-{op.ok}.json"
        try:
            with open(file, 'w', encoding='utf-8') as f:
                f.write(txt)
        except Exception as e:
            logging.error(f'Failed to write out low resolver op: {e}')

        return r

    def do_resolve_one(self) -> TargetOnePlanResult:
        # 初始节点
        start_node = LowNode(
            self.node_id,
            replace(self.start_state, timeStart=self.time_offset, timeEnd=self.time_offset, timeNum=1),
            None,
            g=0.0,
            f=self.admissible_heuristic(self.start_state),
            focalValue=0.0,
        )

        heapq.heappush(self.open_set, OpenLowNode(start_node))
        heapq.heappush(self.focal_set, FocalLowNode(start_node))

        # Track the search start time for time-based pruning
        search_start_time = time.time()

        while self.open_set:
            # Check if search time exceeded
            if time.time() - search_start_time > 10.0:  # 10秒超时
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "Search time limit exceeded",
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            # 重建 focal set
            self.rebuild_focal_set()

            if not self.focal_set:
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "No valid path found in focal set",
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            # 从 Focal 集合中选择节点，优先处理冲突较少的节点
            min_focal_n: LowNode = heapq.heappop(self.focal_set).n
            self.remove_open_node(min_focal_n)

            # 记录展开过程
            self.op.expandedList.append(f"{self.op.expandedCount}|{min_focal_n.desc(self.ctx.mapDimX)}")
            self.op.expandedCount += 1
            self.op.openSize.append(len(self.open_set))
            self.op.focalSize.append(len(self.focal_set))

            # 加入 closed 集
            self.closed_set[min_focal_n.state.time_state_key()] = min_focal_n

            # 找到解
            if (min_focal_n.state.is_same_location(self.goal_state) and
                    min_focal_n.state.timeEnd > self.last_goal_constraint):
                return self.build_ok_result(min_focal_n)

            # 扩展节点
            neighbors = self.get_neighbors(min_focal_n.state)
            for neighbor in neighbors:
                self.new_open_node(min_focal_n, neighbor)

        return TargetOnePlanResult(
            self.ctx.robotName,
            False,
            "All states checked",
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedCount,
            fromState=self.start_state,
            toState=self.goal_state,
        )

    def rebuild_focal_set(self):
        """重建focal集合"""
        if not self.open_set:
            return

        min_f = self.open_set[0].n.f
        self.focal_set = []
        bound = min_f * self.ctx.w
        for node in self.open_set:
            # 优化：动态调整 Focal 集合范围
            if node.n.f <= bound and node.n.focalValue < bound * 1.5:
                heapq.heappush(self.focal_set, FocalLowNode(node.n))

    def new_open_node(self, from_node: LowNode, neighbor: State):
        """构造新节点并添加到open集合"""
        if not neighbor.wait:
            if self.detect_loop(from_node, neighbor):
                self.op.logs.append(f"Loop detected, skip: {neighbor}")
                return

        g = from_node.g + neighbor.timeNum
        f = g + self.admissible_heuristic(neighbor)
        focal_value = self.calc_focal_value(from_node, neighbor, g, f)

        # 优化：跳过明显无效的节点
        if focal_value > f * 2.0:  # 如果 Focal 值过高，跳过
            return

        new_node = LowNode(
            self.node_id,
            neighbor,
            from_node,
            g=g,
            f=f,
            focalValue=focal_value
        )
        self.node_id += 1

        # 检查是否已在closed集
        state_key = neighbor.time_state_key()
        if state_key in self.closed_set:
            if f < self.closed_set[state_key].f:
                del self.closed_set[state_key]
            else:
                return

        # 检查是否已在open集
        existing_node = self.find_open_node_by_key(state_key)
        if existing_node:
            if g < existing_node.g:
                existing_node.g = g
                existing_node.f = f
                existing_node.parent = from_node
                heapq.heapify(self.open_set)
        else:
            heapq.heappush(self.open_set, OpenLowNode(new_node))

    def admissible_heuristic(self, state: State) -> float:
        """改进的启发式函数"""
        dx = abs(state.x - self.goal_state.x)
        dy = abs(state.y - self.goal_state.y)
        distance = (dx + dy) / self.ctx.moveUnitCost

        # 旋转成本
        d_head = abs(state.head - self.goal_state.head)
        if d_head > 180:
            d_head = 360 - d_head
        rotation_cost = (d_head / 90.0) / self.ctx.rotateUnitCost

        # 优化：考虑冲突代价
        conflicts = count_robot_transition_conflicts(
            self.ctx.robotName,
            state,
            self.goal_state,
            self.ctx.oldAllPaths
        )
        conflict_cost = conflicts * 5.0  # 冲突代价权重

        # 动态调整启发式值，优先减少冲突
        return distance + rotation_cost * 0.5 + conflict_cost * 1.5

    def calc_focal_value(self, from_node: LowNode, to_state: State, to_state_g: float, to_state_f: float) -> float:
        """计算focal值：冲突数 + 路径成本"""
        conflicts = count_robot_transition_conflicts(
            self.ctx.robotName,
            from_node.state,
            to_state,
            self.ctx.oldAllPaths
        )
        return conflicts * 10.0 + to_state_g * 0.1

    def get_neighbors(self, from_state: State) -> list[State]:
        """获取有效邻居状态"""
        neighbors = []

        # 如果时间点还在约束范围内，考虑等待
        if from_state.timeEnd <= self.constraint_time_end:
            self.add_valid_neighbor(neighbors, from_state, 0, 0, from_state.head)  # 等待

        # 四个移动方向
        directions = [
            (1, 0, 0),  # 右
            (-1, 0, 180),  # 左
            (0, 1, 90),  # 下
            (0, -1, 270)  # 上
        ]

        for dx, dy, to_head in directions:
            self.add_valid_neighbor(neighbors, from_state, dx, dy, to_head)

        # 优化：动态调整等待时间
        if not neighbors and from_state.timeEnd <= self.constraint_time_end:
            self.add_valid_neighbor(neighbors, from_state, 0, 0, from_state.head, max_wait_time=3)

        return neighbors

    def add_valid_neighbor(self, neighbors: list[State], from_state: State, dx: int, dy: int, to_head: int, max_wait_time: int = 5):
        """添加有效邻居状态"""
        x = from_state.x + dx
        y = from_state.y + dy

        # 检查边界和障碍物
        if (x < 0 or x >= self.ctx.mapDimX or
                y < 0 or y >= self.ctx.mapDimY or
                self.state_to_index(x, y) in self.ctx.obstacles):
            return

        # 计算转向角度和时间成本
        d_head = abs(to_head - from_state.head)
        if d_head > 180:
            d_head = 360 - d_head
        rotation_time = ceil(d_head / 90.0 / self.ctx.rotateUnitCost)
        move_time = ceil((abs(dx) + abs(dy)) / self.ctx.moveUnitCost)
        time_num = max(1, rotation_time + move_time)

        # 限制最大等待时间
        if dx == 0 and dy == 0 and time_num > max_wait_time:  # 等待时间超过限制则跳过
            return

        new_state = State(
            x=x,
            y=y,
            head=to_head,
            timeStart=from_state.timeEnd + 1,
            timeEnd=from_state.timeEnd + time_num,
            timeNum=time_num,
            wait=dx == 0 and dy == 0,
        )

        # 检查状态有效性
        if (self.state_valid(new_state) and
                self.transition_valid(from_state, new_state)):
            neighbors.append(new_state)

    def state_valid(self, state: State) -> bool:
        """检查状态是否满足顶点约束"""
        if not self.ctx.constraints:
            return True

        for vc in self.ctx.constraints.vertexConstraints:
            if (state.x == vc.x and state.y == vc.y and
                    is_time_overlay(state.timeStart, state.timeEnd, vc.timeStart, vc.timeEnd)):
                return False
        return True

    def transition_valid(self, from_state: State, to_state: State) -> bool:
        """检查转移是否满足边约束"""
        if from_state.is_same_location(to_state):
            return True

        if not self.ctx.constraints:
            return True

        for ec in self.ctx.constraints.edgeConstraints:
            if (from_state.x == ec.x1 and from_state.y == ec.y1 and
                    to_state.x == ec.x2 and to_state.y == ec.y2 and
                    is_time_overlay(from_state.timeStart, to_state.timeEnd, ec.timeStart, ec.timeEnd)):
                return False
        return True

    def state_to_index(self, x: int, y: int) -> int:
        """坐标转索引"""
        return x + y * self.ctx.mapDimX

    def find_open_node_by_key(self, key: str) -> Optional[LowNode]:
        """在open集中查找节点"""
        for node in self.open_set:
            if node.n.state.time_state_key() == key:
                return node.n
        return None

    def remove_open_node(self, node: LowNode):
        """从open集中移除节点"""
        index = find_index(self.open_set, lambda n: n.n.id == node.id)
        if index >= 0:
            self.open_set[index] = self.open_set[-1]
            self.open_set.pop()
            heapq.heapify(self.open_set)

    def build_ok_result(self, node: LowNode) -> TargetOnePlanResult:
        """构建成功结果"""
        path = []
        current = node
        while current:
            path.append(current.state)
            current = current.parent
        path.reverse()

        # 添加目标点停留时间
        stop_time = max(1, self.ctx.goalStopTimeNum)
        final_state = replace(
            path[-1],
            timeStart=path[-1].timeEnd + 1,
            timeEnd=path[-1].timeEnd + stop_time,
            timeNum=stop_time
        )
        path.append(final_state)

        return TargetOnePlanResult(
            self.ctx.robotName,
            True,
            cost=node.g + stop_time,
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedCount,
            timeNum=final_state.timeEnd,
            timeStart=self.time_offset,
            timeEnd=final_state.timeEnd,
            fromState=self.start_state,
            toState=self.goal_state,
            path=path
        )

    @staticmethod
    def detect_loop(from_node: LowNode, neighbor: State) -> bool:
        """检测路径循环"""
        current = from_node
        while current:
            if current.state.is_same_location(neighbor):
                return True
            current = current.parent
        return False
