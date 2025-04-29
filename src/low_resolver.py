import heapq
import logging
import time
from dataclasses import replace
from math import ceil
from typing import Optional, Set, Dict, Tuple

from pydash import find_index

from src.common import is_time_overlay
from src.conflicts import count_robot_transition_conflicts
from src.domain import State, TargetOnePlanResult, OpenLowNode, FocalLowNode, LowNode, LowContext, LowOp


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

        # Cache visited states by location for cycle detection
        self.visited_locations: Dict[Tuple[int, int], Set[int]] = {}

        # Maximum time allowed for a single path search
        self.max_search_time = 10.0  # seconds

        # Target maximum path length - used for pruning extremely long paths
        self.max_path_length = self.get_max_path_length()

        # Cache for constraint validation to avoid repeated checks
        self.constraint_cache: Dict[str, bool] = {}

        self.op = LowOp(
            robotName=ctx.robotName,
            highId=ctx.highId,
            lowId=low_id,
            w=ctx.w,
            mapDimX=ctx.mapDimX,
            mapDimY=ctx.mapDimY,
            # obstacles=ctx.obstacles,
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
        op.errMsg = r.errMsg
        op.path = r.path

        txt = op.to_json(indent=2)
        file = f"op/high-{op.highId}-low-{op.lowId}-r-{op.robotName}-{op.ok}.json"
        try:
            with open(file, 'w', encoding='utf-8') as file:
                file.write(txt)
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

        # Add start location to visited locations
        self.add_visited_location(self.start_state.x, self.start_state.y, self.start_state.timeEnd)

        while self.open_set:
            # Check if search time exceeded
            if time.time() - search_start_time > self.max_search_time:
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "Search time limit exceeded",
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            # 每次进来重建 focal set，因此后续都不处理向 focal set 添加元素
            self.rebuild_focal_set()

            # 取出下一个有界最优启发值最低的节点
            if not self.focal_set:
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "No valid path found in focal set",
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            min_focal_n: LowNode = heapq.heappop(self.focal_set).n
            min_focal_s = min_focal_n.state

            # 记录展开过程
            self.op.expandedList.append(str(self.op.expandedCount) + "|" + min_focal_n.desc(self.ctx.mapDimX))
            self.op.expandedCount += 1
            self.op.openSize.append(len(self.open_set))
            self.op.focalSize.append(len(self.focal_set))

            # 从 open 中删除
            self.remove_open_node(min_focal_n)

            # 已展开的节点加入到 close
            self.closed_set[min_focal_s.time_state_key()] = min_focal_n

            # Pruning: Early termination if path is getting too long
            if min_focal_n.g > self.max_path_length:
                continue  # Skip this node as the path is too long

            # 如果限制最大展开次数，超过则结束
            expanded_max = self.get_expanded_max()
            if 0 < expanded_max < self.op.expandedCount:
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "Expanded nodes too many",
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            # 找到解：到达目标位置
            if min_focal_s.is_same_location(self.goal_state) and min_focal_s.timeEnd > self.last_goal_constraint:
                return self.build_ok_result(min_focal_n)

            # 扩展节点
            neighbors = self.get_neighbors(min_focal_s)
            for neighbor in neighbors:
                # Pruning: Skip neighbors that we've already visited at the same or earlier time
                if self.is_location_visited_earlier(neighbor.x, neighbor.y, neighbor.timeEnd):
                    continue

                self.new_open_node(min_focal_n, neighbor)

                # Record this location as visited
                self.add_visited_location(neighbor.x, neighbor.y, neighbor.timeEnd)

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
        """
        整体重建 focal 集合
        """
        if not self.open_set:
            return

        min_f = self.open_set[0].n.f  # 读取 open 最小的节点，但不删除
        self.focal_set = []
        bound = min_f * self.ctx.w
        for node in self.open_set:
            if node.n.f <= bound:
                heapq.heappush(self.focal_set, FocalLowNode(node.n))

    def get_expanded_max(self):
        """
        限制最多展开次数。-1 表示不限制。
        """
        # Set a reasonable limit based on map size to prevent excessive exploration
        return self.ctx.mapDimX * self.ctx.mapDimY * 2

    def get_max_path_length(self):
        """
        Estimate maximum reasonable path length based on map dimensions
        """
        # Manhattan distance + some buffer for detours
        direct_distance = abs(self.start_state.x - self.goal_state.x) + abs(self.start_state.y - self.goal_state.y)
        return direct_distance * 3 + 10  # Allow some detours and waiting

    def new_open_node(self, from_node: LowNode, neighbor: State):
        """
        构造新节点并添加到 open 集合
        """
        # Skip if the neighbor would create a cycle (unless it's waiting)
        if not neighbor.wait:
            if self.detect_loop(from_node, neighbor):
                self.op.logs.append(f"Loop, return, {neighbor}")
                return

        g = from_node.g + neighbor.timeNum

        # Pruning: Skip nodes that exceed our maximum path length estimate
        if g > self.max_path_length:
            return

        f = g + self.admissible_heuristic(neighbor)
        focal_value = self.calc_focal_value(from_node, neighbor, g, f)
        self.node_id += 1
        node = LowNode(self.node_id, neighbor, from_node, g=g, f=f, focalValue=focal_value)

        cell_index = self.state_to_index(neighbor.x, neighbor.y)
        state_key = neighbor.time_state_key()
        old_closed = self.closed_set.get(state_key)
        if old_closed:
            # 如果在 close 里
            # Improved: Consider reopening a closed node if we found a better path
            if f < old_closed.f:
                self.op.warnings.append(f"SmallerFInClosed|{cell_index}|{neighbor.x},{neighbor.y}|"
                                        f"{neighbor.timeStart}:{neighbor.timeEnd}|"
                                        f"{old_closed.state.timeStart}:{old_closed.state.timeEnd}|"
                                        f"{f}|{old_closed.f}")
                del self.closed_set[state_key]
            else:
                return
        old_open = self.find_open_node_by_key(state_key)
        if old_open:
            # 如果在 open 里，如果 g 变小了，更新
            if g >= old_open.g:
                return
            node = replace(old_open, g=g, f=old_open.f + g - old_open.g)
            self.replace_open_node(OpenLowNode(node))
        else:
            heapq.heappush(self.open_set, OpenLowNode(node))

    def admissible_heuristic(self, state: State) -> float:
        """
        两点之间的直线路径
        vs 欧氏距离
        不考虑旋转
        """
        # Manhattan distance heuristic
        manhattan_dist = float(abs(state.x - self.goal_state.x) + abs(state.y - self.goal_state.y))

        # Add rotation cost estimate
        rotation_cost = 0.0
        if state.head != self.goal_state.head:
            d_head = abs(state.head - self.goal_state.head)
            if d_head > 180:
                d_head = 360 - d_head
            rotation_cost = float(d_head / 90) / self.ctx.rotateUnitCost

        return (manhattan_dist / self.ctx.moveUnitCost) + rotation_cost

    def calc_focal_value(self, from_node: LowNode, to_state: State, to_state_g: float, to_state_f: float) -> float:
        """
        计算 focal 值。
        """
        # Calculate conflicts with other robots plus a slight preference for shorter paths
        conflicts = count_robot_transition_conflicts(self.ctx.robotName, from_node.state, to_state,
                                                     self.ctx.oldAllPaths)

        # Give slight preference to paths with fewer time steps
        return conflicts * 10.0 + to_state_g * 0.1

    def get_neighbors(self, from_state: State) -> list[State]:
        neighbors = []

        # Pruning: If we're close to the goal, prioritize direct movements toward goal
        is_close_to_goal = (abs(from_state.x - self.goal_state.x) + abs(from_state.y - self.goal_state.y)) <= 2

        # If not close to the goal or time point is still before the last constraint, consider waiting
        if not is_close_to_goal or from_state.timeEnd <= self.constraint_time_end:
            self.add_valid_neighbor(neighbors, from_state, 0, 0, from_state.head)  # waiting

        # Calculate direction to goal for prioritizing movement
        dx_to_goal = self.goal_state.x - from_state.x
        dy_to_goal = self.goal_state.y - from_state.y

        # Prioritize movements that bring us closer to the goal
        directions = []

        # Add horizontal movement toward goal
        if dx_to_goal > 0:
            directions.append((1, 0, 0))  # right
        elif dx_to_goal < 0:
            directions.append((-1, 0, 180))  # left

        # Add vertical movement toward goal
        if dy_to_goal > 0:
            directions.append((0, 1, 90))  # down
        elif dy_to_goal < 0:
            directions.append((0, -1, 270))  # up

        # Add remaining directions if not yet in the list
        for direction in [(1, 0, 0), (-1, 0, 180), (0, 1, 90), (0, -1, 270)]:
            if direction not in directions:
                directions.append(direction)

        # Try each direction
        for dx, dy, to_head in directions:
            self.add_valid_neighbor(neighbors, from_state, dx, dy, to_head)

        return neighbors

    def add_valid_neighbor(self, neighbors: list[State], from_state: State, dx: int, dy: int, to_head: int):
        """
        如果移动有效，则添加到 neighbors
        toHead 目标车头朝向
        """
        x = from_state.x + dx
        y = from_state.y + dy

        # 在地图范围内且不是障碍物
        if (x < 0 or x >= self.ctx.mapDimX or y < 0 or y >= self.ctx.mapDimY
                or self.state_to_index(x, y) in self.ctx.obstacles):
            return

        # 需要转的角度，初始，-270 ~ +270
        d_head = abs(to_head - from_state.head)
        # 270 改成 90
        if d_head > 180:
            d_head = 360 - d_head

        d_head /= 90

        # 耗时，也作为 g 的增量
        # 假设 dx/dy 1 是 1 米
        time_num = ceil(float(abs(dx + dy)) / self.ctx.moveUnitCost + float(d_head) / self.ctx.rotateUnitCost)

        if time_num < 1:
            time_num = 1  # 原地等待

        new_state = State(
            x=x,
            y=y,
            head=to_head,
            timeStart=from_state.timeEnd + 1,
            timeEnd=from_state.timeEnd + time_num,
            timeNum=time_num,
            wait=dx == 0 and dy == 0,
        )

        # 最后一步要等待
        test_state = new_state
        if self.goal_state.is_same_location(new_state):
            # 增加等待时间
            test_state = replace(
                new_state,
                timeEnd=new_state.timeEnd + self.ctx.goalStopTimeNum,
                timeNum=new_state.timeNum + self.ctx.goalStopTimeNum)

        # 额外检查是否有效，主要是检查约束
        if not (self.state_valid(test_state) and self.transition_valid(from_state, test_state)):
            return

        neighbors.append(new_state)

    def state_valid(self, to_state: State) -> bool:
        cs = self.ctx.constraints
        if not cs:
            return True

        # Cache constraint validation results
        cache_key = f"v_{to_state.x}_{to_state.y}_{to_state.timeStart}_{to_state.timeEnd}"
        if cache_key in self.constraint_cache:
            return self.constraint_cache[cache_key]

        v_constraints = cs.vertexConstraints
        for vc in v_constraints:
            if (to_state.is_same_location(vc) and
                    is_time_overlay(to_state.timeStart, to_state.timeEnd, vc.timeStart, vc.timeEnd)):
                self.constraint_cache[cache_key] = False
                return False

        self.constraint_cache[cache_key] = True
        return True

    def transition_valid(self, from_state: State, to_state: State) -> bool:
        if from_state.is_same_location(to_state):
            return True

        cs = self.ctx.constraints
        if not cs:
            return True

        # Cache constraint validation results
        cache_key = f"e_{from_state.x}_{from_state.y}_{to_state.x}_{to_state.y}_{from_state.timeStart}_{to_state.timeEnd}"
        if cache_key in self.constraint_cache:
            return self.constraint_cache[cache_key]

        e_constraints = cs.edgeConstraints
        # 起点、终点位置相同
        for ec in e_constraints:
            if (ec.x1 == from_state.x and ec.y1 == from_state.y and ec.x2 == to_state.x and ec.y2 == to_state.y and
                    is_time_overlay(from_state.timeStart, to_state.timeEnd, ec.timeStart, ec.timeEnd)):
                self.constraint_cache[cache_key] = False
                return False

        self.constraint_cache[cache_key] = True
        return True

    def state_to_index(self, x: int, y: int):
        return x + y * self.ctx.mapDimX

    def find_open_node_by_key(self, key: str) -> Optional[LowNode]:
        for node in self.open_set:
            n = node.n
            if n.state.time_state_key() == key:
                return n
        return None

    def remove_open_node(self, node: LowNode):
        index = find_index(self.open_set, lambda n: n.n.id == node.id)
        if index < 0:
            return
        else:
            # 把最后一个填充过来
            self.open_set[index] = self.open_set[-1]
            self.open_set.pop()
            # 必须重建！o(n)
            heapq.heapify(self.open_set)

    def replace_open_node(self, new_one: OpenLowNode):
        index = find_index(self.open_set, lambda n: n.n.id == new_one.n.id)
        if index < 0:
            heapq.heappush(self.open_set, new_one)
        else:
            self.open_set[index] = new_one
            # 必须重建！o(n)
            heapq.heapify(self.open_set)

    def build_ok_result(self, min_focal_n: LowNode):
        # 达到时间在最后一次目标点被约束的时刻后
        # 到达后等待一段时间
        last_state = min_focal_n.state
        path = [last_state]
        curr_node = min_focal_n.parent
        while curr_node:
            path.append(curr_node.state)
            curr_node = curr_node.parent
        path.reverse()

        goal_stop_time_num = self.ctx.goalStopTimeNum if self.ctx.goalStopTimeNum > 0 else 1

        # 最后追加一个原地等待的，模拟动作时间
        # noinspection PyTypeChecker
        action_state = replace(last_state,
                               timeStart=last_state.timeEnd + 1,
                               timeEnd=last_state.timeEnd + goal_stop_time_num,
                               timeNum=goal_stop_time_num)
        path.append(action_state)

        return TargetOnePlanResult(
            self.ctx.robotName,
            True,
            cost=min_focal_n.g + goal_stop_time_num,
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedCount,
            timeNum=action_state.timeEnd,
            timeStart=self.time_offset,
            timeEnd=action_state.timeEnd,
            fromState=self.start_state,
            toState=self.goal_state,
            path=path
        )

    def detect_loop(self, from_node: LowNode, neighbor: State):
        """
        检查路径中是否出现过此位置
        """
        n = from_node
        while n:
            if n.state.is_same_location(neighbor):
                return True
            n = n.parent
        return False

    def add_visited_location(self, x: int, y: int, time_end: int):
        """
        Record a visited location and its time
        """
        loc_key = (x, y)
        if loc_key not in self.visited_locations:
            self.visited_locations[loc_key] = set()
        self.visited_locations[loc_key].add(time_end)

    def is_location_visited_earlier(self, x: int, y: int, time_end: int) -> bool:
        """
        Check if location was visited at an earlier or equal time
        """
        loc_key = (x, y)
        if loc_key not in self.visited_locations:
            return False

        # If we have visited this location at any earlier or same time, we can prune
        for visited_time in self.visited_locations[loc_key]:
            if visited_time <= time_end:
                return True

        return False
