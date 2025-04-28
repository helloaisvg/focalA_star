import heapq
import logging
import time
from dataclasses import replace
from math import ceil

from pydash import find, find_index

from src.domain import State, TargetOnePlanResult, OpenLowNode, FocalLowNode, LowNode, FaContext, FaOp


class FaOne:
    """
    单目标
    有界最优，Focal Search A*
    f, g, h 单位是时间，表示时间成本
    """

    def __init__(self, ctx: FaContext, low_id: int, time_offset: int, start_state: State, goal_state: State,
                 last_goal_constraint: int = -1):
        """
        :param time_offset: 起始时刻
        :param start_state:
        :param goal_state:
        :param last_goal_constraint: 如果目标位置被约束了，last_goal_constraint 是被约束的最后一个时刻
        """
        print(f"Search one: offset={time_offset}, start={start_state}, goal={goal_state}")
        self.ctx = ctx
        self.time_offset = time_offset
        self.start_state = start_state
        self.goal_state = goal_state
        self.last_goal_constraint = last_goal_constraint

        self.op = FaOp(
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
            expandedList=[],
            openSize=[],
            focalSize=[],
            warnings=[],
            startedOn=time.time()
        )

        self.open_set: list[OpenLowNode] = []
        self.focal_set: list[FocalLowNode] = []
        self.closed_set: dict[int, LowNode] = {}  # by cell index

        # 字典，用于存储每个单元格位置的最佳g值
        self.best_g_values: dict[int, float] = {}

    def search_one(self):
        r = self.do_search_one()

        op = self.op
        op.endedOn = time.time()
        op.timeCost = op.endedOn - op.startedOn
        op.ok = r.ok
        op.errMsg = op.errMsg
        op.path = op.path

        txt = op.to_json(indent=2)
        file = f"op/high-{op.highId}-low-{op.lowId}.json"
        try:
            with open(file, 'w', encoding='utf-8') as file:
                file.write(txt)
        except Exception as e:
            logging.error(f'写入文件时出错: {e}')

        return r

    def do_search_one(self) -> TargetOnePlanResult:
        # noinspection PyTypeChecker
        start_node = LowNode(
            replace(self.start_state, timeStart=self.time_offset, timeEnd=self.time_offset, timeNum=1),
            None,
            f=self.admissible_heuristic(self.start_state),
            g=0.0,
            f2=0.0,
        )

        heapq.heappush(self.open_set, OpenLowNode(start_node))
        heapq.heappush(self.focal_set, FocalLowNode(start_node))

        # 初始化起始位置的最佳g值
        start_cell_index = self.state_to_index(start_node.state.x, start_node.state.y)
        self.best_g_values[start_cell_index] = 0.0

        # 跟踪open中的当前最小f值
        min_f = start_node.f

        # 跟踪最后绑定以优化焦点集更新
        last_bound = min_f * self.ctx.w

        while self.open_set:
            # 获取open中的当前最小f值
            current_min_f = self.open_set[0].n.f

            # 如果min_f已更改，则更新焦点集
            if current_min_f > min_f:
                new_bound = current_min_f * self.ctx.w

                # 如果新节点现在符合条件，则将其添加到焦点集中
                if new_bound > last_bound:
                    for open_node in self.open_set:
                        if last_bound < open_node.n.f <= new_bound:
                            heapq.heappush(self.focal_set, FocalLowNode(open_node.n))

                min_f = current_min_f
                last_bound = new_bound

            # 无需每次从头开始重建焦点集
            # 仅在min_f更改时更新它，绑定也会更改

            # 如果焦点集为空重建它
            if not self.focal_set:
                bound = min_f * self.ctx.w
                for node in self.open_set:
                    if node.n.f <= bound:
                        heapq.heappush(self.focal_set, FocalLowNode(node.n))
                last_bound = bound

            # 从焦点集中获取最佳节点
            if not self.focal_set:
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "Focal set is empty",
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            min_focal_n: LowNode = heapq.heappop(self.focal_set).n
            min_focal_s = min_focal_n.state

            self.op.expandedList.append(str(self.op.expandedCount) + "|" + min_focal_n.desc(self.ctx.mapDimX))
            self.op.expandedCount += 1
            self.op.openSize.append(len(self.open_set))
            self.op.focalSize.append(len(self.focal_set))

            # 从open中删除
            self.remove_open_node(min_focal_n)

            # 加入到closed
            cell_index = self.state_to_index(min_focal_s.x, min_focal_s.y)
            self.closed_set[cell_index] = min_focal_n

            if self.op.expandedCount > self.ctx.mapDimX * self.ctx.mapDimY * 2:  # Increased limit a bit
                return TargetOnePlanResult(
                    self.ctx.robotName,
                    False,
                    "展开过多",
                    planCost=time.time() - self.op.startedOn,
                    fromState=self.start_state,
                    toState=self.goal_state,
                )

            if min_focal_s.is_same_location(self.goal_state) and min_focal_s.timeEnd > self.last_goal_constraint:
                return self.build_ok_result(min_focal_n)

            # traverse neighbors
            neighbors = self.get_neighbors(min_focal_s)
            for neighbor in neighbors:
                g = min_focal_n.g + neighbor.timeNum
                f = g + self.admissible_heuristic(neighbor)
                focal_heuristic = (min_focal_n.f2 +
                                   self.focal_state_heuristic(neighbor, g) +
                                   self.focal_transition_heuristic(min_focal_s, neighbor, min_focal_n.g, g))

                cell_index = self.state_to_index(neighbor.x, neighbor.y)

                # 检查是否已经为这个位置提供了更好的g值
                if cell_index in self.best_g_values and g >= self.best_g_values[cell_index]:
                    # 如果已经找到了更好路径，请跳过
                    continue

                # 跟新 best g-value
                self.best_g_values[cell_index] = g

                node = LowNode(neighbor, min_focal_n, f=f, f2=focal_heuristic, g=g)

                # 检查是否在f值较低的closed中
                old_closed = self.closed_set.get(cell_index)
                if old_closed:
                    if f < old_closed.f:
                        self.op.warnings.append(f"SmallerFInClosed|{cell_index}|{neighbor.x},{neighbor.y}|"
                                                f"{neighbor.timeStart}:{neighbor.timeEnd}|"
                                                f"{old_closed.state.timeStart}:{old_closed.state.timeEnd}|"
                                                f"{f}|{old_closed.f}")
                        del self.closed_set[cell_index]
                    else:
                        continue

                # 检查是否在open中
                old_open = find(self.open_set, lambda n: n.n.state.x == neighbor.x and n.n.state.y == neighbor.y)
                if old_open:
                    # 如果在g值较差的open中，更新它
                    if g >= old_open.n.g:
                        continue
                    node = replace(old_open.n, g=g, f=g + self.admissible_heuristic(neighbor), parent=min_focal_n)
                    self.replace_open_node(OpenLowNode(node))

                    #同样如果焦点集中有，也要更新
                    focal_index = find_index(self.focal_set,
                                             lambda n: n.n.state.x == neighbor.x and n.n.state.y == neighbor.y)
                    if focal_index >= 0:
                        self.focal_set[focal_index] = FocalLowNode(node)
                        heapq.heapify(self.focal_set)  #重建 focal set
                else:
                    # 加到open
                    heapq.heappush(self.open_set, OpenLowNode(node))

                    # 如果符合约束标准，则添加到焦点集
                    if f <= min_f * self.ctx.w:
                        heapq.heappush(self.focal_set, FocalLowNode(node))

        return TargetOnePlanResult(
            self.ctx.robotName,
            False,
            "找不到路径",
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedCount,
            fromState=self.start_state,
            toState=self.goal_state,
        )

    def admissible_heuristic(self, from_state: State) -> float:
        """
        两点之间的直线路径
        vs 欧氏距离
        TODO 未考虑旋转
        """
        return (float(abs(from_state.x - self.goal_state.x) + abs(from_state.y - self.goal_state.y))
                / self.ctx.moveUnitCost)

    def focal_state_heuristic(self, to_state: State, to_state_g: float) -> float:
        """
        故意 inadmissible 的启发式
        """
        if self.ctx.focalStateHeuristic is not None:
            return self.ctx.focalStateHeuristic(to_state, to_state_g)
        else:
            return to_state_g

    def focal_transition_heuristic(self, from_state: State, to_state: State,
                                   from_state_g: float, to_state_g: float) -> float:
        if self.ctx.focalTransitionHeuristic is not None:
            return self.ctx.focalTransitionHeuristic(from_state, to_state, from_state_g, to_state_g)
        else:
            return to_state_g - from_state_g

    def get_neighbors(self, from_state: State) -> list[State]:
        neighbors = []
        self.add_valid_neighbor(neighbors, from_state, 0, 0, from_state.head)  # waiting
        self.add_valid_neighbor(neighbors, from_state, 1, 0, 0)
        self.add_valid_neighbor(neighbors, from_state, -1, 0, 180)
        self.add_valid_neighbor(neighbors, from_state, 0, 1, 90)
        self.add_valid_neighbor(neighbors, from_state, 0, -1, 270)
        return neighbors

    def add_valid_neighbor(self, neighbors: list[State], from_state: State, dx: int, dy: int, to_head: int):
        """
        toHead 目标车头朝向
        """
        x = from_state.x + dx
        y = from_state.y + dy

        if (x < 0 or x >= self.ctx.mapDimX or y < 0 or y >= self.ctx.mapDimY
                or self.state_to_index(x, y) in self.ctx.obstacles):
            return

        # 需要转的角度，初始，-270 ~ +270
        d_head = abs(to_head - from_state.head)

        # 270 改成 90
        if d_head > 180:
            d_head = 360 - d_head  # 改角度

        d_head /= 90

        # 耗时，也作为 g 的增量
        # 假设 dx/dy 1 是 1 米
        time_num = ceil(float(abs(dx + dy)) / self.ctx.moveUnitCost + float(d_head) / self.ctx.rotateUnitCost)

        if time_num < 1:
            time_num = 1  # 原地等待

        new_state = State(x=x,
                          y=y,
                          head=to_head,
                          timeStart=from_state.timeEnd + 1,
                          timeEnd=from_state.timeEnd + time_num,
                          timeNum=time_num)

        # 最后一步要等待
        test_state = new_state
        if self.goal_state.is_same_location(new_state):
            # noinspection PyTypeChecker
            test_state = replace(new_state,
                                 timeEnd=new_state.timeEnd + self.ctx.goalStopTimeNum,
                                 timeNum=new_state.timeNum + self.ctx.goalStopTimeNum)

        if self.ctx.neighborValidator and not self.ctx.neighborValidator(from_state, test_state):
            return

        neighbors.append(new_state)

    def state_to_index(self, x: int, y: int):
        return x + y * self.ctx.mapDimX

    def remove_open_node(self, node: LowNode):
        s = node.state
        index = find_index(self.open_set, lambda n: n.n.state.x == s.x and n.n.state.y == s.y)
        if index < 0:
            return
        else:
            # 更快删除
            if index == 0:
                # 是root直接弹出
                heapq.heappop(self.open_set)
            else:
                # 不是则替换最后一个
                self.open_set[index] = self.open_set[-1]
                self.open_set.pop()

                if index < len(self.open_set):
                    #没到终点
                    self._sift_down(self.open_set, 0, index)

    def _sift_down(self, heap, startpos, pos):
        """
        替换元素后恢复堆不变
        """
        newitem = heap[pos]
        # 把较小的项向上冒泡，直到到根部
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if newitem < parent:
                heap[pos] = parent
                pos = parentpos
                continue
            break
        heap[pos] = newitem

    def replace_open_node(self, new_one: OpenLowNode):
        s = new_one.n.state
        index = find_index(self.open_set, lambda n: n.n.state.x == s.x and n.n.state.y == s.y)
        if index < 0:
            heapq.heappush(self.open_set, new_one)
        else:
            # 还原堆属性
            self.open_set[index] = new_one

            # 确保两个方向的 heap 属性
            self._sift_down(self.open_set, 0, index)
            heapq._siftup(self.open_set, index)  #使用 heapq 的内部_siftup

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

        action_state = replace(last_state,
                               timeStart=last_state.timeEnd + 1,
                               timeEnd=last_state.timeEnd + goal_stop_time_num,
                               timeNum=goal_stop_time_num)
        path.append(action_state)

        return TargetOnePlanResult(
            self.ctx.robotName,
            True,
            cost=min_focal_n.g + goal_stop_time_num,
            minF=min_focal_n.f + goal_stop_time_num,
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedCount,
            timeNum=action_state.timeEnd,
            timeStart=self.time_offset,
            timeEnd=action_state.timeEnd,
            fromState=self.start_state,
            toState=self.goal_state,
            path=path)
