import heapq
import logging
import time
from dataclasses import dataclass, field, replace
from math import ceil
from typing import Dict, Tuple, Optional

from pydash import find, find_index

from src.domain import State, TargetOnePlanResult, LowNode, FaContext, FaOp
"""字典优化"""

@dataclass(order=False, frozen=True)
class OpenLowNode:
    """open节点，按 f 值排序"""
    n: LowNode = field(compare=False)

    # For heap comparison
    def __lt__(self, other):
        return self.n.f < other.n.f


@dataclass(order=False, frozen=True)
class FocalLowNode:
    """焦点集结点按f2排序"""
    n: LowNode = field(compare=False)

    # 比较堆
    def __lt__(self, other):
        return self.n.f2 < other.n.f2


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

        """key 为 cell_index，value为实际的OpenLowNode 对象"""
        self.open_dict: Dict[int, OpenLowNode] = {}

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

        open_node = OpenLowNode(start_node)
        heapq.heappush(self.open_set, open_node)
        self.open_dict[self.state_to_index(start_node.state.x, start_node.state.y)] = open_node

        heapq.heappush(self.focal_set, FocalLowNode(start_node))

        while self.open_set:
            min_f = self.open_set[0].n.f  # 读取最小的节点，但不删除

            self.focal_set = []
            bound = min_f * self.ctx.w
            for node in self.open_set:
                if node.n.f <= bound:
                    heapq.heappush(self.focal_set, FocalLowNode(node.n))

            # 取出下一个有界最优启发值最低的节点
            min_focal_n: LowNode = heapq.heappop(self.focal_set).n
            min_focal_s = min_focal_n.state

            self.op.expandedList.append(str(self.op.expandedCount) + "|" + min_focal_n.desc(self.ctx.mapDimX))
            self.op.expandedCount += 1
            self.op.openSize.append(len(self.open_set))
            self.op.focalSize.append(len(self.focal_set))

            # 从 open 中删除
            self.remove_open_node(min_focal_n)

            # 加入到 close
            self.closed_set[self.state_to_index(min_focal_s.x, min_focal_s.y)] = min_focal_n

            if self.op.expandedCount > self.ctx.mapDimX * self.ctx.mapDimY:
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
                node = LowNode(neighbor, min_focal_n, f=f, f2=focal_heuristic, g=g)

                cell_index = self.state_to_index(neighbor.x, neighbor.y)
                old_closed = self.closed_set.get(cell_index)
                if old_closed:
                    # 如果在 close 里
                    if f < old_closed.f:
                        self.op.warnings.append(f"SmallerFInClosed|{cell_index}|{neighbor.x},{neighbor.y}|"
                                                f"{neighbor.timeStart}:{neighbor.timeEnd}|"
                                                f"{old_closed.state.timeStart}:{old_closed.state.timeEnd}|"
                                                f"{f}|{old_closed.f}")
                        del self.closed_set[cell_index]
                    else:
                        continue

                #使用字典检查是否已经处于 open set 中
                old_open = self.open_dict.get(cell_index)
                if old_open:
                    # 如果在 open 里，如果 g 变小了，更新
                    if g >= old_open.n.g:
                        continue
                    new_node = replace(old_open.n, g=g, f=old_open.n.f + g - old_open.n.g)
                    new_open_node = OpenLowNode(new_node)
                    self.replace_open_node(new_open_node)
                else:
                    new_open_node = OpenLowNode(node)
                    heapq.heappush(self.open_set, new_open_node)
                    self.open_dict[cell_index] = new_open_node

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
            d_head = 90

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
        """
        使用字典进行 O（1） 查找并从open中删除节点
        """
        s = node.state
        cell_index = self.state_to_index(s.x, s.y)

        # 检查节点是否在opendict
        if cell_index not in self.open_dict:
            return

        # 在open找节点
        target_node = self.open_dict[cell_index]

        #从字典中删除
        del self.open_dict[cell_index]

        #在堆中找索引
        idx = None
        for i, open_node in enumerate(self.open_set):
            if open_node.n.state.x == s.x and open_node.n.state.y == s.y:
                idx = i
                break

        if idx is not None:
            # 和最后一个元素交换
            self.open_set[idx] = self.open_set[-1]
            self.open_set.pop()
            #重建堆
            heapq.heapify(self.open_set)

    def replace_open_node(self, new_one: OpenLowNode):
        """
        将打开集中的节点替换为具有相同位置的新节点
        """
        s = new_one.n.state
        cell_index = self.state_to_index(s.x, s.y)

        # 先检查节点是不是在opendict中
        if cell_index not in self.open_dict:
            # Add
            heapq.heappush(self.open_set, new_one)
            self.open_dict[cell_index] = new_one
            return

        # open set找节点
        idx = None
        for i, open_node in enumerate(self.open_set):
            if open_node.n.state.x == s.x and open_node.n.state.y == s.y:
                idx = i
                break

        if idx is not None:
            # 替换
            self.open_set[idx] = new_one
            # 跟新字典
            self.open_dict[cell_index] = new_one
            #重建堆
            heapq.heapify(self.open_set)
    """ else:
            heapq.heappush(self.open_set, new_one)
            self.open_dict[cell_index] = new_one"""

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
            minF=min_focal_n.f + goal_stop_time_num,
            planCost=time.time() - self.op.startedOn,
            expandedCount=self.op.expandedCount,
            timeNum=action_state.timeEnd,
            timeStart=self.time_offset,
            timeEnd=action_state.timeEnd,
            fromState=self.start_state,
            toState=self.goal_state,
            path=path)
