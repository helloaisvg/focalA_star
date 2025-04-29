import heapq
import logging
import time
from dataclasses import dataclass, field, replace
from math import ceil
from typing import Dict, Tuple, Optional

from pydash import find, find_index

from src.domain import State, TargetOnePlanResult, FaContext, FaOp


@dataclass(order=False, frozen=True)
class OpenLowNode:
    """open节点，按 f 值排序"""
    n: 'LowNode' = field(compare=False)

    # For heap comparison
    def __lt__(self, other):
        return self.n.f < other.n.f


@dataclass(order=False, frozen=True)
class FocalLowNode:
    """焦点集结点按f2排序"""
    n: 'LowNode' = field(compare=False)

    # 比较堆
    def __lt__(self, other):
        return self.n.f2 < other.n.f2


@dataclass
class LowNode:
    """
    低层搜索节点
    """
    state: State
    parent: Optional['LowNode'] = None
    g: float = 0.0  # 实际成本
    f: float = 0.0  # 估计总成本 (g + h)
    f2: float = 0.0  # 用于Focal搜索的值
    id: int = 0

    def desc(self, map_dim_x: int) -> str:
        """节点描述信息"""
        return (f"{self.state.x + self.state.y * map_dim_x}|"
                f"{self.state.x},{self.state.y}|"
                f"{self.state.timeStart}:{self.state.timeEnd}|"
                f"{self.g:.1f}|{self.f:.1f}|{self.f2:.1f}")


class FaOne:
    """
    单目标
    有界最优，Focal Search A*
    f, g, h 单位是时间，表示时间成本

    优化整合：
    1. 焦点集增量更新策略 - 只在min_f更改时更新focal_set
    2. 字典优化 - 使用open_dict实现O(1)时间复杂度的查找
    3. 启发函数缓存 - 避免重复计算启发函数
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

        # 添加缓存字典，用于存储启发式函数的计算结果
        self.heuristic_cache = {}
        self.focal_state_heuristic_cache = {}
        self.focal_transition_heuristic_cache = {}

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

        # 字典，用于O(1)时间复杂度查找open_set中的节点
        self.open_dict: Dict[int, OpenLowNode] = {}

        # 字典，用于存储每个单元格位置的最佳g值
        self.best_g_values: dict[int, float] = {}

    @property
    def search_one(self):
        r = self.do_search_one()

        op = self.op
        op.endedOn = time.time()
        op.timeCost = op.endedOn - op.startedOn
        op.ok = r.ok
        op.errMsg = getattr(r, 'errMsg', '')
        op.path = r.path

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
            state=replace(self.start_state, timeStart=self.time_offset, timeEnd=self.time_offset, timeNum=1),
            parent=None,
            f=self.admissible_heuristic(self.start_state),
            g=0.0,
            f2=0.0,
            id=0
        )

        # 将起始节点添加到open_set和open_dict
        open_node = OpenLowNode(start_node)
        heapq.heappush(self.open_set, open_node)
        start_cell_index = self.state_to_index(start_node.state.x, start_node.state.y)
        self.open_dict[start_cell_index] = open_node

        # 初始化起始位置的最佳g值
        self.best_g_values[start_cell_index] = 0.0

        # 添加到focal_set
        heapq.heappush(self.focal_set, FocalLowNode(start_node))

        # 跟踪open中的当前最小f值
        min_f = start_node.f

        # 跟踪最后绑定以优化焦点集更新
        last_bound = min_f * self.ctx.w

        node_id = 1

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

            # traverse neighbors
            if not min_focal_s.is_same_location(self.goal_state) or min_focal_s.timeEnd <= self.last_goal_constraint:
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

                    # 更新 best g-value
                    self.best_g_values[cell_index] = g

                    node = LowNode(state=neighbor, parent=min_focal_n, f=f, f2=focal_heuristic, g=g, id=node_id)
                    node_id += 1

                    # 检查是否在closed中
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

                    # 检查是否在open中，使用字典实现O(1)时间复杂度查找
                    old_open = self.open_dict.get(cell_index)
                    if old_open:
                        # 如果在g值较差的open中，更新它
                        if g >= old_open.n.g:
                            continue

                        # 创建更新后的节点
                        updated_node = replace(old_open.n, g=g, f=f, parent=min_focal_n, f2=focal_heuristic)
                        new_open_node = OpenLowNode(updated_node)

                        # 替换open_set中的节点
                        self.replace_open_node(new_open_node)

                        # 同样如果焦点集中有，也要更新
                        focal_index = find_index(self.focal_set,
                                                 lambda n: n.n.state.x == neighbor.x and n.n.state.y == neighbor.y)
                        if focal_index >= 0:
                            self.focal_set[focal_index] = FocalLowNode(updated_node)
                            heapq.heapify(self.focal_set)  # 重建 focal set
                    else:
                        # 加到open
                        new_open_node = OpenLowNode(node)
                        heapq.heappush(self.open_set, new_open_node)
                        self.open_dict[cell_index] = new_open_node

                        # 如果符合约束标准，则添加到焦点集
                        if f <= min_f * self.ctx.w:
                            heapq.heappush(self.focal_set, FocalLowNode(node))
                continue

            return self.build_ok_result(min_focal_n)

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
        使用缓存避免重复计算
        """
        # 创建基于位置的缓存键
        cache_key = (from_state.x, from_state.y)

        # 检查缓存中是否已存在该计算结果
        if cache_key in self.heuristic_cache:
            return self.heuristic_cache[cache_key]

        # 计算启发式值
        h_value = (float(abs(from_state.x - self.goal_state.x) + abs(from_state.y - self.goal_state.y))
                   / self.ctx.moveUnitCost)

        # 存入缓存
        self.heuristic_cache[cache_key] = h_value

        return h_value

    def focal_state_heuristic(self, to_state: State, to_state_g: float) -> float:
        """
        故意 inadmissible 的启发式
        使用缓存优化
        """
        if self.ctx.focalStateHeuristic is None:
            return to_state_g

        # 创建基于位置和g值的缓存键
        cache_key = (to_state.x, to_state.y, to_state_g)

        # 检查缓存中是否已存在该计算结果
        if cache_key in self.focal_state_heuristic_cache:
            return self.focal_state_heuristic_cache[cache_key]

        # 计算启发式值
        h_value = self.ctx.focalStateHeuristic(to_state, to_state_g)

        # 存入缓存
        self.focal_state_heuristic_cache[cache_key] = h_value

        return h_value

    def focal_transition_heuristic(self, from_state: State, to_state: State,
                                   from_state_g: float, to_state_g: float) -> float:
        """
        使用缓存优化
        """
        if self.ctx.focalTransitionHeuristic is None:
            return to_state_g - from_state_g

        # 创建基于位置和g值的缓存键
        cache_key = (from_state.x, from_state.y, to_state.x, to_state.y, from_state_g, to_state_g)

        # 检查缓存中是否已存在该计算结果
        if cache_key in self.focal_transition_heuristic_cache:
            return self.focal_transition_heuristic_cache[cache_key]

        # 计算启发式值
        h_value = self.ctx.focalTransitionHeuristic(from_state, to_state, from_state_g, to_state_g)

        # 存入缓存
        self.focal_transition_heuristic_cache[cache_key] = h_value

        return h_value

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
            d_head = 360 - d_head  # 使用正确的角度计算逻辑

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
        使用字典进行 O(1) 查找并从open中删除节点
        """
        s = node.state
        cell_index = self.state_to_index(s.x, s.y)

        # 检查节点是否在open_dict中
        if cell_index not in self.open_dict:
            return

        # 从字典中删除
        del self.open_dict[cell_index]

        # 在堆中找到索引
        idx = None
        for i, open_node in enumerate(self.open_set):
            if open_node.n.state.x == s.x and open_node.n.state.y == s.y:
                idx = i
                break

        if idx is not None:
            # 实现更高效的删除方法
            if idx == 0:
                # 如果是堆顶元素，直接弹出
                heapq.heappop(self.open_set)
            else:
                # 用最后一个元素替换要删除的元素
                self.open_set[idx] = self.open_set[-1]
                self.open_set.pop()
                # 维护堆属性
                if idx < len(self.open_set):
                    self._sift_down(self.open_set, 0, idx)

    def _sift_down(self, heap, startpos, pos):
        """
        替换元素后恢复堆不变性
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
        """
        将open_set中的节点替换为具有相同位置的新节点
        """
        s = new_one.n.state
        cell_index = self.state_to_index(s.x, s.y)

        # 先检查节点是否在open_dict中
        if cell_index not in self.open_dict:
            # 如果不在，添加到open_set和open_dict
            heapq.heappush(self.open_set, new_one)
            self.open_dict[cell_index] = new_one
            return

        # 在open_set中找到节点
        idx = None
        for i, open_node in enumerate(self.open_set):
            if open_node.n.state.x == s.x and open_node.n.state.y == s.y:
                idx = i
                break

        if idx is not None:
            # 替换节点
            self.open_set[idx] = new_one
            # 更新字典
            self.open_dict[cell_index] = new_one
            # 维护堆属性
            self._sift_down(self.open_set, 0, idx)
            heapq._siftup(self.open_set, idx)  # 使用heapq的内部_siftup方法

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
