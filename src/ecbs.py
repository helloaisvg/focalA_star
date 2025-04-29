import heapq
import logging
import os
import shutil
import time
from dataclasses import replace
from typing import Optional

from pydash import remove, find

from src.domain import RobotTask, MapfResult, TargetManyPlanResult, State, RobotConstraints as Constraints, \
    VertexConstraint, EdgeConstraint, HighNode, OpenHighNode, FocalHighNode, FaContext, \
    TargetOnePlanResult
from src.common import cell_to_state, is_time_overlay
from src.fa_one import FaOne


class ECBS:

    def __init__(self, w: float, map_dim_x: int, map_dim_y: int, obstacles: set[int], tasks: dict[str, RobotTask]):
        """
        :param w:
        :param map_dim_x:
        :param map_dim_y: y 向下为正
        :param obstacles:
        :param tasks: by robot
        """
        self.w = w
        self.map_dim_x = map_dim_x
        self.map_dim_y = map_dim_y
        self.obstacles = obstacles
        self.tasks = tasks

        # TODO 它们会被并发访问
        self.high_id = 0
        self.low_id = 0
        self.high_node_expanded = 0
        self.low_node_expanded = 0
        self.start_on = time.time()

    def search(self) -> MapfResult:
        logging.info(f"ECBS start: {self.tasks}")

        op_dir = "op"
        abs_op_dir = os.path.abspath(op_dir)
        logging.info(f"ECBS op dir: {abs_op_dir}")
        # 删除 op 目录
        shutil.rmtree(op_dir, ignore_errors=True)
        # 重新创建目录
        os.makedirs(op_dir)

        root_node = self.build_root_hl_node()
        if root_node is None:
            return MapfResult(
                ok=False,
                plans={},
                timeCost=time.time() - self.start_on,
            )

        open_set: list[OpenHighNode] = []
        focal_set: list[FocalHighNode] = []

        heapq.heappush(open_set, OpenHighNode(root_node))
        heapq.heappush(focal_set, FocalHighNode(root_node))

        best_cost = root_node.cost

        while open_set:
            if time.time() - self.start_on > 5:
                MapfResult(
                    ok=False,
                    msg="Timeout",
                    plans={},
                    timeCost=time.time() - self.start_on,
                )
            old_best_cost = best_cost
            best_cost = open_set[0].n.cost
            if best_cost > old_best_cost:
                for n in open_set:
                    if old_best_cost * self.w < n.n.cost <= best_cost * self.w:
                        heapq.heappush(focal_set, FocalHighNode(n.n))

            p = heapq.heappop(focal_set)
            remove(open_set, lambda item: item.n.id == p.n.id)
            self.high_node_expanded += 1

            constraints = self.get_first_conflict_constraints(p.n.solution)
            if constraints is None:
                return MapfResult(
                    ok=True,
                    plans=p.n.solution,
                    timeCost=time.time() - self.start_on)

            for robot_name, c in constraints.items():
                child_node = self.build_child_hl_node(p.n, robot_name, c)
                if not child_node:
                    continue
                heapq.heappush(open_set, OpenHighNode(child_node))
                if child_node.cost <= best_cost * self.w:
                    heapq.heappush(focal_set, FocalHighNode(child_node))

        return MapfResult(
            ok=False,
            msg="All states expanded",
            plans={},
            timeCost=time.time() - self.start_on,
        )

    def build_root_hl_node(self) -> Optional[HighNode]:
        solution: dict[str, TargetManyPlanResult] = {}
        cost = 0.0
        lb = 0.0

        self.low_id = 0
        for (robot_name, task) in self.tasks.items():
            fa_ctx = FaContext(
                robotName=robot_name,
                highId=self.high_id,
                w=self.w,
                mapDimX=self.map_dim_x,
                mapDimY=self.map_dim_y,
                obstacles=self.obstacles,
                moveUnitCost=1.0,
                rotateUnitCost=1.0,
                goalStopTimeNum=task.stopTimes,
            )
            rs = self.fa_search_many_targets(fa_ctx, cell_to_state(task.fromState),
                                             [cell_to_state(s) for s in task.toStates])
            self.low_node_expanded += rs.expandedCount

            if not rs.ok:
                print(f"Robot {rs.robotName} no init solution")
                return None

            solution[robot_name] = rs
            cost += rs.cost
            lb += rs.minF

        focal_heuristic = self.focal_heuristic(solution)

        # 确保每个机器人都有一个空约束
        constraints_map: dict[str, Constraints] = {}
        for robot_name in self.tasks.keys():
            constraints_map[robot_name] = Constraints(robot_name, vertexConstraints=[], edgeConstraints=[])

        return HighNode(
            id=self.high_id,
            parentId=-1,
            solution=solution,
            constraints=constraints_map,
            cost=cost,
            lb=lb,
            focalHeuristic=focal_heuristic,
        )

    def build_child_hl_node(self, parent: HighNode, robot_name: str,
                            new_constraints: Constraints) -> Optional[HighNode]:
        # robotName 施加约束的机器人
        child_constraints = self.add_constraints(parent, robot_name, new_constraints)
        robot_constraints = child_constraints[robot_name]
        solution = parent.solution.copy()

        cost = parent.cost - solution[robot_name].cost
        lb = parent.lb - solution[robot_name].minF

        child_node_id = self.high_id
        self.high_id += 1

        self.low_id = 0

        task = self.tasks[robot_name]
        fa_ctx = FaContext(
            robotName=robot_name,
            highId=child_node_id,
            w=self.w,
            mapDimX=self.map_dim_x,
            mapDimY=self.map_dim_y,
            obstacles=self.obstacles,
            moveUnitCost=1.0,
            rotateUnitCost=1.0,
            goalStopTimeNum=task.stopTimes,
            neighborValidator=lambda from_, to_: (self.state_valid(to_, robot_constraints) and
                                                  self.transition_valid(from_, to_, robot_constraints)),
            # 这里是子节点的 solution，不能是父节点的
            focalStateHeuristic=lambda s, _: self.focal_state_heuristic(s, robot_name, solution),
            # 这里是子节点的 solution，不能是父节点的
            focalTransitionHeuristic=lambda s1a, s1b, _fg, _tg: self.focal_transition_heuristic(s1a, s1b, robot_name,
                                                                                                solution)
        )
        rs = self.fa_search_many_targets(fa_ctx, cell_to_state(task.fromState),
                                         [cell_to_state(it) for it in task.toStates])
        self.low_node_expanded += rs.expandedCount
        if not rs.ok:
            return None
        solution[robot_name] = rs

        cost += rs.cost
        lb += rs.minF
        focal_heuristic = self.focal_heuristic(solution)

        return HighNode(
            id=child_node_id,
            parentId=parent.id,
            constraints=child_constraints,
            solution=solution,
            cost=cost,
            lb=lb,
            focalHeuristic=focal_heuristic,
        )

    @staticmethod
    def add_constraints(parent: HighNode, robot_name: str,
                        new_constraints: Constraints) -> dict[str, Constraints]:
        child_constraints = parent.constraints.copy()  # 继承父节点的约束
        old_robot_constraints = child_constraints[robot_name]
        # if new_constraints in old_robot_constraints:
        #     raise Exception("Constraints already exist for robot " + robot_name)
        vertex_constraints = old_robot_constraints.vertexConstraints + new_constraints.vertexConstraints
        edge_constraints = old_robot_constraints.edgeConstraints + new_constraints.edgeConstraints
        new_robot_constraints = replace(old_robot_constraints,
                                        vertexConstraints=vertex_constraints,
                                        edgeConstraints=edge_constraints)
        child_constraints[robot_name] = new_robot_constraints
        return child_constraints

    @staticmethod
    def get_first_conflict_constraints(solution: dict[str, TargetManyPlanResult]) -> Optional[dict[str, Constraints]]:
        # 返回的 map 有两个元素。每个表示参与约束的一个机器人
        constraints: dict[str, Constraints] = {}
        robot_names = list(solution.keys())

        for r1i in range(len(robot_names) - 1):
            robot_name1 = robot_names[r1i]
            sol1 = solution[robot_name1]
            for r2i in range(r1i + 1, len(robot_names)):
                robot_name2 = robot_names[r2i]
                sol2 = solution[robot_name2]
                # 顶点约束
                # 遍历两个机器人的每个状态，如果位置相同且时间交叠
                for s1 in sol1.path:
                    for s2 in sol2.path:
                        if (s1.is_same_location(s2) and
                                is_time_overlay(s1.timeStart, s1.timeEnd, s2.timeStart, s2.timeEnd)):
                            # 注意用另一个机器人的时间
                            constraints[robot_name1] = Constraints(robot_name1, vertexConstraints=[
                                VertexConstraint(s1.x, s1.y, s2.timeStart, s2.timeEnd)], edgeConstraints=[])
                            constraints[robot_name2] = Constraints(robot_name2, vertexConstraints=[
                                VertexConstraint(s2.x, s2.y, s1.timeStart, s1.timeEnd)], edgeConstraints=[])
                            return constraints
                # 边约束
                # 遍历两个机器人的所有移动 i -> i+1
                # 时间范围取占起点的开始时间到终点的结束时间
                for s1i in range(len(sol1.path) - 1):
                    s1a = sol1.path[s1i]
                    s1b = sol1.path[s1i + 1]
                    if s1a.is_same_location(s1b): continue
                    for s2i in range(len(sol2.path) - 1):
                        s2a = sol2.path[s2i]
                        s2b = sol2.path[s2i + 1]
                        if s2a.is_same_location(s2b): continue
                        if s1a.is_same_location(s2b) and s1b.is_same_location(s2a) and s1b.is_time_overlay(s2b):
                            # 注意用另一个机器人的时间
                            constraints[robot_name1] = Constraints(robot_name1, vertexConstraints=[], edgeConstraints=[
                                EdgeConstraint(s1a.x, s1a.y, s1b.x, s1b.y, s2a.timeStart, s2b.timeEnd)])
                            constraints[robot_name2] = Constraints(robot_name2, vertexConstraints=[], edgeConstraints=[
                                EdgeConstraint(s2a.x, s2a.y, s2b.x, s2b.y, s1a.timeStart, s1b.timeEnd)])
                            return constraints

        return None

    def fa_search_many_targets(self, ctx: FaContext, start_state: State,
                               goal_states: list[State]) -> TargetManyPlanResult:
        """
        多目标搜索
        """
        start_on = time.time()

        from_state = start_state
        ok = True
        reason = ""

        steps: list[TargetOnePlanResult] = []
        path: list[State] = []

        expanded_count = 0
        cost = 0.0
        min_f = 0.0
        time_num = 0

        for ti, goal_state in enumerate(goal_states):
            low_id = self.low_id
            self.low_id += 1
            one = FaOne(ctx, low_id, time_num, from_state, goal_state)
            sr = one.search_one
            expanded_count += sr.expandedCount

            if not sr.ok:
                ok = False
                reason = sr.reason
                break

            steps.append(sr)
            path.extend(sr.path)

            cost += sr.cost
            min_f += sr.minF
            time_num += sr.timeNum

            from_state = goal_state

        return TargetManyPlanResult(
            ctx.robotName,
            ok,
            reason,
            cost=cost,
            minF=min_f,
            expandedCount=expanded_count,
            planCost=time.time() - start_on,
            timeNum=time_num, timeStart=0, timeEnd=time_num,
            steps=steps, path=path,
        )

    def state_valid(self, to_state: State, constraints: Constraints) -> bool:
        """
        一个智能体的。在边界内，不是障碍物，不与约束冲突。
        """
        v_constraints = constraints.vertexConstraints
        if not (0 <= to_state.x < self.map_dim_x and 0 <= to_state.y < self.map_dim_y and
                to_state.y * self.map_dim_x + to_state.x not in self.obstacles):
            return False
        c = find(v_constraints, lambda vc: to_state.is_same_location(vc) and to_state.is_time_overlay(vc))
        if c:
            return False
        return True

    @staticmethod
    def transition_valid(from_state: State, to_state: State, constraints: Constraints) -> bool:
        if from_state.is_same_location(to_state):
            return True
        e_constraints = constraints.edgeConstraints
        # 起点、终点位置相同
        c = find(e_constraints, lambda ec: (
                ec.x1 == from_state.x and ec.y1 == from_state.y and ec.x2 == to_state.x and ec.y2 == to_state.y and
                is_time_overlay(from_state.timeStart, to_state.timeEnd, ec.timeStart, ec.timeEnd)))
        if c:
            return False
        return True

    @staticmethod
    def constraints_to_last_goal_constraint(constraints: Constraints, goal: State) -> int:
        """
        如果目标位置被约束了，等约束结束后，机器人才能到达目标
        """
        last_goal_constraint = -1
        for vc in constraints.vertexConstraints:
            if vc.x == goal.x and vc.y == goal.y:
                last_goal_constraint = max(last_goal_constraint, vc.timeEnd)
        return last_goal_constraint

    def focal_heuristic(self, solution: dict[str, TargetManyPlanResult]) -> float:
        """
        Count all conflicts
        """
        num_conflicts = 0
        t_max = 0
        for sol in solution.values():
            t_max = max(t_max, len(sol.path) - 1)

        robot_names = list(solution.keys())
        for t in range(t_max):
            # check drive-drive vertex collisions
            for i in range(len(robot_names)):
                state1 = self.get_state_1(robot_names[i], solution, t)
                for j in range(i + 1, len(robot_names)):
                    state2 = self.get_state_1(robot_names[j], solution, t)
                    if state1.x == state2.x and state1.y == state2.y:
                        num_conflicts += 1
            # check drive-drive edge (swap)
            for i in range(len(robot_names)):
                state1a = self.get_state_1(robot_names[i], solution, t)
                state1b = self.get_state_1(robot_names[i], solution, t + 1)
                for j in range(i + 1, len(robot_names)):
                    state2a = self.get_state_1(robot_names[j], solution, t)
                    state2b = self.get_state_1(robot_names[j], solution, t + 1)
                    if (state1a.x == state2b.x and state1a.y == state2b.y and state1b.x == state2a.x
                            and state1b.y == state2a.y):
                        num_conflicts += 1

        return float(num_conflicts)

    @staticmethod
    def get_state_1(robot_name: str, solution: dict[str, TargetManyPlanResult], t: int) -> State:
        path = solution[robot_name].path
        return find(path, lambda p: p.timeStart >= t >= p.timeEnd) or path[-1]

    def focal_state_heuristic(self, s: State, from_robot: str, solution: dict[str, TargetManyPlanResult]) -> float:
        """
        low level state heuristic
        """
        num_conflicts = 0.0
        for robot_name, rs in solution.items():
            if robot_name != from_robot and rs is not None:
                s2 = self.get_state_2(robot_name, solution, s.timeStart, s.timeEnd)
                if s.x == s2.x and s.y == s2.y:
                    num_conflicts += 1

        return num_conflicts

    def focal_transition_heuristic(self, s1a: State, s1b: State, from_robot: str,
                                   solution: dict[str, TargetManyPlanResult]) -> float:
        """
        low level
        """
        num_conflicts = 0.0
        for robot_name, rs in solution.items():
            if robot_name != from_robot and rs is not None:
                s2a = self.get_state_2(robot_name, solution, s1a.timeStart, s1a.timeEnd)
                s2b = self.get_state_2(robot_name, solution, s1b.timeStart, s1b.timeEnd)
                if s1a.x == s2b.x and s1a.y == s2b.y and s1b.x == s2a.x and s1b.y == s2a.y:
                    num_conflicts += 1
        return num_conflicts

    @staticmethod
    def get_state_2(robot_name: str, solution: dict[str, TargetManyPlanResult], t_start: int, t_end: int) -> State:
        """
        相交的
        """
        path = solution[robot_name].path
        return find(path, lambda p: p.timeStart <= t_end and p.timeEnd >= t_start) or path[-1]
