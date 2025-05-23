import heapq
import logging
import os
import shutil
import time
from dataclasses import replace
from typing import Optional

from pydash import find_index

from src.common import cell_to_state, robots_solution_to_robots_paths
from src.conflicts import count_conflicts_find_first_constraints
from src.domain import RobotTask, MapfResult, TargetManyPlanResult, State, RobotConstraints, \
    HighNode, OpenHighNode, FocalHighNode, LowContext, \
    TargetOnePlanResult, Constraint, VertexConstraint, EdgeConstraint
from src.low_resolver import LowResolver


class HighResolver:

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
        logging.info(f"High resolver start: {self.tasks}")

        op_dir = "op"
        abs_op_dir = os.path.abspath(op_dir)
        logging.info(f"High resolver on dir: {abs_op_dir}")
        # 删除 on 目录
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

        while open_set:
            if time.time() - self.start_on > 60:
                return MapfResult(
                    ok=False,
                    msg="Timeout",
                    plans={},
                    timeCost=time.time() - self.start_on,
                )

            # 重建 focal 集合
            best_cost = open_set[0].n.cost
            focal_set = []
            bound = best_cost * self.w
            for on in open_set:
                if on.n.cost <= bound:  # 机器人执行成本:
                    heapq.heappush(focal_set, FocalHighNode(on.n))

            high_node = heapq.heappop(focal_set).n
            self.remove_open_node(open_set, high_node)
            self.high_node_expanded += 1
            logging.info(f"High node expanded: {self.high_node_expanded}: {high_node}")

            if not high_node.firstConstraints:
                return MapfResult(
                    ok=True,
                    plans=high_node.solution,
                    timeCost=time.time() - self.start_on)

            for robot_name, c in high_node.firstConstraints.constraints.items():
                child_node = self.build_child_hl_node(high_node, robot_name, c)
                if not child_node:
                    continue
                heapq.heappush(open_set, OpenHighNode(child_node))

        return MapfResult(
            ok=False,
            msg="All high states expanded",
            plans={},
            timeCost=time.time() - self.start_on,
        )

    def build_root_hl_node(self) -> Optional[HighNode]:
        solution: dict[str, TargetManyPlanResult] = {}
        cost = 0.0

        self.low_id = 0
        for (robot_name, task) in self.tasks.items():
            fa_ctx = LowContext(
                robotName=robot_name,
                highId=self.high_id,
                w=self.w,
                mapDimX=self.map_dim_x,
                mapDimY=self.map_dim_y,
                obstacles=self.obstacles,
                moveUnitCost=1.0,
                rotateUnitCost=1.0,
                goalStopTimeNum=task.stopTimes,
                constraints=RobotConstraints(robot_name),
                oldAllPaths={}
            )
            rs = self.fa_search_many_targets(fa_ctx, cell_to_state(task.fromState),
                                             [cell_to_state(s) for s in task.toStates])
            self.low_node_expanded += rs.expandedCount

            if not rs.ok:
                print(f"Robot {rs.robotName} no init solution")
                return None

            solution[robot_name] = rs
            cost += rs.cost

        conflicts_count, first_constraints = count_conflicts_find_first_constraints(
            robots_solution_to_robots_paths(solution))

        # 确保每个机器人都有一个空约束
        constraints_map: dict[str, RobotConstraints] = {}
        for robot_name in self.tasks.keys():
            constraints_map[robot_name] = RobotConstraints(robot_name)

        return HighNode(
            id=self.high_id,
            parentId=-1,
            solution=solution,
            constraints=constraints_map,
            cost=cost,
            conflictsCount=conflicts_count,
            firstConstraints=first_constraints,
        )

    def build_child_hl_node(self, parent: HighNode, robot_name: str, constraint: Constraint) -> Optional[HighNode]:
        child_constraints = self.add_constraints(parent, robot_name, constraint)
        robot_constraints = child_constraints[robot_name]
        solution = parent.solution.copy()

        all_paths = robots_solution_to_robots_paths(solution)

        cost = parent.cost - solution[robot_name].cost

        self.high_id += 1
        child_node_id = self.high_id

        self.low_id = 0

        task = self.tasks[robot_name]
        fa_ctx = LowContext(
            robotName=robot_name,
            highId=child_node_id,
            w=self.w,
            mapDimX=self.map_dim_x,
            mapDimY=self.map_dim_y,
            obstacles=self.obstacles,
            moveUnitCost=1.0,
            rotateUnitCost=1.0,
            goalStopTimeNum=task.stopTimes,
            constraints=robot_constraints,
            oldAllPaths=all_paths,
        )
        rs = self.fa_search_many_targets(fa_ctx, cell_to_state(task.fromState),
                                         [cell_to_state(it) for it in task.toStates])
        self.low_node_expanded += rs.expandedCount
        if not rs.ok:
            logging.info(f"Failed to build child high node, no solution for robot {robot_name}")
            return None
        solution[robot_name] = rs

        cost += rs.cost

        conflicts_count, first_constraints = count_conflicts_find_first_constraints(
            robots_solution_to_robots_paths(solution))

        # 检查是否遵守了约束
        first_constraint: Optional[Constraint] = first_constraints and first_constraints.constraints.get(robot_name)
        if first_constraint:
            if isinstance(first_constraint, VertexConstraint):
                v_constraints = robot_constraints.vertexConstraints
                if first_constraint in v_constraints:
                    raise Exception(f"Constraints already exist for robot {robot_name}, high={child_node_id}: "
                                    f"{constraint} in {v_constraints}")
            elif isinstance(first_constraint, EdgeConstraint):
                e_constraints = robot_constraints.edgeConstraints
                if first_constraint in e_constraints:
                    raise Exception(
                        f"Constraints already exist for robot {robot_name}, high={child_node_id}: "
                        f"{constraint} in {e_constraints}")

        return HighNode(
            id=child_node_id,
            parentId=parent.id,
            constraints=child_constraints,
            solution=solution,
            cost=cost,
            conflictsCount=conflicts_count,
            firstConstraints=first_constraints
        )

    @staticmethod
    def add_constraints(node: HighNode, robot_name: str, constraint: Constraint) -> dict[str, RobotConstraints]:
        constraints = node.constraints.copy()  # 继承父节点的约束
        robot_constraints = constraints[robot_name]

        if isinstance(constraint, VertexConstraint):
            v_constraints = robot_constraints.vertexConstraints.copy()
            if constraint in v_constraints:
                raise Exception(f"Constraints already exist for robot {robot_name}: {constraint}in {v_constraints}")
            # noinspection PyTypeChecker
            v_constraints.add(constraint)
            new_robot_constraints = replace(robot_constraints, vertexConstraints=v_constraints)
            constraints[robot_name] = new_robot_constraints
        elif isinstance(constraint, EdgeConstraint):
            e_constraints = robot_constraints.edgeConstraints.copy()
            if constraint in e_constraints:
                raise Exception(f"Constraints already exist for robot {robot_name}: {constraint} in {e_constraints}")
            # noinspection PyTypeChecker
            e_constraints.add(constraint)
            new_robot_constraints = replace(robot_constraints, edgeConstraints=e_constraints)
            constraints[robot_name] = new_robot_constraints

        return constraints

    def fa_search_many_targets(self, ctx: LowContext, start_state: State,
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
        time_num = 0

        for ti, goal_state in enumerate(goal_states):
            low_id = self.low_id
            self.low_id += 1
            one = LowResolver(ctx, low_id, time_num, from_state, goal_state)
            sr = one.resolve_one()
            expanded_count += sr.expandedCount

            if not sr.ok:
                ok = False
                reason = sr.reason
                break

            steps.append(sr)
            path.extend(sr.path)

            cost += sr.cost
            time_num += sr.timeNum

            from_state = goal_state

        return TargetManyPlanResult(
            ctx.robotName,
            ok,
            reason,
            cost=cost,
            expandedCount=expanded_count,
            planCost=time.time() - start_on,
            timeNum=time_num, timeStart=0, timeEnd=time_num,
            steps=steps, path=path,
        )

    @staticmethod
    def remove_open_node(open_set: list[OpenHighNode], node: HighNode):
        index = find_index(open_set, lambda n: n.n.id == node.id)
        if index < 0:
            return
        else:
            # 把最后一个填充过来
            open_set[index] = open_set[-1]
            open_set.pop()
            # 必须重建！o(n)
            heapq.heapify(open_set)

    @staticmethod
    def constraints_to_last_goal_constraint(constraints: RobotConstraints, goal: State) -> int:
        """
        如果目标位置被约束了，等约束结束后，机器人才能到达目标
        """
        last_goal_constraint = -1
        for vc in constraints.vertexConstraints:
            if vc.x == goal.x and vc.y == goal.y:
                last_goal_constraint = max(last_goal_constraint, vc.timeEnd)
        return last_goal_constraint
